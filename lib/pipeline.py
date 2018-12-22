import logging as log

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVR

from lib.converters import SalePriceConverter
from lib.ioutilites import read_json

log.getLogger().setLevel(log.INFO)


def load_data():
    train_csv = pd.read_csv("../dataset/train.csv")
    test_csv = pd.read_csv("../dataset/test.csv")
    train = train_csv.drop(["Id"], axis=1)
    submission = test_csv.drop(["Id"], axis=1)
    return train, submission


def isnan(value):
    return isinstance(value, float) and np.isnan(value)


def constant(new_value):
    return lambda v: isnan_to(v, new_value)


def isnan_to(value, new_value):
    if isnan(value):
        return new_value
    else:
        return value


def fill_nans(train: pd.DataFrame, converters) -> pd.DataFrame:
    train_copy = train.copy()
    for index, row in train_copy.iterrows():
        for feature in row.keys():
            if feature in converters:
                new_value = converters[feature](row[feature])
                train_copy.at[index, feature] = new_value
    return train_copy


def preprocess_data(data: pd.DataFrame):
    explicit_converters = {
        "LotFrontage": constant(0.0),
        "MasVnrArea": constant(0.0),
        "Alley": constant("NA"),
        "MasVnrType": constant("None"),
        "BsmtQual": constant("NA"),
        "BsmtCond": constant("NA"),
        "BsmtExposure": constant("NA"),
        "BsmtFinType1": constant("NA"),
        "BsmtFinType2": constant("NA"),
        "BsmtHalfBath": constant(0),
        "BsmtFullBath": constant(0),
        "TotalBsmtSF": constant(0),
        "BsmtUnfSF": constant(0),
        "BsmtFinSF1": constant(0),
        "BsmtFinSF2": constant(0),
        "Electrical": constant("SBrkr"),  # most frequent
        "FireplaceQu": constant("NA"),
        "PoolQC": constant("NA"),
        "Fence": constant("NA"),
        "MiscFeature": constant("NA"),
        "GarageCond": constant("NA"),
        "GarageQual": constant("NA"),
        "GarageFinish": constant("NA"),
        "GarageType": constant("NA"),
        "GarageCars": constant(0),
        "GarageArea": constant(0),
        "MSZoning": constant("RL"),
        "Utilities": constant("AllPub"),
        "Exterior1st": constant("VinylSd"),
        "Exterior2nd": constant("VinylSd"),
        "KitchenQual": constant("TA"),
        "Functional": constant("Typ"),
        "SaleType": constant("WD")
    }

    preprocessed_data = fill_nans(data, explicit_converters)
    return preprocessed_data.drop("GarageYrBlt", axis=1)


def encode_labels(data: pd.DataFrame, labels_by_column) -> pd.DataFrame:
    data_copy = data.copy()
    for feature in labels_by_column:
        feature_labels = labels_by_column[feature]["feature_labels"]
        le = LabelEncoder()
        le.fit(feature_labels)
        labels_by_column[feature]["encoder"] = le
        data_copy[feature] = le.transform(data[feature])
    return data_copy


def min_max(encoded):
    return MinMaxScaler().fit_transform(encoded)


def root_mean_square_error(y_predicted, y_actual):
    assert len(y_actual) == len(y_predicted)
    return np.sqrt(metrics.mean_squared_error(y_actual, y_predicted))


def predict(scaled_submission):
    return lambda r: r["model"].predict(scaled_submission)


def assert_has_no_nan(array: np.array):
    assert not np.isnan(array).any()


def run_pipeline():
    train, submission = load_data()

    target = train["SalePrice"]
    data = train.drop(["SalePrice"], axis=1)

    # fill nans
    log.info("data preprocessing...")
    preprocessed_data, preprocessed_submission = preprocess_data(data), preprocess_data(submission)

    # encode labels
    log.info("encode labels...")
    labels_by_column = read_json("../dataset/meta/labels.json")
    encoded_submission = encode_labels(preprocessed_submission, labels_by_column)
    encoded_data = encode_labels(preprocessed_data, labels_by_column)

    # scale data
    log.info("scale data...")
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(encoded_data.values)
    scaled_data = min_max_scaler.transform(encoded_data.values)
    scaled_submission = min_max_scaler.transform(encoded_submission.values)

    assert_has_no_nan(scaled_data)
    assert_has_no_nan(scaled_submission)

    sale_price_converter = SalePriceConverter()
    scaled_target = sale_price_converter.scale(target.values.reshape(-1, 1))

    # run models
    log.info("run models...")
    results = []
    regressor_gens = [
        ('LinearRegression(fit_intercept=False)', lambda: LinearRegression(fit_intercept=False)),
        ('KNeighborsRegressor(n_neighbors=10)', lambda: KNeighborsRegressor(n_neighbors=10)),
        ('RandomForestRegressor(n_estimators=250)', lambda: RandomForestRegressor(n_estimators=250)),
    ]

    for description, regressor_gen in regressor_gens:
        log.info("model: " + description)
        result = {
            "description": description,
            "model_factory": regressor_gen,
            "launches": []
        }
        try:
            train_test_splitter = KFold(n_splits=5)
            for train_index, test_index in train_test_splitter.split(scaled_data, scaled_target):
                train_data, test_data = scaled_data[train_index], scaled_data[test_index]
                train_target, test_target = scaled_target[train_index], scaled_target[test_index]

                regressor = regressor_gen()
                regressor.fit(train_data, train_target)
                predicted_target = regressor.predict(test_data)

                rmse = root_mean_square_error(y_predicted=predicted_target, y_actual=test_target)

                log.info("rmse:\t\t{}".format(rmse))

                result["launches"].append({
                    "rmse": rmse,
                    "model": regressor,
                    "data_indexes": {
                        "train_index": train_index,
                        "test_index": test_index
                    }
                })
        except Exception as e:
            error_message = "Error occurred while executing {}: {}".format(description, e)
            log.error(error_message)
            result["error"] = error_message
        results.append(result)

    for result in results:
        launches = result["launches"]
        scaled_submission_predicted_mean = np.mean(np.array(list(map(predict(scaled_submission), launches))), axis=0)
        kfold_average_submissions = sale_price_converter.inv_scale(scaled_submission_predicted_mean)
        log.info(kfold_average_submissions)
        save_submissions(result, kfold_average_submissions, "average")

        regressor = result["model_factory"]()
        regressor.fit(scaled_data, scaled_target)
        full_set_submission = regressor.predict(scaled_submission)
        save_submissions(result, sale_price_converter.inv_scale(full_set_submission), "fullset")


def save_submissions(result, submissions, filename_postfix):
    output_df = pd.DataFrame(data={"Id": range(1461, 2920), "SalePrice": submissions.reshape(1, -1)[0]})
    filename = "submission-{}-{}.csv".format(result["description"], filename_postfix)
    output_df.to_csv(filename, index=False)


if __name__ == '__main__':
    run_pipeline()
