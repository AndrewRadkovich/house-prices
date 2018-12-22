import logging as log

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from lib.converters import SalePriceConverter
from lib.ioutilites import read_json, save_json

log.getLogger().setLevel(log.INFO)


def load_data():
    train_csv = pd.read_csv("../dataset/train.csv")
    test_csv = pd.read_csv("../dataset/test.csv")
    train = train_csv.drop(["Id"], axis=1)
    submission = test_csv.drop(["Id"], axis=1)
    return train, submission


def isnan(value):
    return isinstance(value, float) and np.isnan(value)


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
                new_value = converters[feature](row[feature], row)
                train_copy.at[index, feature] = new_value
    return train_copy


def preprocess_data(data: pd.DataFrame):
    explicit_converters = {
        "LotFrontage": lambda value, row: isnan_to(value, 0.0),
        "MasVnrArea": lambda value, row: isnan_to(value, 0.0),
        "Alley": lambda value, row: isnan_to(value, "NA"),
        "MasVnrType": lambda value, row: isnan_to(value, "None"),
        "BsmtQual": lambda value, row: isnan_to(value, "NA"),
        "BsmtCond": lambda value, row: isnan_to(value, "NA"),
        "BsmtExposure": lambda value, row: isnan_to(value, "NA"),
        "BsmtFinType1": lambda value, row: isnan_to(value, "NA"),
        "BsmtFinType2": lambda value, row: isnan_to(value, "NA"),
        "Electrical": lambda value, row: isnan_to(value, "SBrkr"),  # most frequent
        "FireplaceQu": lambda value, row: isnan_to(value, "NA"),
        "PoolQC": lambda value, row: isnan_to(value, "NA"),
        "Fence": lambda value, row: isnan_to(value, "NA"),
        "MiscFeature": lambda value, row: isnan_to(value, "NA"),
        "GarageCond": lambda value, row: isnan_to(value, "NA"),
        "GarageQual": lambda value, row: isnan_to(value, "NA"),
        "GarageFinish": lambda value, row: isnan_to(value, "NA"),
        "GarageType": lambda value, row: isnan_to(value, "NA"),
        "MSZoning": lambda value, row: isnan_to(value, "RL"),
        "Utilities": lambda value, row: isnan_to(value, "AllPub"),
        "Exterior1st": lambda value, row: isnan_to(value, "VinylSd"),
        "Exterior2nd": lambda value, row: isnan_to(value, "VinylSd"),
        "KitchenQual": lambda value, row: isnan_to(value, "TA"),
        "Functional": lambda value, row: isnan_to(value, "Typ"),
        "SaleType": lambda value, row: isnan_to(value, "WD"),
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
    min_max_scaler.fit(np.concatenate((encoded_data.values, encoded_submission.values), axis=0))
    scaled_data = min_max_scaler.transform(encoded_data.values)
    scaled_submission = min_max_scaler.transform(encoded_submission.values)

    sale_price_converter = SalePriceConverter()
    scaled_target = sale_price_converter.scale(target.values.reshape(-1, 1))

    # run models
    log.info("run models...")
    results = []
    regressor_gens = [
        ('LinearRegression(fit_intercept=False)', lambda: LinearRegression(fit_intercept=False)),
        ('LinearRegression(fit_intercept=True)', lambda: LinearRegression(fit_intercept=True)),
    ]

    for description, regressor_gen in regressor_gens:
        log.info("model: " + description)
        result = {
            "description": description,
            "laucnhes": []
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

                result["laucnhes"].append({
                    "rmse": rmse,
                    "model": regressor,
                    "data_indexes": {
                        "train_index": train_index.tolist(),
                        "test_index": test_index.tolist()
                    }
                })
        except Exception as e:
            error_message = "Error occurred while executing {}: {}".format(description, e)
            log.error(error_message)
            result["error"] = error_message
        results.append(result)

    # save_json("../dataset/results.json", results)

    scaled_submission_predicted_mean = np.mean(np.array(list(map(predict(scaled_submission), results))), axis=0)
    submissions = sale_price_converter.inv_scale(scaled_submission_predicted_mean)
    log.info(submissions)


if __name__ == '__main__':
    run_pipeline()
