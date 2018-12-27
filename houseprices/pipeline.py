import logging as log

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from houseprices.ioutilites import read_json
from houseprices.preprocessing import SalePriceConverter

log.getLogger().setLevel(log.INFO)


def load_data():
    train_csv = pd.read_csv("../dataset/train.csv")
    test_csv = pd.read_csv("../dataset/test.csv")
    return train_csv, test_csv


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
        # "GarageArea": constant(0),
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


def run_pipeline(remove_outliers_op, cv=KFold(n_splits=5), estimator_factory=None):
    train, submission = load_data()
    log.info("train data shape: {}".format(train.shape))
    log.info("submission shape: {}".format(submission.shape))

    # remove outliers
    log.info("remove outliers...")

    train = remove_outliers_op(train)

    train = train[train["OverallQual"] < 5]
    submission = submission[submission["OverallQual"] < 5]

    target = train["SalePrice"]
    data = train.drop(["SalePrice", "Id", "GarageArea"], axis=1)
    submission = submission.drop(["Id", "GarageArea"], axis=1)

    log.info("train data shape: {}".format(data.shape))
    log.info("submission shape: {}".format(submission.shape))

    # fill nans
    log.info("data preprocessing...")
    preprocessed_data, preprocessed_submission = preprocess_data(data), preprocess_data(submission)
    log.info("train data shape: {}".format(preprocessed_data.shape))
    log.info("submission shape: {}".format(preprocessed_submission.shape))

    # encode labels
    log.info("encode labels...")
    labels_by_column = read_json("../dataset/meta/labels.json")
    encoded_submission = encode_labels(preprocessed_submission, labels_by_column)
    encoded_data = encode_labels(preprocessed_data, labels_by_column)
    log.info("train data shape: {}".format(encoded_data.shape))
    log.info("submission shape: {}".format(encoded_submission.shape))

    # scale data
    log.info("scale data...")
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(encoded_data.values)
    scaled_data = min_max_scaler.transform(encoded_data.values)
    scaled_submission = min_max_scaler.transform(encoded_submission.values)

    assert_has_no_nan(scaled_data)
    assert_has_no_nan(scaled_submission)

    sale_price_converter = SalePriceConverter()
    scaled_target = sale_price_converter.scale(target.values.reshape(-1, 1)).reshape(1, -1)[0]

    # run models
    log.info("run models...")
    results = []

    description = str(estimator_factory())
    log.info("model: " + description)
    result = {
        "description": description,
        "model_factory": estimator_factory,
        "launches": []
    }
    try:
        for train_index, test_index in cv.split(scaled_data, scaled_target):
            train_data, test_data = scaled_data[train_index], scaled_data[test_index]
            train_target, test_target = scaled_target[train_index], scaled_target[test_index]

            estimator = estimator_factory()
            estimator.fit(train_data, train_target)
            predicted_target = estimator.predict(test_data)

            rmse = root_mean_square_error(y_predicted=predicted_target, y_actual=test_target)

            log.info("rmse:\t\t{}".format(rmse))

            result["launches"].append({
                "rmse": rmse,
                "model": estimator,
                "data_indexes": {
                    "train_index": train_index,
                    "test_index": test_index
                }
            })
        rmse_list = np.array(list(map(lambda r: r["rmse"], result["launches"])))
        log.info("mean rmse: {}±{}".format(rmse_list.mean(), rmse_list.std()))
        results.append(result)

        make_fold_average_prediction(results, sale_price_converter, scaled_submission)
        make_fullset_prediction(estimator_factory, sale_price_converter, scaled_data, scaled_submission, scaled_target)
    except Exception as e:
        error_message = "Error occurred while executing {}: {}".format(description, e)
        log.error(error_message)
        result["error"] = error_message


def make_fullset_prediction(estimator_factory, sale_price_converter, scaled_data, scaled_submission, scaled_target):
    estimator = estimator_factory()
    estimator.fit(scaled_data, scaled_target)
    fullset_predicted = estimator.predict(scaled_data)
    fullset_rmse = root_mean_square_error(y_predicted=fullset_predicted, y_actual=scaled_target)
    log.info("fullset rmse: {}".format(fullset_rmse))
    log.info("fullset score: {}".format(estimator.score(scaled_data, scaled_target)))
    full_set_submission = estimator.predict(scaled_submission)
    # save_submissions(sale_price_converter.inv_scale(full_set_submission), "-fullset")


def make_fold_average_prediction(results, sale_price_converter, scaled_submission):
    for result in results:
        launches = result["launches"]
        scaled_submission_predicted_mean = np.mean(np.array(list(map(predict(scaled_submission), launches))),
                                                   axis=0)
        kfold_average_submissions = sale_price_converter.inv_scale(scaled_submission_predicted_mean)
        # save_submissions(kfold_average_submissions, "-average")


def remove_grlivarea_top2_right(train):
    train = train.drop(train[(train["GrLivArea"] > 4500) & (train["SalePrice"] < 300000)].index)
    # train = train.drop(train[train["TotalBsmtSF"] > 3000].index)
    # train = train.drop(train[train["SalePrice"] > 700000].index)
    return train


def save_submissions(submissions, filename_postfix):
    output_df = pd.DataFrame(data={"Id": range(1461, 2920), "SalePrice": submissions.reshape(1, -1)[0]})
    filename = "submission{}.csv".format(filename_postfix)
    output_df.to_csv(filename, index=False)


if __name__ == '__main__':
    # run_pipeline(remove_outliers_op=remove_grlivarea_top2_right,
    #              cv=KFold(n_splits=5),
    #              estimator_factory=lambda: RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
    #                                                              max_features='auto', max_leaf_nodes=None,
    #                                                              min_impurity_decrease=0.0, min_impurity_split=None,
    #                                                              min_samples_leaf=1, min_samples_split=2,
    #                                                              min_weight_fraction_leaf=0.0, n_estimators=500,
    #                                                              n_jobs=16,
    #                                                              oob_score=False, random_state=None, verbose=0,
    #                                                              warm_start=False))

    run_pipeline(remove_outliers_op=remove_grlivarea_top2_right,
                 cv=KFold(n_splits=5),
                 estimator_factory=lambda: LinearRegression(fit_intercept=False))
