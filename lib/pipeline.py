from logging import info, error, getLogger, INFO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from lib.ioutilites import save_json, read_json

getLogger().setLevel(INFO)


def load_data():
    train_csv = pd.read_csv("../dataset/train.csv")
    test_csv = pd.read_csv("../dataset/test.csv")
    train_x = train_csv.drop(["Id", "SalePrice"], axis=1)
    train_y = train_csv["SalePrice"]
    test_x = test_csv.drop(["Id"], axis=1)
    return train_x, train_y, test_x


def data_norm(data, meta_info):
    min_max = MinMaxScaler()
    for key in data:
        if len(meta_info[key]["type"]) > 1:
            le = LabelEncoder()
            string_values = list(map(lambda v: str(v), data[key]))
            data[key] = le.fit_transform(string_values)
        else:
            if meta_info[key]["type"] in ["<class 'int'>", "<class 'float'>"]:
                data[key] = data[key]
            else:
                le = LabelEncoder()
                data[key] = le.fit_transform(data[key])
    normalized_data = min_max.fit_transform(data[:])
    return normalized_data


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


def check_nans(data):
    for feature in data.keys():
        has_nan = any([isnan(value) for value in data[feature]])
        if has_nan:
            print("{} has NaN".format(feature))


def get_or_key(grouped_data, key):
    value = grouped_data[key]
    if isnan(value):
        return key
    else:
        return value


def preprocess_data(submission: pd.DataFrame, data: pd.DataFrame):
    # garage_year_built_dependency = data.groupby("YearBuilt")["GarageYrBlt"].mean()

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
        # "GarageYrBlt": lambda value, row: get_or_key(garage_year_built_dependency, row["YearBuilt"])
    }

    preprocessed_data = fill_nans(data, explicit_converters)
    preprocessed_submission = fill_nans(submission, explicit_converters)
    return preprocessed_data.drop("GarageYrBlt", axis=1), preprocessed_submission.drop("GarageYrBlt", axis=1)


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


def root_mean_square_log_error(y_pred, y_test):
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1 + y_pred) - np.log(1 + y_test)) ** 2))


def run_pipeline():
    data, target, submission = load_data()

    # fill nans
    preprocessed_data, preprocessed_submission = preprocess_data(submission, data)

    # encode labels
    labels_by_column = read_json("../dataset/meta/labels.json")
    encoded_submission = encode_labels(preprocessed_submission, labels_by_column)
    encoded_data = encode_labels(preprocessed_data, labels_by_column)

    # scale data
    scaled_submission = min_max(encoded_submission.values)
    scaled_target = min_max(target.values.reshape(-1, 1))
    scaled_data = min_max(encoded_data.values)

    regressors = [
        ('LinearRegression()', LinearRegression())
    ]

    results = {}

    for description, regressor in regressors:
        info("Started: {}".format(description))
        results[description] = {
            "rmsle": [],
            "error": "no_error"
        }

        try:
            rskf = KFold(n_splits=3)
            for train_index, test_index in rskf.split(scaled_data, scaled_target):
                train_data, test_data = scaled_data[train_index], scaled_data[test_index]
                train_target, test_target = scaled_target[train_index], scaled_target[test_index]

                regressor.fit(train_data, train_target)
                predicted_target = regressor.predict(test_data)

                rmsle = root_mean_square_log_error(y_pred=predicted_target, y_test=test_target)
                results[description]["rmsle"].append(rmsle)

        except Exception as e:
            error_message = "Error occurred while executing {}: {}".format(description, e)
            error(error_message)
            results[description]["error"] = error_message

    save_json("../dataset/results.json", results)


if __name__ == '__main__':
    run_pipeline()
