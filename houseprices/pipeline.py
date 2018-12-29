import logging as log

import numpy as np
import pandas as pd
from sklearn import metrics, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from houseprices.ioutilites import read_json
from houseprices.preprocessing import SalePriceConverter

log.getLogger().setLevel(log.INFO)


def load_data():
    train_csv = pd.read_csv("../dataset/train.csv")
    test_csv = pd.read_csv("../dataset/test.csv")
    log.info("initial train data shape: {}".format(train_csv.shape))
    log.info("initial submission shape: {}".format(test_csv.shape))
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


def root_mean_square_error(y_predicted, y_actual):
    assert len(y_actual) == len(y_predicted)
    return np.sqrt(metrics.mean_squared_error(y_actual, y_predicted))


def assert_has_no_nan(array: np.array):
    assert not np.isnan(array).any()


def save_submissions(filename, submissions: pd.DataFrame):
    submissions.to_csv(filename, index=False)


class HousePricesModel:
    def __init__(self, estimator, labels_by_column, cv=KFold(n_splits=5)):
        self.estimator = estimator
        self.labels_by_column = labels_by_column
        self.train_data_scaler = MinMaxScaler()
        self.sale_price_converter = SalePriceConverter()
        self.features_to_remove = ["GarageArea", "GarageYrBlt"]
        self.scaled_test_data = None
        self.scaled_target = None
        self.cv = cv

    def fit(self, data):
        self.prepare_train_data(data)

        self.run_cross_validation()

        self.estimator.fit(self.scaled_test_data, self.scaled_target)

    def predict(self, test_data):
        test_data_ids = test_data["Id"]
        test_data_rows = test_data.drop(["Id"], axis=1)

        cleaned_from_feature_test_data = self.remove_features(test_data_rows)

        preprocessed_test_data = self.preprocess_data(cleaned_from_feature_test_data)
        encoded_test_data = self.encode(preprocessed_test_data)
        scaled_test_data = self.scale_test_data(encoded_test_data)

        sale_price_scaled_predictions = self.estimator.predict(scaled_test_data)

        sale_price_predictions = self.invert_scale(sale_price_scaled_predictions)

        return pd.DataFrame({"Id": test_data_ids, "SalePrice": sale_price_predictions})

    def run_cross_validation(self):
        result = {
            "description": str(clone(self.estimator)),
            "launches": []
        }
        for train_index, test_index in self.cv.split(self.scaled_test_data, self.scaled_target):
            train_data, test_data = self.scaled_test_data[train_index], self.scaled_test_data[test_index]
            train_target, test_target = self.scaled_target[train_index], self.scaled_target[test_index]

            estimator = clone(self.estimator)
            estimator.fit(train_data, train_target)
            predicted_target = estimator.predict(test_data)

            rmse = root_mean_square_error(y_predicted=predicted_target, y_actual=test_target)
            log.info("rmse:\t\t{}".format(rmse))
            result["launches"].append({
                "rmse": rmse,
                "estimator": estimator
            })
        rmse_list = np.array(list(map(lambda r: r["rmse"], result["launches"])))
        log.info("mean rmse: {}±{}".format(rmse_list.mean(), rmse_list.std()))

    def prepare_train_data(self, data):
        train = self.remove_outliers(data)
        cleaned_from_features_train = self.remove_features(train)
        train_data, train_target = self.split_train_target(cleaned_from_features_train)
        preprocessed_train_data = self.preprocess_data(train_data)
        encoded_train_data = self.encode(preprocessed_train_data)
        self.scaled_test_data = self.scale_train_data(encoded_train_data)
        self.scaled_target = self.scale_target(train_target)
        assert_has_no_nan(self.scaled_test_data)

    def split_train_target(self, train):
        train_target = train["SalePrice"]
        train_data = train.drop(["SalePrice", "Id"], axis=1)
        return train_data, train_target

    def preprocess_data(self, data: pd.DataFrame):
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

        return fill_nans(data, explicit_converters)

    def encode_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        for feature in self.labels_by_column:
            if feature not in data:
                continue
            feature_labels = labels_by_column[feature]["feature_labels"]
            le = LabelEncoder()
            le.fit(feature_labels)
            labels_by_column[feature]["encoder"] = le
            data_copy[feature] = le.transform(data[feature])
        return data_copy

    def invert_scale(self, sale_price_scaled_predictions):
        return self.sale_price_converter.inv_scale(sale_price_scaled_predictions).reshape(1, -1)[0]

    def encode(self, values):
        return self.encode_labels(values).values

    def scale_train_data(self, encoded_train):
        self.train_data_scaler.fit(encoded_train)
        return self.train_data_scaler.transform(encoded_train)

    def scale_test_data(self, encoded_train):
        return self.train_data_scaler.transform(encoded_train)

    def scale_target(self, values):
        return self.sale_price_converter.scale(values.values.reshape(-1, 1)).reshape(1, -1)[0]

    def remove_outliers(self, data):
        return data.drop(data[(data["GrLivArea"] > 4500) & (data["SalePrice"] < 300000)].index)

    def remove_features(self, data: pd.DataFrame):
        return data.drop(self.features_to_remove, axis=1)


if __name__ == '__main__':
    train, test = load_data()
    labels_by_column = read_json("../dataset/meta/labels.json")

    model = HousePricesModel(estimator=LinearRegression(fit_intercept=False),
                             labels_by_column=labels_by_column)

    model.fit(train)
    submission = model.predict(test)
    save_submissions("submission.cvs", submission)
