import logging as log

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from houseprices.ioutilites import read_json
from houseprices.preprocessing import SalePriceConverter

log.getLogger().setLevel(log.INFO)


def load_data():
    train_csv = pd.read_csv("../dataset/train.csv")
    test_csv = pd.read_csv("../dataset/test.csv")
    print("initial train data shape: {}".format(train_csv.shape))
    print("initial submission shape: {}".format(test_csv.shape))
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


class HousePricesModel:
    def __init__(self, estimator, labels_by_column, cv=KFold(n_splits=5)):
        self.estimator = estimator
        self.labels_by_column = labels_by_column
        self.train_data_scaler = StandardScaler()
        self.sale_price_converter = SalePriceConverter()
        self.features_to_remove = ["GarageArea", "GarageYrBlt"]
        self.scaled_test_data = None
        self.scaled_target = None
        self.score = -1
        self.cv = cv
        self.explicit_converters = {
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

    def fit(self, data):
        self.prepare_train_data(data)

        self.run_cross_validation()

        self.fit_full_train()

    def fit_full_train(self):
        self.estimator.fit(self.scaled_test_data, self.scaled_target)
        full_train_score = root_mean_square_error(self.estimator.predict(self.scaled_test_data), self.scaled_target)
        self.score = full_train_score
        print("full train score: {}".format(full_train_score))

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
        score = np.sqrt(-cross_val_score(estimator=self.estimator,
                                         X=self.scaled_test_data,
                                         y=self.scaled_target,
                                         scoring="neg_mean_squared_error",
                                         cv=self.cv.get_n_splits(self.scaled_test_data)))
        print("        cv score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    def prepare_train_data(self, data):
        train_data = self.remove_outliers(data)
        cleaned_from_features_train = self.remove_features(train_data)
        train_data, train_target = self.split_train_target(cleaned_from_features_train)
        preprocessed_train_data = self.preprocess_data(train_data)
        encoded_train_data = self.encode(preprocessed_train_data)
        self.scaled_test_data = self.scale_train_data(encoded_train_data)
        self.scaled_target = self.scale_target(train_target)
        print("data shape: {}".format(self.scaled_test_data.shape))
        assert_has_no_nan(self.scaled_test_data)

    @staticmethod
    def split_train_target(train):
        train_target = train["SalePrice"]
        train_data = train.drop(["SalePrice", "Id"], axis=1)
        return train_data, train_target

    def preprocess_data(self, data: pd.DataFrame):
        filled_data = fill_nans(data, self.explicit_converters)
        filled_data['TotalSF'] = filled_data['TotalBsmtSF'] + filled_data['1stFlrSF'] + filled_data['2ndFlrSF']
        filled_data["YrMoSold"] = filled_data["YrSold"] + filled_data["MoSold"] / 12
        return filled_data

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

    @staticmethod
    def remove_outliers(data):
        return data.drop(data[(data["GrLivArea"] > 4500) & (data["SalePrice"] < 300000)].index)

    def remove_features(self, data: pd.DataFrame):
        return data.drop(self.features_to_remove, axis=1)


def save(filename, submissions: pd.DataFrame):
    submissions.to_csv("../predictions/" + filename, index=False)


if __name__ == '__main__':
    labels_by_column = read_json("../dataset/meta/labels.json")


    def run(description, estimator):
        print("\n", str(estimator))
        model = HousePricesModel(estimator=estimator, labels_by_column=labels_by_column)
        model.fit(train)
        save(str(model.score) + "-" + description + ".csv", model.predict(test))


    train, test = load_data()

    # save_submissions("KNeighborsRegressor(n_neighbors=5).csv", run(KNeighborsRegressor(n_neighbors=5)))
    # run("RandomForest(n_estimators=3000)", RandomForestRegressor(n_estimators=300))
    run("GradientBoosting(n_estimators=3000)", GradientBoostingRegressor(n_estimators=3000))
