from logging import warning, info, getLogger, INFO

import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

from lib.helpers import extract_labels_from_data_description
from lib.io import save_json
from lib.plot import plot_2d_simple

getLogger().setLevel(INFO)


def extract_type(value):
    return str(type(value))


def load_data():
    train_csv = pandas.read_csv("../dataset/train.csv")
    test_csv = pandas.read_csv("../dataset/test.csv")
    train_x = train_csv.drop(["Id", "SalePrice"], axis=1)
    train_y = train_csv["SalePrice"]
    test_x = test_csv.drop(["Id"], axis=1)
    return train_x, train_y, test_x


def extract_meta_info(csv):
    meta = {}
    keys = csv.keys()
    info("Shape: {}".format(csv.shape))
    for key in keys:
        meta[key] = {}
        value_set = set(csv[key])
        type_set = set(map(extract_type, value_set))
        meta[key]["type"] = list(type_set)

        if len(type_set) > 1:
            warning("'{}' column has different data type: {}".format(key, type_set))

        if "<class 'str'>" in meta[key]["type"]:
            meta[key]["value_set"] = list(value_set)
        else:
            meta[key]["stats"] = {
                "mean": csv[key].mean(),
                "max": float(csv[key].max()),
                "min": float(csv[key].min())
            }
    return meta


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


def check_dataset_range(meta_info_train_x, meta_info_test_x):
    for label in meta_info_train_x:
        info(label)
        if "<class 'str'>" in meta_info_train_x[label]["type"]:
            diff = meta_info_test_x[label]["value_set"] - meta_info_train_x[label]["value_set"]
            info(diff)
            if len(diff) > 0:
                warning(diff)
        else:
            test_stats = meta_info_test_x[label]["stats"]
            train_stats = meta_info_train_x[label]["stats"]

            info("   train\ttest")

            train_min, test_min, train_max, test_max = train_stats["min"], test_stats["min"], train_stats["max"], \
                                                       test_stats["max"]

            info("min: {},\t{}".format(train_min, test_min))
            info("max: {},\t{}".format(train_max, test_max))
            if train_min != test_min or train_max != test_max:
                warning("train_min = {}, train_max = {}".format(train_min, train_max))
                warning("test_min = {}, test_max = {}".format(test_min, test_max))


def isnan(value):
    return isinstance(value, float) and np.isnan(value)


def isnan_to(value, new_value):
    if isnan(value):
        return new_value
    else:
        return value


def fill_nans(train, converters):
    for index, row in train.iterrows():
        for feature in row.keys():
            if feature in converters:
                new_value = converters[feature](row[feature], row)
                train.at[index, feature] = new_value


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


def run_pipeline():

    def fit(data, target):
        regressor.fit(data.values, target.values)

    def predict(test_data):
        return regressor.predict(test_data)

    train_x, train_y, test_x = load_data()

    garage_year_built_dependency = train_x.groupby("YearBuilt")["GarageYrBlt"].mean()

    # plot_two_features_correlation(train_x, "GarageYrBlt", "YearBuilt")

    plot_2d_simple(garage_year_built_dependency.keys(), garage_year_built_dependency.values)

    explicit_converters = {
        "LotFrontage": lambda value, row: isnan_to(value, 0.0),
        "MasVnrArea": lambda value, row: isnan_to(value, 0.0),
        "Alley": lambda value, row: isnan_to(value, "NA"),
        "MasVnrType": lambda value, row: isnan_to(value, "None"),
        "BsmtQual": lambda value, row: isnan_to(value, "None"),
        "BsmtCond": lambda value, row: isnan_to(value, "NA"),
        "BsmtExposure": lambda value, row: isnan_to(value, "NA"),
        "BsmtFinType1": lambda value, row: isnan_to(value, "NA"),
        "BsmtFinType2": lambda value, row: isnan_to(value, "NA"),
        "Electrical": lambda value, row: isnan_to(value, "NA"),
        "FireplaceQu": lambda value, row: isnan_to(value, "NA"),
        "PoolQC": lambda value, row: isnan_to(value, "NA"),
        "Fence": lambda value, row: isnan_to(value, "NA"),
        "MiscFeature": lambda value, row: isnan_to(value, "NA"),
        "GarageCond": lambda value, row: isnan_to(value, "NA"),
        "GarageQual": lambda value, row: isnan_to(value, "NA"),
        "GarageFinish": lambda value, row: isnan_to(value, "NA"),
        "GarageType": lambda value, row: isnan_to(value, "NA"),
        "GarageYrBlt": lambda value, row: get_or_key(garage_year_built_dependency, row["YearBuilt"])
    }

    fill_nans(train_x, explicit_converters)
    fill_nans(test_x, explicit_converters)

    check_nans(train_x)
    check_nans(test_x)

    meta_info_train_x = extract_meta_info(train_x)
    meta_info_test_x = extract_meta_info(test_x)

    save_json('../dataset/meta/train_x_meta.json', meta_info_train_x)
    save_json('../dataset/meta/test_x_meta.json', meta_info_test_x)
    save_json("../dataset/meta/labels.json", extract_labels_from_data_description())

    regressor = DecisionTreeRegressor(max_depth=16)

    fit(train_x, train_y)
    predicted = predict(test_x.values)
    print(predicted)


if __name__ == '__main__':
    run_pipeline()
