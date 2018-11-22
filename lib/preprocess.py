from json import dumps
from logging import warning, info

import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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


def to_2d_array(array):
    return numpy.array([numpy.array([x]) for x in array])


def generate_correlation_plots():
    fig, ax = plt.subplots(20, 4, figsize=(20, 80))
    for feature_idx, feature in enumerate(train_x.keys()):
        values = train_x[feature].values
        x_axis = int(feature_idx / 4)
        y_axis = feature_idx % 4

        if len(meta_info_train_x[feature]["type"]) > 1:
            le = LabelEncoder()
            string_values = list(map(lambda v: str(v), values))
            values = le.fit_transform(string_values)

        ax[x_axis][y_axis].plot(values, train_y, 'x')
        ax[x_axis][y_axis].set_xlabel(feature)
        ax[x_axis][y_axis].set_ylabel('SalePrice')
    fig.savefig("test.png")


if __name__ == '__main__':
    train_x, train_y, test_x = load_data()
    meta_info_train_x = extract_meta_info(train_x)
    meta_info_test_x = extract_meta_info(test_x)

    for label in meta_info_train_x:
        print(label + " =============================================")
        print(dumps(meta_info_train_x[label], indent=2))
        print(dumps(meta_info_test_x[label], indent=2))

    # normalized_train_x = data_norm(train_x, meta_info_train_x)
    # normalized_test_x = data_norm(test_x, meta_info_test_x)
