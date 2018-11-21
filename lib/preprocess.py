from logging import warning, info

import numpy
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
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
        elif meta[key]["type"] in ["<class 'int'>", "<class 'float'>"]:
            meta[key]["stats"] = {
                "mean": csv[key].mean()
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


if __name__ == '__main__':
    train_x, train_y, test_x = load_data()
    meta_info_train_x = extract_meta_info(train_x)
    meta_info_test_x = extract_meta_info(test_x)
    # print(dumps(meta_info_train_x, indent=2))
    # print(dumps(meta_info_test_x, indent=2))

    normalized_train_x = data_norm(train_x, meta_info_train_x)
    normalized_test_x = data_norm(test_x, meta_info_test_x)

    fig, ax = plt.subplots()

    ax = fig.add_subplot(111, projection='3d')
    X = to_2d_array(PCA(n_components=1).fit_transform(train_x[["YearBuilt"]].values))

    Y = to_2d_array(PCA(n_components=1).fit_transform(train_x[["GarageType",
                                                               "GarageYrBlt",
                                                               "GarageFinish",
                                                               "GarageCars",
                                                               "GarageArea",
                                                               "GarageQual",
                                                               "GarageCond"]].values))
    Z = to_2d_array(train_y.values)
    ax.scatter(X, Y, Z)

    ax.set_xlabel('YearBuilt')
    ax.set_ylabel('Functional')
    ax.set_zlabel('SalePrice')

    # ax.plot(to_2d_array(train_x["YearBuilt"].values), to_2d_array(train_y.values), 'x')
    # ax.plot(pca_transform, to_2d_array(train_y.values), 'x')
    ax.grid()
    # fig.savefig("test.png")
    plt.show()
