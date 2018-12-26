import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from houseprices.plot import plot_3d_simple, plot_2d_simple, plot_two_features_correlation


def fill_nans(data: pd.DataFrame, converters):
    for index, row in data.iterrows():
        for feature in row.keys():
            if feature in converters:
                new_value = converters[feature](row[feature], row)
                data.at[index, feature] = new_value


def isnan(value):
    return isinstance(value, float) and np.isnan(value)


def isnan_to(value, new_value):
    if isnan(value):
        return new_value
    else:
        return value


def check_garage_info_nans(data: pd.DataFrame) -> None:
    garage_features = ["GarageCond", "GarageQual", "GarageFinish", "GarageType", "GarageYrBlt"]
    explicit_converters = {
        "GarageCond": lambda value, row: isnan_to(value, "NA"),
        "GarageQual": lambda value, row: isnan_to(value, "NA"),
        "GarageFinish": lambda value, row: isnan_to(value, "NA"),
        "GarageType": lambda value, row: isnan_to(value, "NA"),
        "GarageYrBlt": lambda value, row: isnan_to(value, 0.0)
    }

    fill_nans(data, explicit_converters)

    grg_yr_built_by_garage_features = data.groupby(garage_features)["SalePrice"].apply(list)
    print(len(grg_yr_built_by_garage_features.keys()))
    for key in grg_yr_built_by_garage_features.keys():
        sorted_prices = sorted(grg_yr_built_by_garage_features[key])
        print(key, ' -> ', sorted_prices)


def compose_year_built_and_garage_year_built(data: pd.DataFrame) -> None:
    x = data["YearBuilt"].values
    y = data["MSSubClass"].values
    z = data["SalePrice"].values
    plot_3d_simple(x, y, z)


def check_correlation_of_rows_with_garage_and_without(data: pd.DataFrame):
    explicit_converters = {
        "GarageCond": lambda value, row: isnan_to(value, "NA"),
        "GarageQual": lambda value, row: isnan_to(value, "NA"),
        "GarageFinish": lambda value, row: isnan_to(value, "NA"),
        "GarageType": lambda value, row: isnan_to(value, "NA"),
        "GarageYrBlt": lambda value, row: isnan_to(value, 0.0)
    }

    fill_nans(data, explicit_converters)

    overall_quality = 7
    with_garage = data[data["OverallQual"] >= overall_quality]
    no_garage = data[data["OverallQual"] < overall_quality]

    plt.plot(with_garage["YearBuilt"].values, with_garage["SalePrice"].values, '.')
    plt.plot(no_garage["YearBuilt"].values, no_garage["SalePrice"].values, '.')
    plt.savefig('no_garage.png')
    plt.show()


if __name__ == '__main__':
    train = pd.read_csv("../dataset/train.csv")
    x_feature = "YearBuilt"
    y_feature = "SalePrice"
    scaled_target = MinMaxScaler().fit_transform(train[y_feature].values.reshape(-1, 1))
    scaled_target_log = np.log(1 + scaled_target.reshape(1, -1)[0])
    plt.plot(train[x_feature].values, scaled_target, ".")
    # plt.plot(train[x_feature].values, scaled_target_log, ".")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.show()
