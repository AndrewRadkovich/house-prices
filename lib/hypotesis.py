import pandas as pd
import numpy as np

from lib.plot import plot_3d_simple


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


if __name__ == '__main__':
    train = pd.read_csv("../dataset/train.csv")
    compose_year_built_and_garage_year_built(train)
