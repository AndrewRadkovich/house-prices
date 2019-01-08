import pandas as pd


def load_data():
    train = pd.read_csv("../dataset/train.csv")
    test = pd.read_csv("../dataset/test.csv")
    return train, test


def concat_by_features(train: pd.DataFrame, test: pd.DataFrame):
    return pd.concat((train.drop(["SalePrice", "Id"], axis=1), test.drop(["Id"], axis=1)), ignore_index=True)
