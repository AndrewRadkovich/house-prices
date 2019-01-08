import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion

from houseprices.preprocessing import MissingValuesImputer, LotFrotageImputer


def load_data():
    train = pd.read_csv("../dataset/train.csv")
    test = pd.read_csv("../dataset/test.csv")
    return train, test


def concat_by_features(train: pd.DataFrame, test: pd.DataFrame):
    return pd.concat((train.drop(["SalePrice", "Id"], axis=1), test.drop(["Id"], axis=1)), ignore_index=True)


def main():
    train, test = load_data()
    all_data = concat_by_features(train, test)

    pipeline = Pipeline([
        ('MissingValuesImputer', MissingValuesImputer([
            LotFrotageImputer()
        ])),
        ('FeatureUnion', FeatureUnion([

        ]))
    ])

    transformed_all_data = pipeline.fit_transform(all_data)
    transformed_train_data = transformed_all_data[:train.shape[0]]
    transformed_test_data = transformed_all_data[test.shape[0]:]

    print("transformed_all_data", transformed_all_data.shape)
    print("transformed_train_data", transformed_train_data.shape)
    print("transformed_test_data", transformed_test_data.shape)


if __name__ == '__main__':
    main()
