import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from houseprices.ioutilites import read_json
from houseprices.preprocessing import SalePriceConverter


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


def fill_nans(data: pd.DataFrame, converters) -> pd.DataFrame:
    train_copy = data.copy()
    for index, row in train_copy.iterrows():
        for feature in row.keys():
            if feature in converters:
                new_value = converters[feature](row[feature])
                train_copy.at[index, feature] = new_value
    return train_copy


def root_mean_square_error(y_predicted, y_actual):
    assert len(y_actual) == len(y_predicted)
    return np.sqrt(metrics.mean_squared_error(y_actual, y_predicted))


def save(filename, submissions: pd.DataFrame):
    submissions.to_csv("../predictions/" + filename, index=False)


class OutlierRemover(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data.drop(data[(data["GrLivArea"] > 4500) & (data["SalePrice"] < 300000)].index)


class FeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_remove):
        self.features_to_remove = features_to_remove

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data.drop(self.features_to_remove, axis=1)


class TotalSquareFeetFeatureEnhancer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data: pd.DataFrame):
        data['FlrSF'] = data['1stFlrSF'] + data['2ndFlrSF']
        data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
        return data


class YearMonthSoldFeatureEnhancer(BaseEstimator, TransformerMixin):
    def __init__(self, ted_rates):
        self.ted_rates = ted_rates

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        data["YrMoSold"] = data["YrSold"] + data["MoSold"] / 12
        return data


class HousePricesLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.labels_by_column = read_json("../dataset/meta/labels.json")

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        for feature in self.labels_by_column:
            if feature not in data:
                continue
            feature_labels = self.labels_by_column[feature]["feature_labels"]
            le = LabelEncoder()
            le.fit(feature_labels)
            self.labels_by_column[feature]["encoder"] = le
            data[feature] = le.transform(data[feature])

        for feature in data.select_dtypes(include=['object']):
            if feature not in self.labels_by_column:
                print(feature + " not in encoder file: applying default label encoder")
                data[feature] = LabelEncoder().fit_transform(data[feature])
        return data


class MissingValuesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
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
            "MSZoning": constant("RL"),
            "Utilities": constant("AllPub"),
            "Exterior1st": constant("VinylSd"),
            "Exterior2nd": constant("VinylSd"),
            "KitchenQual": constant("TA"),
            "Functional": constant("Typ"),
            "SaleType": constant("WD")
        }

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        train_copy = data.copy()
        for index, row in train_copy.iterrows():
            for feature in row.keys():
                if feature in self.explicit_converters:
                    new_value = self.explicit_converters[feature](row[feature])
                    train_copy.at[index, feature] = new_value
        return train_copy


class SkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, all_data):
        self.all_data = all_data

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        numeric_feats = self.all_data.dtypes[self.all_data.dtypes != "object"].index
        skewed_feats = self.all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew': skewed_feats})
        skewness = skewness[abs(skewness.Skew) > 0.75]
        for feat in skewness.index:
            data[feat] = np.log1p(data[feat])
        return data


class OrderedLabelsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ordered_labels = {
            "Alley": {"NA": 0, "Grvl": 1, "Pave": 2},
            "BsmtCond": {"NA": 0, "No": 1, "Po": 2, "Fa": 3, "TA": 4, "Gd": 5, "Ex": 6},
            "BsmtExposure": {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
            "BsmtFinType1": {"NA": 0, "No": 1, "Unf": 2, "LwQ": 3, "Rec": 4, "BLQ": 5, "ALQ": 6, "GLQ": 7},
            "BsmtFinType2": {"NA": 0, "No": 1, "Unf": 2, "LwQ": 3, "Rec": 4, "BLQ": 5, "ALQ": 6, "GLQ": 7},
            "BsmtQual": {"NA": 0, "No": 1, "Po": 2, "Fa": 3, "TA": 4, "Gd": 5, "Ex": 6},
            "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "FireplaceQu": {"NA": 0, "No": 1, "Po": 2, "Fa": 3, "TA": 4, "Gd": 5, "Ex": 6},
            "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
            "GarageCond": {"NA": 0, "No": 1, "Po": 2, "Fa": 3, "TA": 4, "Gd": 5, "Ex": 6},
            "GarageQual": {"NA": 0, "No": 1, "Po": 2, "Fa": 3, "TA": 4, "Gd": 5, "Ex": 6},
            "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
            "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
            "PavedDrive": {"N": 0, "P": 1, "Y": 2},
            "PoolQC": {"NA": 0, "No": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "Street": {"Grvl": 1, "Pave": 2},
            "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}
        }

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data.replace(self.ordered_labels)


def split_train_target(data):
    train_target = data["SalePrice"]
    train_data = data.drop(["SalePrice", "Id"], axis=1)
    return train_data, train_target


def rmse_cv(estimator, X, y, cv):
    return np.sqrt(-cross_val_score(estimator=estimator,
                                    X=X,
                                    y=y,
                                    scoring="neg_mean_squared_error",
                                    cv=cv.get_n_splits(y),
                                    verbose=2,
                                    n_jobs=8))


def fit_full_predict(estimator, scaled_train_data, scaled_train_target, scaled_test_data):
    estimator.fit(X=scaled_train_data, y=scaled_train_target)
    p = estimator.predict(X=scaled_train_data, y=scaled_train_target)
    rmse = root_mean_square_error(p, scaled_train_target)
    print("rullset rmse: {:.5f}".format(rmse))
    predictions = estimator.predict(X=scaled_test_data)
    return sale_price_converter.inv_scale(predictions).reshape(1, -1)[0]


if __name__ == '__main__':
    train, test = load_data()
    tedrate_csv = pd.read_csv("../dataset/TEDRATE.csv")
    tedrate_csv["DATE"] = tedrate_csv["DATE"].apply(lambda x: x[:7])
    tedrate_by_year_month = tedrate_csv.groupby("DATE").mean()

    train = OutlierRemover().fit_transform(train)
    print("train data shape: {}".format(train.shape))

    train_data, train_target = split_train_target(train)
    all_data = pd.concat((train_data, test), sort=True)

    sale_price_converter = SalePriceConverter()
    scaled_train_target = sale_price_converter.scale(train_target.values.reshape(-1, 1)).reshape(1, -1)[0]

    pipeline = Pipeline([
        ('remove_features', FeatureRemover(["GarageYrBlt"])),
        ('fill_missing', MissingValuesTransformer()),
        ('ordered_features', OrderedLabelsTransformer()),
        ('total_square_feet_feature', TotalSquareFeetFeatureEnhancer()),
        ('year_month_sold_feature', YearMonthSoldFeatureEnhancer(tedrate_by_year_month)),
        ('encode_labels', HousePricesLabelEncoder()),
        ('skewness', SkewnessTransformer(all_data)),
        ('scale_data', StandardScaler())
    ])

    scaled_train_data = pipeline.fit_transform(X=train_data, y=scaled_train_target)

    estimator = lgb.LGBMRegressor(objective='regression', n_estimators=100, random_state=42)

    score = rmse_cv(estimator=estimator,
                    X=scaled_train_data,
                    y=scaled_train_target,
                    cv=KFold(n_splits=5, shuffle=False))
    formatted_cv_score = "{:.4f} ({:.4f})".format(score.mean(), score.std())
    print("        cv score: " + formatted_cv_score)

    ids = test["Id"]
    scaled_test_data = pipeline.fit_transform(X=test.drop(["Id"], axis=1))
    sale_prices = fit_full_predict(estimator, scaled_train_data, scaled_train_target, scaled_test_data)
    save(formatted_cv_score + ".csv", pd.DataFrame({
        "Id": ids,
        "SalePrice": sale_prices
    }))
