import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SalePriceConverter:
    def __init__(self):
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()

    def fit(self, x, y=None):
        y = np.log(1 + self.scaler.fit(y))
        return self

    def transform(self, data):
        return data

    def scale(self, array: np.ndarray) -> np.ndarray:
        return np.log1p(array)

    def inv_scale(self, array: np.ndarray) -> np.ndarray:
        return np.expm1(array).reshape(-1, 1)


class MissingValuesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return self.pipeline(data).fit_transform()


class LotFrotageImputer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data: pd.DataFrame):
        where_lot_frontage_not_null = np.logical_not(data["LotFrontage"].isnull())
        lot_area = data.loc[where_lot_frontage_not_null, "LotArea"]
        lot_frontage = data.loc[where_lot_frontage_not_null, "LotFrontage"]

        t = (lot_area <= 25000) & (lot_frontage <= 150)
        p = np.polyfit(x=lot_area[t], y=lot_frontage[t], deg=1)

        where_lot_frontage_is_null = data['LotFrontage'].isnull()
        data.loc[where_lot_frontage_is_null, 'LotFrontage'] = np.polyval(p, data.loc[where_lot_frontage_is_null, 'LotArea'])
        return data
