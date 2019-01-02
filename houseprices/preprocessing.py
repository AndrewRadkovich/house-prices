import numpy as np

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
        return self.scaler.fit_transform(np.log1p(self.min_max_scaler.fit_transform(array)))

    def inv_scale(self, array: np.ndarray) -> np.ndarray:
        return self.min_max_scaler.inverse_transform(np.expm1(self.scaler.inverse_transform(array).reshape(-1, 1)))
