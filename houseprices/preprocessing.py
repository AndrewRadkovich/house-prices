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
        return np.log1p(array)

    def inv_scale(self, array: np.ndarray) -> np.ndarray:
        return np.expm1(array).reshape(-1, 1)
