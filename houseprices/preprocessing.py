import numpy as np

from sklearn.preprocessing import MinMaxScaler


class SalePriceConverter:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, x, y=None):
        y = np.log(1 + self.scaler.fit(y))
        return self

    def transform(self, data):
        return data

    def scale(self, array: np.ndarray) -> np.ndarray:
        return np.log(1 + self.scaler.fit_transform(array))

    def inv_scale(self, array: np.ndarray) -> np.ndarray:
        inv_log = np.exp(array) - 1
        return self.scaler.inverse_transform(inv_log.reshape(-1, 1))
