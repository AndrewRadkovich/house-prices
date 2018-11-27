import matplotlib.pyplot as plt
import numpy as np

from scipy import optimize
from sklearn.preprocessing import LabelEncoder


def plot_two_features_correlation(column_oriented_data, x_feature, y_feature):
    plt.plot(column_oriented_data[x_feature].values, column_oriented_data[y_feature].values, 'x')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.show()


def plot_2d_simple(x, y):
    nan_indecis = np.argwhere(np.isnan(y))
    x = np.delete(x, nan_indecis)
    y = np.delete(y, nan_indecis)

    def piecewise_linear(x, x0, y0, k1, k2):
        return np.piecewise(x, [x < x0, x >= x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])

    p, e = optimize.curve_fit(piecewise_linear, x, y)
    plt.plot(x, y)
    plt.plot(x, piecewise_linear(x, *p))
    plt.show()


def generate_correlation_plots(train_x, meta_info_train_x, train_y):
    fig, ax = plt.subplots(20, 4, figsize=(20, 80))
    for feature_idx, feature in enumerate(train_x.keys()):
        values = train_x[feature].values
        x_axis = int(feature_idx / 4)
        y_axis = feature_idx % 4

        if len(meta_info_train_x[feature]["type"]) > 1:
            le = LabelEncoder()
            string_values = list(map(lambda v: str(v), values))
            values = le.fit_transform(string_values)

        ax[x_axis][y_axis].plot(values, train_y, 'x')
        ax[x_axis][y_axis].set_xlabel(feature)
        ax[x_axis][y_axis].set_ylabel('SalePrice')
    fig.savefig("test.png")
