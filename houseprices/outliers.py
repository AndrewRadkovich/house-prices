import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


def dbscan_outlier_finder(train: np.ndarray, eps: float, min_samples: int) -> (list, list):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(train)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print(labels)

    outlier_indexes = np.argwhere(labels == -1).reshape(1, -1)[0]
    data_indexes = np.argwhere(labels == 0).reshape(1, -1)[0]

    return outlier_indexes, data_indexes


def main():
    train_csv = pd.read_csv("../dataset/train.csv")

    outlier_indexes, data_indexes = dbscan_outlier_finder(train=train_csv[["YearBuilt", "SalePrice"]],
                                                          eps=60000,
                                                          min_samples=100)

    year_series = train_csv["YearBuilt"]
    saleprice_series = train_csv["SalePrice"]

    plt.plot(year_series.values[outlier_indexes], saleprice_series.values[outlier_indexes], '.')
    plt.plot(year_series.values[data_indexes], saleprice_series.values[data_indexes], '.')
    plt.xlabel("YearBuilt")
    plt.ylabel("SalePrice")
    plt.show()


if __name__ == '__main__':
    main()
