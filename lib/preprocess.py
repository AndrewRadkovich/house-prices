import pandas

from logging import info, warning, debug
from json import dumps


def extract_type(value):
    return str(type(value))


def extract_meta_info(csv):
    meta = {}
    keys = csv.keys()
    for key in keys:
        meta[key] = {}
        value_set = set(csv[key])
        type_set = set(map(extract_type, value_set))

        if len(type_set) > 1:
            warning("'{}' column has different data type: {}".format(key, type_set))
            value_set = map(lambda v: str(v), value_set)
            meta[key]["type"] = "<class 'str'>"
        else:
            meta[key]["type"] = list(type_set)[0]
        if meta[key]["type"] == "<class 'str'>":
            meta[key]["value_set"] = list(value_set)
        elif meta[key]["type"] in ["<class 'int'>", "<class 'float'>"]:
            meta[key]["stats"] = {
                "mean": csv[key].mean()
            }
    return meta


if __name__ == '__main__':
    csv = pandas.read_csv("../dataset/train.csv")
    meta = extract_meta_info(csv)
    print(dumps(meta, indent=2))
