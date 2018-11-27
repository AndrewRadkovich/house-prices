from json import dumps, loads


def save_json(filename, dictionary):
    with open(filename, 'w') as file:
        file.write(dumps(dictionary, indent=2))


def read_json(filename):
    with open(filename) as file:
        return loads(file.read())
