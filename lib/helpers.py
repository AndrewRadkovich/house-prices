def extract_labels_from_data_description():
    labels = {}
    with open('../dataset/data_description.txt') as data_description_file:
        lines = data_description_file.readlines()
        current_feature = ''
        feature_labels = []
        for line in lines:
            if ':' in line and '\t' not in line:
                if len(feature_labels) > 0:
                    labels[current_feature] = {
                        "feature_labels": feature_labels
                    }
                current_feature = line.split(':')[0]
                feature_labels = []
            else:
                label = line.strip().split('\t')[0].split('s+')[0]
                if '' != label:
                    feature_labels.append(label)
        if len(feature_labels) > 0:
            labels[current_feature] = {
                "feature_labels": feature_labels
            }
    return labels