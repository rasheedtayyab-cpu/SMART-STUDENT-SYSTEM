from sklearn.model_selection import train_test_split

def split_for_regression(data):
    features = data.drop(columns=["FinalScore", "PassFail"])
    target = data["FinalScore"]
    return train_test_split(features, target, test_size=0.2)

def split_for_classification(data):
    features = data.drop(columns=["FinalScore", "PassFail"])
    labels = data["PassFail"]
    return train_test_split(features, labels, test_size=0.2)
