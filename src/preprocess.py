from sklearn.model_selection import train_test_split

def preprocess_regression(df):
    X = df.drop(columns=["FinalScore", "PassFail"])
    y = df["FinalScore"]
    return train_test_split(X, y, test_size=0.2)

def preprocess_classification(df):
    X = df.drop(columns=["FinalScore", "PassFail"])
    y = df["PassFail"]
    return train_test_split(X, y, test_size=0.2)
