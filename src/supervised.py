from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def run_regression(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return model, mse, r2

def run_classification(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc
