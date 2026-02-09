from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def train_score_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return model, mse, r2

def train_pass_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy
