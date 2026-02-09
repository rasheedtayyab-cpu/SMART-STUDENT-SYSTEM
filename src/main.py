import os
import pandas as pd
from dataset import load_data
from preprocess import preprocess_regression, preprocess_classification
from supervised import run_regression, run_classification
from unsupervised import run_clustering
from reinforcement import simulate_study_strategy
from sklearn.metrics import pairwise_distances_argmin_min

# Create outputs folder if it doesn't exist
if not os.path.isdir("outputs"):
    os.mkdir("outputs")

def interactive_prediction():
    df = load_data()

    # Train regression model
    X_train_r, X_test_r, y_train_r, y_test_r = preprocess_regression(df)
    model_reg, _, _ = run_regression(X_train_r, X_test_r, y_train_r, y_test_r)

    # Train classification model
    X_train_c, X_test_c, y_train_c, y_test_c = preprocess_classification(df)
    model_clf, _ = run_classification(X_train_c, X_test_c, y_train_c, y_test_c)

    # Train clustering
    clusters = run_clustering(df)

    # RL simulation
    best_hours = simulate_study_strategy()

    # Take user input
    print("\nEnter student details:")
    study_hours = float(input("Study Hours per day (0-10): "))
    attendance = float(input("Attendance (% 0-100): "))
    previous_gpa = float(input("Previous GPA (0-4): "))
    assignments = float(input("Assignments Score (0-100): "))
    sleep_hours = float(input("Sleep Hours per day (0-12): "))

    # Clip input to realistic range
    input_df = pd.DataFrame([{
        "StudyHours": max(0, min(10, study_hours)),
        "Attendance": max(0, min(100, attendance)),
        "PreviousGPA": max(0, min(4, previous_gpa)),
        "Assignments": max(0, min(100, assignments)),
        "SleepHours": max(0, min(12, sleep_hours))
    }])

    # Regression prediction
    predicted_score = model_reg.predict(input_df)[0]

    # Classification prediction based on threshold
    predicted_passfail = 1 if predicted_score >= 50 else 0

    # Nearest cluster
    cluster_label, _ = pairwise_distances_argmin_min(
        input_df.values,
        df.drop(columns=["FinalScore", "PassFail"]).values
    )

    # Output
    print("\n=== Predicted Student Outcome ===")
    print(f"Predicted Final Score: {predicted_score:.2f}")
    print(f"Predicted Pass/Fail: {'Pass' if predicted_passfail==1 else 'Fail'}")
    print(f"Nearest Student Cluster: {clusters[cluster_label[0]]}")
    print(f"Recommended Study Hours (RL): {best_hours}")

if __name__ == "__main__":
    interactive_prediction()
