import os
import pandas as pd
from dataset import load_data
from preprocess import preprocess_regression, preprocess_classification
from supervised import run_regression, run_classification
from unsupervised import run_clustering
from reinforcement import simulate_study_strategy
from sklearn.metrics import pairwise_distances_argmin_min

if not os.path.isdir("outputs"):
    os.mkdir("outputs")

def predict_student_outcome():
    students = load_data()

    print("Enter the student's details to predict their outcome:")
    hours_studied = float(input("Study Hours per day (0-10): "))
    attendance_pct = float(input("Attendance (% 0-100): "))
    gpa_previous = float(input("Previous GPA (0-4): "))
    assignments_score = float(input("Assignments Score (0-100): "))
    sleep_hours = float(input("Sleep Hours per day (0-12): "))

    student_input = pd.DataFrame([{
        "StudyHours": max(0, min(10, hours_studied)),
        "Attendance": max(0, min(100, attendance_pct)),
        "PreviousGPA": max(0, min(4, gpa_previous)),
        "Assignments": max(0, min(100, assignments_score)),
        "SleepHours": max(0, min(12, sleep_hours))
    }])

    train_features, test_features, train_scores, test_scores = preprocess_regression(students)
    score_model, _, _ = run_regression(train_features, test_features, train_scores, test_scores)

    train_features_clf, test_features_clf, train_labels, test_labels = preprocess_classification(students)
    pass_model, _ = run_classification(train_features_clf, test_features_clf, train_labels, test_labels)

    predicted_score = score_model.predict(student_input)[0]
    predicted_pass = pass_model.predict(student_input)[0]

    clusters = run_clustering(students)
    nearest_cluster, _ = pairwise_distances_argmin_min(
        student_input.values,
        students.drop(columns=["FinalScore", "PassFail"]).values
    )

    suggested_hours = simulate_study_strategy()

    print("\n=== Predicted Outcome ===")
    print(f"Estimated Final Score: {predicted_score:.2f}")
    print(f"Pass/Fail: {'Pass' if predicted_pass == 1 else 'Fail'}")
    print(f"Nearest Peer Cluster: {clusters[nearest_cluster[0]]}")
    print(f"Recommended Study Hours: {suggested_hours}")

if __name__ == "__main__":
    predict_student_outcome()
