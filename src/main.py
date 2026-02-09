import os
import pandas as pd
from dataset import generate_student_data
from preprocess import split_for_regression, split_for_classification
from supervised import train_score_model, train_pass_model
from unsupervised import cluster_students
from reinforcement import find_optimal_study_hours
from sklearn.metrics import pairwise_distances_argmin_min

if not os.path.isdir("outputs"):
    os.mkdir("outputs")

def predict_student_outcome():
    students = generate_student_data()

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

    X_train_scores, X_test_scores, y_train_scores, y_test_scores = split_for_regression(students)
    score_model, _, _ = train_score_model(X_train_scores, X_test_scores, y_train_scores, y_test_scores)

    X_train_pass, X_test_pass, y_train_pass, y_test_pass = split_for_classification(students)
    pass_model, _ = train_pass_model(X_train_pass, X_test_pass, y_train_pass, y_test_pass)

    predicted_score = score_model.predict(student_input)[0]
    predicted_pass = pass_model.predict(student_input)[0]

    clusters = cluster_students(students)
    nearest_cluster, _ = pairwise_distances_argmin_min(
        student_input.values,
        students.drop(columns=["FinalScore", "PassFail"]).values
    )

    suggested_hours = find_optimal_study_hours()

    print("\n=== Predicted Outcome ===")
    print(f"Estimated Final Score: {predicted_score:.2f}")
    print(f"Pass/Fail: {'Pass' if predicted_pass == 1 else 'Fail'}")
    print(f"Nearest Peer Cluster: {clusters[nearest_cluster[0]]}")
    print(f"Recommended Study Hours: {suggested_hours}")

if __name__ == "__main__":
    predict_student_outcome()
