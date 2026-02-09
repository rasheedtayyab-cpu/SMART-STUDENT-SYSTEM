import numpy as np
import pandas as pd

def generate_student_data(num_students=500):
    study_hours = np.random.uniform(0, 10, num_students)
    attendance = np.random.uniform(0, 100, num_students)
    previous_gpa = np.random.uniform(0, 4, num_students)
    assignments = np.random.uniform(0, 100, num_students)
    sleep_hours = np.random.uniform(0, 12, num_students)

    final_scores = (
        5 * study_hours +
        0.3 * attendance +
        10 * previous_gpa +
        0.4 * assignments -
        2 * sleep_hours +
        np.random.normal(0, 5, num_students)
    )
    final_scores = np.clip(final_scores, 0, 100)
    pass_fail = (final_scores >= 50).astype(int)

    data = pd.DataFrame({
        "StudyHours": study_hours,
        "Attendance": attendance,
        "PreviousGPA": previous_gpa,
        "Assignments": assignments,
        "SleepHours": sleep_hours,
        "FinalScore": final_scores,
        "PassFail": pass_fail
    })
    return data
