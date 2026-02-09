import numpy as np
import pandas as pd

def load_data(n_samples=500):
    study_hours = np.random.uniform(0, 10, n_samples)
    attendance = np.random.uniform(0, 100, n_samples)
    previous_gpa = np.random.uniform(0, 4, n_samples)
    assignments = np.random.uniform(0, 100, n_samples)
    sleep_hours = np.random.uniform(0, 12, n_samples)

    final_score = (
        5 * study_hours +
        0.3 * attendance +
        10 * previous_gpa +
        0.4 * assignments -
        2 * sleep_hours +
        np.random.normal(0, 5, n_samples)
    )
    final_score = np.clip(final_score, 0, 100)
    pass_fail = (final_score >= 50).astype(int)

    df = pd.DataFrame({
        "StudyHours": study_hours,
        "Attendance": attendance,
        "PreviousGPA": previous_gpa,
        "Assignments": assignments,
        "SleepHours": sleep_hours,
        "FinalScore": final_score,
        "PassFail": pass_fail
    })
    return df
