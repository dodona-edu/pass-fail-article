from datetime import datetime
import numpy as np

format_date = '%Y-%m-%d %H:%M:%S.%f'
format_date2 = '%Y-%m-%d %H:%M:%S %z.%f'


def datehelper(datestring):
    if "." not in datestring:
        datestring += ".0"
    try:
        ding = datetime.strptime(datestring, format_date)
    except:
        ding = datetime.strptime(datestring, format_date2)

    return ding


def get_total_submissions(serie):
    subm = [1 for exercise in serie["exercises"] for _ in exercise]
    return sum(subm)


def get_nr_ex_no_submissions(serie):
    subm = [1 if len(exercise) == 0 else 0 for exercise in serie["exercises"]]
    return sum(subm)


def get_time_1st_subm_before_deadline(serie):
    deadline = datehelper(serie["deadline"])
    time = deadline
    for exercise in serie["exercises"]:
        if len(exercise) > 0:
            sorted_ex = sorted(exercise, key=lambda subm: subm["time"])
            first_time = datehelper(sorted_ex[0]["time"])
            if first_time < time:
                time = first_time
    diff = (deadline - time).total_seconds()
    return diff


def get_avg_time_last_subm_before_deadline(serie):
    deadline = datehelper(serie["deadline"])
    time = None
    for exercise in serie["exercises"]:
        for subm in exercise:
            subm_time = datehelper(subm["time"])
            if (not time and deadline > subm_time) or deadline > subm_time > time:
                time = subm_time
    if not time:
        diff = 0
    else:
        diff = (deadline - time).total_seconds()
    return diff


def get_nr_right_before_deadline(serie):
    deadline = datehelper(serie["deadline"])
    right = 0
    for exercise in serie["exercises"]:
        if any(subm["status"] == 1 and datehelper(subm["time"]) < deadline for subm in exercise):
            right += 1
    return right


def get_nr_correct_subm(serie):
    right = 0
    for exercise in serie["exercises"]:
        right += sum([1 if subm["status"] == 1 else 0 for subm in exercise])
    return right


def get_nr_submissions_after_first_correct(serie):
    after_correct = 0
    for exercise in serie["exercises"]:
        sorted_ex = sorted(exercise, key=lambda subm: subm["time"])
        correct_ex = list(filter(lambda x: x["status"] == 1, sorted_ex))

        if len(correct_ex) > 0:
            first_correct_ex = sorted_ex.index(correct_ex[0])
            after_correct += len(sorted_ex) - 1 - first_correct_ex

    return after_correct


def get_nr_submissions_till_first_correct(serie):
    till_correct = 0
    for exercise in serie["exercises"]:
        sorted_ex = sorted(exercise, key=lambda subm: subm["time"])
        correct_ex = list(filter(lambda x: x["status"] == 1, sorted_ex))

        if len(correct_ex) > 0:
            first_correct_index = sorted_ex.index(correct_ex[0])
            till_correct += 1 + first_correct_index
        else:
            till_correct += len(sorted_ex)

    return till_correct


def get_time_between_first_last_submmission(serie):
    all_submissions = []
    for exercise in serie["exercises"]:
        all_submissions += exercise

    sorted_subm = sorted(all_submissions, key=lambda subm: subm["time"])
    if len(sorted_subm) > 1:
        first_time = datehelper(sorted_subm[0]["time"])
        last_time = datehelper(sorted_subm[-1]["time"])
        diff = (last_time - first_time).total_seconds()
    else:
        diff = 0
    return diff


def get_nr_of_exercises_solved_in_2_hours(serie):
    return get_nr_of_exercises_solved_in_x_hours(serie, 7200)


def get_nr_of_exercises_solved_in_24_hours(serie):
    return get_nr_of_exercises_solved_in_x_hours(serie, 86400)


def get_nr_of_exercises_solved_in_15_mins(serie):
    return get_nr_of_exercises_solved_in_x_hours(serie, 15 * 60)


def get_nr_of_exercises_solved_in_10_mins(serie):
    return get_nr_of_exercises_solved_in_x_hours(serie, 10 * 60)


def get_nr_of_exercises_solved_in_5_mins(serie):
    return get_nr_of_exercises_solved_in_x_hours(serie, 5 * 60)


def get_nr_of_exercises_solved_in_x_hours(serie, seconds):
    in_time = 0
    for exercise in serie["exercises"]:
        sorted_ex = sorted(exercise, key=lambda subm: subm["time"])
        correct_ex = list(filter(lambda x: x["status"] == 1, sorted_ex))

        if len(correct_ex) > 0:
            first_correct_time = datehelper(correct_ex[0]["time"])
            first_time = datehelper(sorted_ex[0]["time"])

            diff = (first_correct_time - first_time).total_seconds()
            if diff < seconds:  # 86400 = 24 uur in seconden
                in_time += 1
    return in_time


def get_nr_of_submissions_with_status_wrong(serie):
    return get_nr_of_submissions_with_status_x(serie, 2)


def get_nr_of_submissions_with_status_timelimit(serie):
    return get_nr_of_submissions_with_status_x(serie, 3)


def get_nr_of_submissions_with_status_runtimeerror(serie):
    return get_nr_of_submissions_with_status_x(serie, 6)


def get_nr_of_submissions_with_status_comperror(serie):
    return get_nr_of_submissions_with_status_x(serie, 7)


def get_nr_of_submissions_with_status_memlimit(serie):
    return get_nr_of_submissions_with_status_x(serie, 8)


def get_nr_of_submissions_with_status_x(serie, status):
    nr = 0
    for exercise in serie["exercises"]:
        status_ex = list(filter(lambda x: x["status"] == status, exercise))

        nr += len(status_ex)
    return nr


def get_time_first_correct_subm(serie):
    till_correct = []
    for exercise in serie["exercises"]:
        sorted_ex = sorted(exercise, key=lambda subm: subm["time"])
        correct_ex = list(filter(lambda x: x["status"] == 1, sorted_ex))

        if len(correct_ex) > 0:
            first_correct_time = datehelper(correct_ex[0]["time"])
            first_time = datehelper(sorted_ex[0]["time"])
            # first_correct_index = sorted_ex.index(correct_ex[0])
            till_correct.append((first_correct_time - first_time).total_seconds())
        else:
            till_correct.append(-1)

    return np.mean(till_correct)


def get_plag_data(data):
    return [[student["marks"]["plagrank_adjustleader"]] for student in data]


def get_eval_results(data):
    results = []
    for student in data:
        marks = student["marks"]
        st_eval = [marks["ev1"] if ("ev1" in marks and marks["ev1"]) else 0,
                   marks["ev2"] if ("ev2" in marks and marks["ev2"]) else 0]
        results.append(st_eval)
    return results


def get_first_eval_results_binned(data):
    results = []
    for student in data:
        st_eval = []
        marks = student["marks"]
        st_eval.append([2] if ("ev1" in marks and marks["ev1"] and marks["ev1"] >= 12) else
                       [1] if ("ev1" in marks and marks["ev1"] and marks["ev1"] >= 8) else [0])
        results.append(st_eval)
    return results


def get_first_eval_results(data):
    results = []
    for student in data:
        marks = student["marks"]
        st_eval = [marks["ev1"] if ("ev1" in marks and marks["ev1"]) else 0]
        results.append(st_eval)
    return results


def get_eval_results_binned(data):
    results = []
    for student in data:
        st_eval = []
        marks = student["marks"]
        st_eval.append([2] if ("ev1" in marks and marks["ev1"] and marks["ev1"] >= 12) else
                       [1] if ("ev1" in marks and marks["ev1"] and marks["ev1"] >= 8) else [0])
        st_eval.append([2] if ("ev2" in marks and marks["ev2"] and marks["ev2"] >= 12) else
                       [1] if ("ev2" in marks and marks["ev2"] and marks["ev2"] >= 8) else [0])
        results.append(st_eval)
    return results


def get_results(data):
    results = []
    for student in data:
        marks = student["marks"]
        results.append([1] if ("ex1" in marks and marks["ex1"] and marks["ex1"] >= 10) else [0])
    return results


def get_results_binned(data):
    results = []
    for student in data:
        marks = student["marks"]
        results.append([3] if ("ex1" in marks and marks["ex1"] and marks["ex1"] >= 16) else
                       [2] if ("ex1" in marks and marks["ex1"] and marks["ex1"] >= 12) else
                       [1] if ("ex1" in marks and marks["ex1"] and marks["ex1"] >= 7) else [0])
    return results


we_extractors = [
    (get_total_submissions, "total nr of submissions"),
    (get_nr_ex_no_submissions, "total nr of exercises with no submissions"),
    (get_time_1st_subm_before_deadline, "time of 1st submission before deadline"),
    (get_avg_time_last_subm_before_deadline, "time of last submissions before deadline"),
    (get_nr_right_before_deadline, "number of correct submissions before deadline"),
    (get_nr_correct_subm, "number of correct submissions"),
    (get_nr_submissions_after_first_correct, "number of submissions after first correct submission"),
    (get_nr_submissions_till_first_correct, "number of submissions before first correct"),
    (get_time_between_first_last_submmission, "time between first and last submissions of a series"),
    (get_time_first_correct_subm, "time between first submission and first correct submission"),
    (get_nr_of_exercises_solved_in_2_hours, "nr of exercises in 2 hours"),
    (get_nr_of_exercises_solved_in_24_hours, "nr of exercises in 24 hours"),
    (get_nr_of_exercises_solved_in_5_mins, "nr of exercises in 5  mins"),
    # (get_nr_of_exercises_solved_in_10_mins, "nr of exercises in 10 mins"),
    (get_nr_of_exercises_solved_in_15_mins, "nr of exercises in 15 mins"),
    (get_nr_of_submissions_with_status_wrong, "nr of submissions wrong"),
    (get_nr_of_submissions_with_status_comperror, "nr of submissions compilation error"),
    # (get_nr_of_submissions_with_status_memlimit, "nr of submissions memory limit"),
    (get_nr_of_submissions_with_status_runtimeerror, "nr of submissions runtime error")
    # (get_nr_of_submissions_with_status_timelimit, "nr of submissions time limit"),
    # (get_nr_of_submissions_with_status_wrong, "nr of submissions wrong")
]

fea_extractors = [
    (get_total_submissions, "total nr of submissions"),
    (get_nr_ex_no_submissions, "total nr of exercises with no submissions"),
    # (get_time_1st_subm_before_deadline, "time of 1st submission before deadline"),
    # (get_avg_time_last_subm_before_deadline, "time of last submissions before deadline"),
    # (get_nr_right_before_deadline, "number of correct submissions before deadline"),
    (get_nr_correct_subm, "number of correct submissions"),
    (get_nr_submissions_after_first_correct, "number of submissions after first correct submission"),
    (get_nr_submissions_till_first_correct, "number of submissions before first correct"),
    (get_time_between_first_last_submmission, "time between first and last submissions of a series"),
    (get_time_first_correct_subm, "time between first submission and first correct submission"),
    (get_nr_of_exercises_solved_in_2_hours, "nr of exercises in 2 hours"),
    (get_nr_of_exercises_solved_in_24_hours, "nr of exercises in 24 hours"),
    (get_nr_of_exercises_solved_in_5_mins, "nr of exercises in 5  mins"),
    # (get_nr_of_exercises_solved_in_10_mins, "nr of exercises in 10 mins"),
    (get_nr_of_exercises_solved_in_15_mins, "nr of exercises in 15 mins"),
    (get_nr_of_submissions_with_status_wrong, "nr of submissions wrong"),
    (get_nr_of_submissions_with_status_comperror, "nr of submissions compilation error"),
    # (get_nr_of_submissions_with_status_memlimit, "nr of submissions memory limit"),
    (get_nr_of_submissions_with_status_runtimeerror, "nr of submissions runtime error")
    # (get_nr_of_submissions_with_status_timelimit, "nr of submissions time limit"),
    # (get_nr_of_submissions_with_status_wrong, "nr of submissions wrong")
]
