import json
import numpy as np
import csv
from datetime import datetime, timedelta
import pytz

from extractors import \
    we_extractors, \
    get_eval_results, \
    get_first_eval_results, \
    get_results_binned,\
    get_results, \
    fea_extractors


# Usage: call the get_features method, give the data & extractors & a timestamp (optional).
# The extractors are a list of functions that extract certain features from a series of the data.


def read_data(filename):
    with open(filename, 'r') as f:
        return json.loads(f.read())


def get_features(data, extractors):
    feature_vectors = []
    feature_labels = []
    for extractor, label_template in extractors:
        part_data, labels = process_extraction(data, extractor, label_template)
        feature_vectors = merge_features(feature_vectors, part_data)
        feature_labels += labels
    return feature_vectors, feature_labels


def remove_exercises(data, forbidden_exercises):
    for student in data:
        for series in student["series"]:
            for index, exercise in enumerate(series["exercises"]):
                newsubm = []
                for subm in exercise:
                    if subm["exercise_id"] not in forbidden_exercises:
                        newsubm.append(subm)
                series["exercises"][index] = newsubm
    return data


def data_at_time(data, timestamp):
    for student in data:
        for series in student["series"]:
            for index, exercise in enumerate(series["exercises"]):
                newsubm = []
                for subm in exercise:
                    try:
                        tm = datetime.strptime(subm["time"], '%Y-%m-%d %H:%M:%S')
                    except:
                        tm = datetime.strptime(subm["time"], '%Y-%m-%d %H:%M:%S %z')

                    if tm < timestamp:
                        newsubm.append(subm)
                series["exercises"][index] = newsubm
    return data


def merge_features(existing, new_data):
    if not existing:
        return new_data
    return [x + y for x, y in zip(existing, new_data)]


def process_extraction(data, extractor, label_template):
    all_students = []
    for student in data:
        # print(student)
        student_vector = []
        for serie in student["series"]:
            # if serie["deadline"]:
            serie_val = extractor(serie)
            student_vector.append(serie_val)
        mean = np.mean(student_vector)
        # sm = sum(student_vector)
        student_vector.append(mean)
        # student_vector.append(sm)
        all_students.append(student_vector)

    labels = [f'wk{x + 1:02}_{label_template}' for x in range(len(all_students[0]) - 1)]
    labels.append(f'mean_{label_template}')
    # labels.append(f'sum__{label_template}')

    return all_students, labels


def write_features_and_labels(features, labels, classes, path, prefix="we_"):
    all_labels = [lbl for sublist in labels for lbl in sublist]
    all_features = [x + y for x, y in zip(features[0], classes)]

    featurespath = f'feature_vectors/{prefix}features_{path}'
    labelspath = f'feature_vectors/{prefix}labels_{path}'

    with open(featurespath, 'w') as fp, open(labelspath, 'w') as lp:
        feature_writer = csv.writer(fp)
        feature_writer.writerows(all_features)

        label_writer = csv.writer(lp)
        label_writer.writerow(all_labels)


def get_all_data(filepath, timestamps = None):
    all = read_data(filepath)
    if timestamps is None:
        timestamps = [serie["deadline"] for serie in all[0]["series"]]
    timestamped_data = []
    for timestamp in timestamps:
    # for i in range(1):
    #     timestamp = timestamps[0]
        if type(timestamp) != datetime:
            try:
                tm = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            except:
                tm = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S %z')
        else:
            tm = timestamp
        wk = data_at_time(read_data(filepath), tm)
        timestamped_data.append(wk)

    return all, timestamped_data


def get_weekly_data(filepath, start, end):
    cur_time = start
    weekly_data = []
    i = 0
    while cur_time <= end:
        print(f"{i}: {cur_time}")
        i += 1
        wk = data_at_time(read_data(filepath), cur_time)
        weekly_data.append(wk)
        # TODO: add 7 days? -> in seconds?
        cur_time += timedelta(seconds=7 * 24 * 60 * 60)
    return weekly_data


def get_all_features(data_complete, weekly, extractors):
    eval_data = get_eval_results(data_complete)
    eval1_data = get_first_eval_results(data_complete)
    marks = get_results(data_complete)
    marks_binned = get_results_binned(data_complete)

    weekly_data = []
    subm_lbls = []
    for data in weekly:
        subm_data, subm_lbls = get_features(data, extractors)
        weekly_data.append(subm_data)

    return eval_data, eval1_data, marks, weekly_data, subm_lbls


def combine_all(datasets, extractors, extractorprefix):
    ds = ["1617", "1718", "1819"]
    timestamps = [eval1_time3, eval1_time1, eval1_time2]
    for i, dataset in enumerate(datasets):
        all, weekly = get_all_data(dataset)

        # submissions = [sum(len(subm) for sub in s['series'] for subm in sub['exercises']) for s in all]

        evalall, eval1, marks, weekly_data, subm_lbls = get_all_features(all, weekly, extractors)

        evals_lbl = ["eval1", "eval2"]
        eval_lbl = ["eval1"]

        for j, week in enumerate(weekly_data):
            fts = [week]
            lbl = [subm_lbls]
            # write_features_and_labels(fts, lbl, marks, f'{ds[i]}_series{j+1}.csv')

            if j == 4:
                write_features_and_labels(fts, lbl, marks, f'{extractorprefix}_{ds[i]}_series{j + 1}.csv')
                fts = [[x + y for x, y in zip(fts[0], eval1)]]

                # fts.append(eval1)
                lbl = [lbl[0] + eval_lbl]
                write_features_and_labels(fts, lbl, marks, f'{extractorprefix}_{ds[i]}_series{j + 1}_eval.csv')

            elif 4 <= j < 9:
                fts = [[x + y for x, y in zip(fts[0], eval1)]]
                lbl = [lbl[0] + eval_lbl]
                write_features_and_labels(fts, lbl, marks, f'{extractorprefix}_{ds[i]}_series{j + 1}.csv')

            elif j >= 9:
                fts = [[x + y for x, y in zip(fts[0], eval1)]]
                lbl = [lbl[0] + eval_lbl]
                write_features_and_labels(fts, lbl, marks, f'{extractorprefix}_{ds[i]}_series{j + 1}.csv')
                # fts.append(eval1)
                eval2 = [[ding[1]] for ding in evalall]
                fts = [[x + y for x, y in zip(fts[0], eval2)]]
                lbl = [lbl[0] + [evals_lbl[1]]]
                write_features_and_labels(fts, lbl, marks, f'{extractorprefix}_{ds[i]}_series{j + 1}_eval.csv')

            else:
                write_features_and_labels(fts, lbl, marks, f'{extractorprefix}_{ds[i]}_series{j + 1}.csv')


def get_marks(data):
    results = []
    for student in data:
        marks = student["marks"]
        m = marks["ex1"] if ("ex1" in marks) and isinstance(marks["ex1"], int) else -1
        m = [student["dodid"], m]
        results.append(m)
    return results

def calc_fea(dataset, start, end, ds, timestamps=None):
    if timestamps is None:
        weekly = get_weekly_data(dataset, start, end)
        all_data = read_data(dataset)

    else:
        all_data, weekly = get_all_data(dataset, timestamps)
    # weekly = []

    evalall, eval1, marks, weekly_data, subm_lbls = get_all_features(all_data, weekly, fea_extractors)

    evals_lbl = ["eval1", "eval2"]
    eval_lbl = ["eval1"]

    for j, week in enumerate(weekly_data):
        fts = [week]
        lbl = [subm_lbls]

        if j == 9:
            write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}.csv', "fea_")
            fts = [[x + y for x, y in zip(fts[0], eval1)]]

            lbl = [lbl[0] + eval_lbl]
            write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}_eval1.csv', "fea_")

        elif 9 <= j < 16:
            fts = [[x + y for x, y in zip(fts[0], eval1)]]
            lbl = [lbl[0] + eval_lbl]
            write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}.csv', "fea_")

        elif j == 16:
            fts = [[x + y for x, y in zip(fts[0], eval1)]]
            lbl = [lbl[0] + eval_lbl]
            write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}.csv', "fea_")

            eval2 = [[ding[1]] for ding in evalall]
            fts = [[x + y for x, y in zip(fts[0], eval2)]]
            lbl = [lbl[0] + [evals_lbl[1]]]
            write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}_eval2.csv', "fea_")

        elif j > 16:
            fts = [[x + y for x, y in zip(fts[0], eval1)]]
            lbl = [lbl[0] + eval_lbl]

            eval2 = [[ding[1]] for ding in evalall]
            fts = [[x + y for x, y in zip(fts[0], eval2)]]
            lbl = [lbl[0] + [evals_lbl[1]]]
            write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}l.csv', "fea_")

        else:
            write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}.csv', "fea_")


def write_marks(faculty, name, year):
    all_data = read_data(f'data/{faculty}/{name}')
    results = []
    none = 0
    for student in all_data:
        marks = student["marks"]
        if marks["ex1"] is None:
            results.append(-1)
            none += 1
        else:
            results.append(marks["ex1"] if ("ex1" in marks and marks["ex1"]) else 0)

    with open(f'feature_vectors/{faculty}_marks{year}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(results)

isbn_exercises = [
    910319224,
    182880102,
    1898834779,
    174432505,
    620641000,
    1316294687,
    387454511,
    933472639,
    2055708402,
    341848809
]

eval1_time1 = '2017-11-06 22:00:00'
eval1_time2 = '2016-11-07 22:00:00'
eval1_time3 = '2018-11-09 22:00:00'

end1617 = datetime.strptime("2017-01-31 22:00:00 +0100", '%Y-%m-%d %H:%M:%S %z')
end1718 = datetime.strptime("2018-01-30 22:00:00 +0100", '%Y-%m-%d %H:%M:%S %z')
end1819 = datetime.strptime("2019-01-29 22:00:00 +0100", '%Y-%m-%d %H:%M:%S %z')

end_sem_1617 = datetime.strptime("2016-12-25 22:00:00 +0100", '%Y-%m-%d %H:%M:%S %z')
end_sem_1718 = datetime.strptime("2017-12-24 22:00:00 +0100", '%Y-%m-%d %H:%M:%S %z')
end_sem_1819 = datetime.strptime("2018-12-23 22:00:00 +0100", '%Y-%m-%d %H:%M:%S %z')

start1617 = datetime.strptime("2016-09-26 22:00:00 +0100", '%Y-%m-%d %H:%M:%S %z')
start1718 = datetime.strptime("2017-09-25 22:00:00 +0100", '%Y-%m-%d %H:%M:%S %z')
start1819 = datetime.strptime("2018-09-24 22:00:00 +0100", '%Y-%m-%d %H:%M:%S %z')

dls161_ = [1476093600000, 1476352800000, 1476698400000, 1476957600000, 1477303200000, 1477562400000, 1477915200000, 1478174400000, 1478520000000, 1478779200000, 1479985200000, 1480330800000, 1480590000000, 1480849200000, 1478520000000, 1481194800000, 1481540400000, 1481799600000, 1482404400000, 1482750000000, 1483009200000]
zone = pytz.timezone("Europe/Brussels")
deadlines1617 = [zone.localize(datetime.fromtimestamp(x/1000)) for x in dls161_]

deadlines1718 = [
    '2017-10-09 12:00:00 +0200',
    '2017-10-12 12:00:00 +0200',
    '2017-10-16 12:00:00 +0200',
    '2017-10-19 12:00:00 +0200',
    '2017-10-26 12:00:00 +0200',
    '2017-10-30 12:00:00 +0100',
    '2017-11-02 12:00:00 +0100',
    '2017-11-06 12:00:00 +0100',
    '2017-11-09 12:00:00 +0100',
    '2017-11-14 12:00:00 +0100',
    '2017-11-23 12:00:00 +0100',
    '2017-11-27 12:00:00 +0100',
    '2017-11-30 12:00:00 +0100',
    '2017-12-04 12:00:00 +0100',
    '2017-12-07 12:00:00 +0100',
    '2017-12-11 12:00:00 +0100',
    '2017-12-14 12:00:00 +0100',
    '2017-12-21 12:00:00 +0100',
    '2017-12-25 12:00:00 +0100',
    '2017-12-28 12:00:00 +0100'
]
deadlines1718 = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S %z') for x in deadlines1718]

deadlines1819 = [
    '2018-10-08 12:00:00 +0200',
    '2018-10-11 12:00:00 +0200',
    '2018-10-15 12:00:00 +0200',
    '2018-10-18 12:00:00 +0200',
    '2018-10-22 12:00:00 +0200',
    '2018-10-25 12:00:00 +0200',
    '2018-10-29 12:00:00 +0100',
    '2018-11-01 12:00:00 +0100',
    '2018-11-05 12:00:00 +0100',
    '2018-11-08 12:00:00 +0100',
    '2018-11-22 12:00:00 +0100',
    '2018-11-26 12:00:00 +0100',
    '2018-12-03 12:00:00 +0100',
    '2018-12-06 12:00:00 +0100',
    '2018-12-10 12:00:00 +0100',
    '2018-12-13 12:00:00 +0100',
    '2018-12-20 12:00:00 +0100',
    '2018-12-24 12:00:00 +0100',
    '2018-12-27 12:00:00 +0100',
    '2019-02-01 12:00:00 +0100'
]
deadlines1819 = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S %z') for x in deadlines1819]

write_marks("fea", "2016-2017-formatted_data.json", "1617")
write_marks("fea", "2017-2018-formatted_data.json", "1718")
write_marks("fea", "2018-2019-formatted_data.json", "1819")
write_marks("we", "studentdata1617.json", "1617")
write_marks("we", "studentdata1718.json", "1718")
write_marks("we", "studentdata1819.json", "1819")

print("MARKS DONE")

combine_all(['data/we/studentdata1617.json',
             'data/we/studentdata1718.json',
             'data/we/studentdata1819.json'], fea_extractors, "fea_extractor")
print("WE1 DONE")
combine_all(['data/we/studentdata1617.json',
             'data/we/studentdata1718.json',
             'data/we/studentdata1819.json'], we_extractors, "we_extractor")

print("DONE WE")

calc_fea('data/fea/2016-2017-formatted_data.json', start1617, end_sem_1617, "1617", deadlines1617)
calc_fea('data/fea/2017-2018-formatted_data.json', start1718, end_sem_1718, "1718", deadlines1718)
calc_fea('data/fea/2018-2019-formatted_data.json', start1819, end_sem_1819, "1819", deadlines1819)
