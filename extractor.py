import json
import numpy as np
import csv
from datetime import datetime, timedelta

from extractors import we_extractors, get_plag_data, get_eval_results, get_first_eval_results, get_results_binned, \
    get_results, fea_extractors


# Get weekly data
# Extract features from the weekly data
# Write all features


class Extractor:
    def __init__(self, student_data_path: str, extractors: list, prefix: str, faculty: str):
        self.extractors = extractors
        self.prefix = prefix
        self.faculty = faculty
        with open(student_data_path, 'r') as sdf:
            self.student_data = json.loads(sdf.read())

    def get_features(self, data):
        feature_vectors = []
        feature_labels = []
        for extractor, label_template in self.extractors:
            part_data, labels = self.process_extraction(data, extractor, label_template)
            feature_vectors = merge_features(feature_vectors, part_data)
            feature_labels += labels
        return feature_vectors, feature_labels

    def process_extraction(self, data, extractor, label_template):
        all_students = []
        for student in data:
            # print(student)
            student_vector = []
            for serie in student["series"]:
                # if serie["deadline"]:
                serie_val = extractor(serie)
                student_vector.append(serie_val)
            mean = np.mean(student_vector)
            sm = sum(student_vector)
            student_vector.append(mean)
            student_vector.append(sm)
            all_students.append(student_vector)

        labels = [f'wk{x + 1:02}_{label_template}' for x in range(len(all_students[0]) - 2)]
        labels.append(f'mean_{label_template}')
        labels.append(f'sum__{label_template}')

        return all_students, labels

    def write_features_and_labels(self, features, labels, classes, path):
        all_labels = [lbl for sublist in labels for lbl in sublist]
        all_features = [x + y for x, y in zip(features[0], classes)]

        featurespath = f'feature_vectors/{self.prefix}features_{path}'
        labelspath = f'feature_vectors/{self.prefix}labels_{path}'

        with open(featurespath, 'w') as fp, open(labelspath, 'w') as lp:
            feature_writer = csv.writer(fp)
            feature_writer.writerows(all_features)

            label_writer = csv.writer(lp)
            label_writer.writerow(all_labels)


# TODO: super call?
class DeadlineExtractor(Extractor):
    def __init__(self, student_data_path: str, extractors: list, faculty, prefix, deadlines=None):
        super().__init__(student_data_path, extractors, faculty, prefix)
        if deadlines is None:
            deadlines = [serie["deadline"] for serie in self.student_data[0]["series"]]
        self.deadlines = deadlines

    def get_all_data(self):
        # if timestamps is None:
        #     timestamps = [serie["deadline"] for serie in all[0]["series"]]
        timestamped_data = []
        for timestamp in self.deadlines:
            if type(timestamp) != datetime:
                try:
                    tm = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                except:
                    tm = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S %z')
            else:
                tm = timestamp
            wk = self.data_at_time(tm)
            timestamped_data.append(wk)

        return self.student_data, timestamped_data

    def data_at_time(self, timestamp):
        all_data = self.student_data.deepcopy()
        for student in all_data:
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
        return all_data

    def get_all_features(self, data_complete, weekly, extractors):
        # plag_data = get_plag_data(data_complete)
        eval_data = get_eval_results(data_complete)
        eval1_data = get_first_eval_results(data_complete)
        marks = get_results(data_complete)
        marks_binned = get_results_binned(data_complete)

        weekly_data = []
        subm_lbls = []
        for data in weekly:
            subm_data, subm_lbls = self.get_features(data, extractors)
            weekly_data.append(subm_data)

        return eval_data, eval1_data, marks, weekly_data, subm_lbls

    def write_we(self, ds):
        all, weekly = self.get_all_data()

        # submissions = [sum(len(subm) for sub in s['series'] for subm in sub['exercises']) for s in all]

        evalall, eval1, marks, weekly_data, subm_lbls = self.get_all_features(all, weekly, fea_extractors)

        evals_lbl = ["eval1", "eval2"]
        eval_lbl = ["eval1"]

        for j, week in enumerate(weekly_data):
            fts = [week]
            lbl = [subm_lbls]
            # write_features_and_labels(fts, lbl, marks, f'{ds[i]}_series{j+1}.csv')

            if j == 4:
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}.csv')
                fts = [[x + y for x, y in zip(fts[0], eval1)]]

                # fts.append(eval1)
                lbl = [lbl[0] + eval_lbl]
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}_eval.csv')

            elif 4 <= j < 9:
                fts = [[x + y for x, y in zip(fts[0], eval1)]]
                lbl = [lbl[0] + eval_lbl]
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}.csv')

            elif j >= 9:
                fts = [[x + y for x, y in zip(fts[0], eval1)]]
                lbl = [lbl[0] + eval_lbl]
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}.csv')
                # fts.append(eval1)
                eval2 = [[ding[1]] for ding in evalall]
                fts = [[x + y for x, y in zip(fts[0], eval2)]]
                lbl = [lbl[0] + [evals_lbl[1]]]
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}_eval.csv')

            else:
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}.csv')

    def write_fea(self, ds):
        all_data, weekly = self.get_all_data()

        evalall, eval1, marks, weekly_data, subm_lbls = self.get_all_features(all_data, weekly, fea_extractors)

        evals_lbl = ["eval1", "eval2"]
        eval_lbl = ["eval1"]

        for j, week in enumerate(weekly_data):
            fts = [week]
            lbl = [subm_lbls]

            if j == 6:
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}.csv')
                fts = [[x + y for x, y in zip(fts[0], eval1)]]

                lbl = [lbl[0] + eval_lbl]
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}_eval.csv')

            elif 6 <= j < 10:
                fts = [[x + y for x, y in zip(fts[0], eval1)]]
                lbl = [lbl[0] + eval_lbl]
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}.csv')

            elif j == 10:
                fts = [[x + y for x, y in zip(fts[0], eval1)]]
                lbl = [lbl[0] + eval_lbl]
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}.csv')

                eval2 = [[ding[1]] for ding in evalall]
                fts = [[x + y for x, y in zip(fts[0], eval2)]]
                lbl = [lbl[0] + [evals_lbl[1]]]
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}_eval.csv')

            elif j > 10:
                eval2 = [[ding[1]] for ding in evalall]
                fts = [[x + y for x, y in zip(fts[0], eval2)]]
                lbl = [lbl[0] + [evals_lbl[1]]]
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}_eval.csv')

            else:
                self.write_features_and_labels(fts, lbl, marks, f'{ds}_series{j + 1}.csv')


class WeeklyExtractor(DeadlineExtractor):
    def __init__(self, student_data_path: str, start: datetime, end: datetime, extractors: list, prefix, faculty):
        deadlines = self.calculate_deadlines(start, end, timedelta(7))
        super().__init__(student_data_path, extractors, prefix, faculty, deadlines)

    def calculate_deadlines(self, start: datetime, end: datetime, interval: timedelta) -> list:
        deadlines = []
        while start < end:
            start += interval
            deadlines.append(start)
        return deadlines


# Usage: call the get_features method, give the data & extractors & a timestamp (optional).
# The extractors are a list of functions that extract certain features from a series of the data.


# def read_data(filename):
#     with open(filename, 'r') as f:
#         return json.loads(f.read())


# def get_features(data, extractors):
#     feature_vectors = []
#     feature_labels = []
#     for extractor, label_template in extractors:
#         part_data, labels = process_extraction(data, extractor, label_template)
#         feature_vectors = merge_features(feature_vectors, part_data)
#         feature_labels += labels
#     return feature_vectors, feature_labels


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


def merge_features(existing, new_data):
    if not existing:
        return new_data
    return [x + y for x, y in zip(existing, new_data)]


def get_marks(data):
    results = []
    for student in data:
        marks = student["marks"]
        m = marks["ex1"] if ("ex1" in marks) and isinstance(marks["ex1"], int) else -1
        m = [student["dodid"], m]
        results.append(m)
    return results


def write_marks(faculty, name, year):
    with open(f'data/{faculty}/{name}', 'r') as sdf:
        all_data = json.loads(sdf.read())
        # all_data = read_data(f'data/{faculty}/{name}')
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


write_marks("fea", "2016-2017-formatted_data.json", "1617")
write_marks("fea", "2017-2018-formatted_data.json", "1718")
write_marks("fea", "2018-2019-formatted_data.json", "1819")
write_marks("we", "studentdata1617.json", "1617")
write_marks("we", "studentdata1718.json", "1718")
write_marks("we", "studentdata1819.json", "1819")

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

# combine_all(['data/we/studentdata1617.json',
#              'data/we/studentdata1718.json',
#              'data/we/studentdata1819.json'])


deadlines1617 = [
    '26-9-2016 12:00:00 +0200',
    '29-9-2016 12:00:00 +0200',
    '3-10-2016 12:00:00 +0200',
    '6-10-2016 12:00:00 +0200',
    '10-10-2016 12:00:00 +0200',
    '13-10-2016 12:00:00 +0200',
    '17-10-2016 12:00:00 +0100',
    '20-10-2016 12:00:00 +0100',
    '24-10-2016 12:00:00 +0100',
    '27-10-2016 12:00:00 +0100',
    '10-11-2016 12:00:00 +0100',
    '14-11-2016 12:00:00 +0100',
    '17-11-2016 12:00:00 +0100',
    '20-11-2016 12:00:00 +0100',
    '24-10-2016 12:00:00 +0100',
    '27-10-2016 12:00:00 +0100',
    '10-11-2016 12:00:00 +0100',
    '14-11-2016 12:00:00 +0100',
    '17-11-2016 12:00:00 +0100',
    '21-11-2016 12:00:00 +0100',
    '24-11-2016 12:00:00 +0100',
    '28-11-2016 12:00:00 +0100',
    '1-12-2016 12:00:00 +0100',
    '8-12-2016 12:00:00 +0100',
    '12-12-2016 12:00:00 +0100',
    '15-12-2016 12:00:00 +0100'
]
deadlines1617 = [datetime.strptime(x, '%d-%m-%Y %H:%M:%S %z') for x in deadlines1617]

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

fea_deadline_extractor_16 = DeadlineExtractor('data/fea/2016-2017-formatted_data.json', deadlines1617, we_extractors,
                                              "fea_dl_", "fea")
fea_deadline_extractor_17 = DeadlineExtractor('data/fea/2017-2018-formatted_data.json', deadlines1718, we_extractors,
                                              "fea_dl_", "fea")
fea_deadline_extractor_18 = DeadlineExtractor('data/fea/2018-2019-formatted_data.json', deadlines1819, we_extractors,
                                              "fea_dl_", "fea")

fea_weekly_extractor_16 = WeeklyExtractor('data/fea/2016-2017-formatted_data.json', start1617, end1617, fea_extractors,
                                          "fea_", "fea")
fea_weekly_extractor_17 = WeeklyExtractor('data/fea/2017-2018-formatted_data.json', start1718, end1718, fea_extractors,
                                          "fea_", "fea")
fea_weekly_extractor_18 = WeeklyExtractor('data/fea/2018-2019-formatted_data.json', start1819, end1819, fea_extractors,
                                          "fea_", "fea")

we_deadline_extractor_16 = DeadlineExtractor('data/we/studentdata1617.json', deadlines1617, we_extractors, "we_dl_",
                                             "we")
we_deadline_extractor_17 = DeadlineExtractor('data/we/studentdata1718.json', deadlines1718, we_extractors, "we_dl_",
                                             "we")
we_deadline_extractor_18 = DeadlineExtractor('data/we/studentdata1819.json', deadlines1819, we_extractors, "we_dl_",
                                             "we")

we_weekly_extractor_16 = WeeklyExtractor('data/we/studentdata1617.json', start1617, end1617, fea_extractors, "we_",
                                         "we")
we_weekly_extractor_17 = WeeklyExtractor('data/we/studentdata1718.json', start1718, end1718, fea_extractors, "we_",
                                         "we")
we_weekly_extractor_18 = WeeklyExtractor('data/we/studentdata1819.json', start1819, end1819, fea_extractors, "we_",
                                         "we")


def extract_fea():
    fea_deadline_extractor_16.write_fea("2016-2017")
    fea_deadline_extractor_17.write_fea("2017-2018")
    fea_deadline_extractor_18.write_fea("2018-2019")

    fea_weekly_extractor_16.write_fea("2016-2017")
    fea_weekly_extractor_17.write_fea("2017-2018")
    fea_weekly_extractor_18.write_fea("2018-2019")


def extract_we():
    we_deadline_extractor_16.write_fea("2016-2017")
    we_deadline_extractor_17.write_fea("2017-2018")
    we_deadline_extractor_18.write_fea("2018-2019")

    we_weekly_extractor_16.write_we("2016-2017")
    we_weekly_extractor_17.write_we("2017-2018")
    we_weekly_extractor_18.write_we("2018-2019")

# we_data_16 =
# we_deadlines_16 = [serie["deadline"] for serie in all[0]["series"]]

#
# print("DONE WE")
#
# combine_all_fea(['data/fea/2016-2017-formatted_data.json',
#                  'data/fea/2017-2018-formatted_data.json',
#                  'data/fea/2018-2019-formatted_data.json'])
#
# calc_fea('data/fea/2016-2017-formatted_data.json', start1617, end_sem_1617, "sem_1617", deadlines1617)
# calc_fea('data/fea/2017-2018-formatted_data.json', start1718, end_sem_1718, "sem_1718", deadlines1718)
# calc_fea('data/fea/2018-2019-formatted_data.json', start1819, end_sem_1819, "sem_1819", deadlines1819)
