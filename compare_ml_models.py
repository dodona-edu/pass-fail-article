import csv

from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn import preprocessing

from TestModel import WeeklyTestModel, TestModel1Year
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import math
import argparse

scaler = preprocessing.StandardScaler()

random_forest_params = {'n_estimators': list(range(10, 101, 15)), 'min_samples_leaf': [5, 10, 20, 40]}
log_reg_params = {
    'tol':  [1 * math.exp(-6), 1 * math.exp(-2)],
    'C': [0.1, 1.0, 1.5],
    'solver': ['liblinear', 'sag',  'saga'],
    'max_iter': [500]
}
sgd_params = {
    'tol':  [1 * math.exp(-6), 1 * math.exp(-2)],
    'max_iter': [1000],
    'loss': ['hinge', 'modified_huber', 'perceptron'],
    'penalty': ['l1', 'elasticnet'],
    'eta0': [0.01],
    'learning_rate': ['optimal', 'adaptive']
}
svm_params = {
    'tol':  [1 * math.exp(-6), 1 * math.exp(-2)],
    'C': [0.1, 1.0, 1.5],
    'loss': ['squared_hinge'],
    'penalty': ['l2'],
    'max_iter': [50000]
}
nb_params = {}

we_weekly_names = ["1", "2", "3", "4", "5", "5_eval", "6", "7", "8", "9", "10", "10_eval"]
fea_weekly_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10_eval1", "11", "12", "13",
                    "14", "15", "16", "17", "17_eval2", "18l", "19l", "20l"]


def flatten(l):
    return [item for sublist in l for item in sublist]


def random_forest_features(model: RandomForestClassifier):
    return model.feature_importances_


def non_random_forest_features(model):
    return model.coef_


def logistic_regression_features(model: LogisticRegression):
    return model.coef_


def stochastic_gd_features(model: SGDClassifier):
    return model.coef_


def svm_features(model: LinearSVC):
    return model.coef_


def naive_bayes_features(model: MultinomialNB):
    return model.coef_


def read_weekly_data(base, amount, names):
    weekly_data = []
    for i in range(amount):
        week = np.loadtxt(f"{base}{names[i]}.csv", delimiter=",")
        wkf, wkl = week[:, :-1], week[:, -1]
        wkf = scaler.fit_transform(wkf)
        wk = np.column_stack((wkf, wkl))
        weekly_data.append(wk)
    return weekly_data


def test_weekly(model, test, train, marks, name, scores_fp):
    if fac == "we":
        print("FAC == WE")
        scores = model.test_weekly_we(test, train, marks, name, epf)
    else:
        print("FAC == FEA")
        scores = model.test_weekly(test, train, marks, name, sp, epf)

    scs = []
    for i, x in enumerate(scores):
        towrite = f"series {i:02}: {x}\n"

        sc = [f"{name} series_{i:02}"] + list(map(lambda e: str(round(e, 3)), x))
        scs.append(sc)

        scores_fp.write(towrite)
    return scs


def test_1_year(model, test, marks, name, scores_fp):
    scores = model.test_model1year_weekly(test, marks, name, fc="fea")
    # scores, all_scores = model.test_model1year(test, marks, name)
    # scs = []
    # for i in range(5):
    #     x = scores[i]
    #     towrite = f"iteration_{i}: {x}\n"
    #
    #     sc = [f"{name} iteration_{i:02}"] + list(map(lambda e: str(round(e, 3)), x))
    #     scs.append(sc)
    #
    #     scores_fp.write(towrite)
    #
    # sc = [f"{name} MEAN"] + list(map(lambda e: str(round(e, 3)), all_scores))
    # scs.append(sc)
    # towrite = f"MEAN: {all_scores}\n"
    # scores_fp.write(towrite)
    headers = ["prediction", "acc_failed", "acc_passed", "bal_acc", "f1", "recall", "tn", "fp", "fn", "tp"]
    writer = csv.writer(scores_fp, delimiter=",")
    writer.writerow(headers)
    series = ["part1", "part2", "part3", "part4", "part5", "MEAN"]
    for j, sc in enumerate(scores):
        ths = []
        for i, scs in enumerate(sc):
            scs.insert(0, f"{name} series_{j+1:02} {series[i]}")
            ths.append(scs)
        writer.writerows(ths)

    return scores


def add_data(data1, data2):
    all_data = []
    for w1, w2 in zip(data1, data2):
        d = np.concatenate((w1, w2))
        all_data.append(d)
    return all_data


def train_2_year(model, prefix):
    sc1 = test_weekly(model, d1617_1718, d1819, m1819, f"{epf}{sp}{prefix}_train1617_1718_test1819",
                      open(f"{cpf}plots/{fac}/scores/{epf}{sp}{prefix}_train1617_1718_test1819.txt", "w"))
    sc2 = test_weekly(model, d1617_1819, d1718, m1718, f"{epf}{sp}{prefix}_train1617_1819_test1718",
                      open(f"{cpf}plots/{fac}/scores/{epf}{sp}{prefix}_train1617_1819_test1718.txt", "w"))
    sc3 = test_weekly(model, d1718_1819, d1617, m1617, f"{epf}{sp}{prefix}_train1718_1819_test1617",
                      open(f"{cpf}plots/{fac}/scores/{epf}{sp}{prefix}_train1718_1819_test1617.txt", "w"))
    return sc1, sc2, sc3


def train_1_year_test_2_year(model, prefix):
    ## Test 1 year on 1 other year
    sc1 = test_weekly(model, d1617, d1718, m1718, f"{epf}{sp}{prefix}_train1617_test1718",
                      open(f"{cpf}plots/{fac}/scores/{epf}{sp}{prefix}_train1617_test1718.txt", "w"))
    sc2 = test_weekly(model, d1617, d1819, m1819, f"{epf}{sp}{prefix}_train1617_test1819",
                      open(f"{cpf}plots/{fac}/scores/{epf}{sp}{prefix}_train1617_test1819.txt", "w"))

    sc3 = test_weekly(model, d1718, d1617, m1617, f"{epf}{sp}{prefix}_train1718_test1617",
                      open(f"{cpf}plots/{fac}/scores/{epf}{sp}{prefix}_train1718_test1617.txt", "w"))
    sc4 = test_weekly(model, d1718, d1819, m1819, f"{epf}{sp}{prefix}_train1718_test1819",
                      open(f"{cpf}plots/{fac}/scores/{epf}{sp}{prefix}_train1718_test1819.txt", "w"))

    sc5 = test_weekly(model, d1819, d1617, m1617, f"{epf}{sp}{prefix}_train1819_test1617",
                      open(f"{cpf}plots/{fac}/scores/{epf}{sp}{prefix}_train1819_test1617.txt", "w"))
    sc6 = test_weekly(model, d1819, d1718, m1718, f"{epf}{sp}{prefix}_train1819_test1718",
                      open(f"{cpf}plots/{fac}/scores/{epf}{sp}{prefix}_train1819_test1718.txt", "w"))
    return sc1, sc2, sc3, sc4, sc5, sc6


def train_1_year(model, prefix):
    sc1 = test_1_year(model, d1617, m1617, f"{epf}{prefix}_all1617",
                      open(f"{cpf}plots/{fac}/scores/{epf}{prefix}_all1617.txt", "w"))
    sc2 = test_1_year(model, d1718, m1718, f"{epf}{prefix}_all1718",
                      open(f"{cpf}plots/{fac}/scores/{epf}{prefix}_all1718.txt", "w"))
    sc3 = test_1_year(model, d1819, m1819, f"{epf}{prefix}_all1819",
                      open(f"{cpf}plots/{fac}/scores/{epf}{prefix}_all1819.txt", "w"))
    return sc1, sc2, sc3


def run_scorer(scorer, prefix, model, model_method, params):
    fp = open(f"{cpf}plots/{fac}/scores/all_{epf}{sp}{prefix}.txt", "w")
    headers = ["prediction", "acc_failed", "acc_passed", "bal_acc", "f1", "recall", "tn", "fp", "fn", "tp"]
    writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(headers)
    feature_labels = np.genfromtxt(f'{cpf}feature_vectors/{pf}labels_{epf}{sp}1617_series10.csv', dtype='str', delimiter=",")

    m1 = WeeklyTestModel(model, model_method, params, 10, scorer, feature_labels, cpf)
    m2 = TestModel1Year(model, model_method, params, 10, scorer, feature_labels, cpf)

    scores1 = train_2_year(m1, prefix)
    scores3 = train_1_year(m2, prefix)
    scores2 = train_1_year_test_2_year(m1, prefix)

    all_scores = flatten(scores1) + flatten(scores2) + flatten(flatten(scores3))
    # all_scores = scores1 + scores2  # + scores3

    writer.writerows(all_scores)
    fp.close()


def run_random_forest(scorer, prefix):
    run_scorer(scorer, f"rf_{prefix}", RandomForestClassifier, random_forest_features, random_forest_params)


def run_logistic_regression(scorer, prefix):
    run_scorer(scorer, f"lg_{prefix}", LogisticRegression, non_random_forest_features, log_reg_params)


def run_gd(scorer, prefix):
    run_scorer(scorer, f"gd_{prefix}", SGDClassifier, non_random_forest_features, sgd_params)


def run_svm(scorer, prefix):
    run_scorer(scorer, f"svm_{prefix}", LinearSVC, non_random_forest_features, svm_params)


def run_nb(scorer, prefix):
    run_scorer(scorer, f"nb_{prefix}", MultinomialNB, non_random_forest_features, nb_params)


# def run_all():
#     fp = open(f"{cpf}plots/{fac}/scores/all.txt", "w")
#     headers = ["prediction", "acc_failed", "acc_passed", "bal_acc", "f1", "recall", "tn", "fp", "fn", "tp"]
#     writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(headers)
#
#     scorers = ["balanced_accuracy", "accuracy", recall_scorer, f1_scorer]
#     prefixes = ["bal_acc", "acc", "recall", "f1"]
#
#     train_2_year_names = ["train1617_1718_test1819", "train1617_1819_test1718", "train1718_1819_test1617"]
#     train_1_year_test_2_names = ["train1617_test1718", "train1617_test1819", "train1718_test1617", "train1718_test1819",
#                                  "train1819_test1617", "train1819_test1718"]
#     train_1_year_names = ["all1617", "all1718", "all1819"]
#
#     feature_labels = np.genfromtxt(
#         f'{cpf}feature_vectors/{pf}labels_1617_series18_eval.csv', dtype='str',
#         delimiter=",")
#
#     for i, prefix in enumerate(prefixes):
#         # m1 = WeeklyTestModel(RandomForestClassifier, random_forest_params, 10, scorers[i], feature_labels)
#         m2 = TestModel1Year(RandomForestClassifier, random_forest_params, 10, scorers[i], feature_labels, cpf)
#
#         # scores1 = train_2_year(m1, prefix)
#         # scores2 = train_1_year_test_2_year(m1, prefix)
#         scores3 = train_1_year(m2, prefix)
#
#         all_scores = scores3  # scores1 + scores2 + scores3
#
#         writer.writerows(all_scores)
#
#     fp.close()


model_args = {
    "rf": run_random_forest,
    "lg": run_logistic_regression,
    "sgd": run_gd,
    "svm": run_svm
}


parser = argparse.ArgumentParser()
parser.add_argument("faculty")
parser.add_argument("ml")
parser.add_argument("-f", "--extractor", action="store_true")
parser.add_argument("-s", "--semester", help="Use semester data, only compatible with fea", action="store_true")
parser.add_argument("scorer")
parser.add_argument("-c", "--cloud", help="Is run on Intel AI Devcloud",
                    action="store_true")

args = parser.parse_args()

run_method = model_args[args.ml]

if args.faculty == "we":
    pf = "we_"
    fac = "we"
    am = 12
    names = we_weekly_names
    print("WEWE")

elif args.faculty == "fea":
    pf = "fea_"
    fac = "fea"
    am = 22
    names = fea_weekly_names
else:
    raise ValueError("Faculty argument must be either 'we' or 'fea'")

if args.cloud:
    cpf = "/home/u25721/Thesis/performance_prediction/"
else:
    cpf = ""

if args.semester:
    sp = "sem_"
    am = 14
    pf = "fea_"
else:
    sp = ""

epf = ""
if args.extractor:
    epf = "fea_extractor_"
elif fac == "we":
    epf = "we_extractor_"

## READ IN ALL DATA
m1617 = np.loadtxt(f"{cpf}feature_vectors/{pf}marks1617.csv", delimiter=",")
m1718 = np.loadtxt(f"{cpf}feature_vectors/{pf}marks1718.csv", delimiter=",")
m1819 = np.loadtxt(f"{cpf}feature_vectors/{pf}marks1819.csv", delimiter=",")

d1617 = read_weekly_data(f"{cpf}feature_vectors/{pf}features_{epf}{sp}1617_series", am, names)
d1718 = read_weekly_data(f"{cpf}feature_vectors/{pf}features_{epf}{sp}1718_series", am, names)
d1819 = read_weekly_data(f"{cpf}feature_vectors/{pf}features_{epf}{sp}1819_series", am, names)

d1617_1718 = add_data(d1617, d1718)  # d1617 + d1718
d1617_1819 = add_data(d1617, d1819)  # d1617 + d1819
d1718_1819 = add_data(d1718, d1819)  # d1718 + d1819

f1_scorer = make_scorer(f1_score, pos_label=0)
recall_scorer = make_scorer(recall_score, pos_label=0)

# run_all()

if args.scorer == "balanced_accuracy":
    run_method("balanced_accuracy", "bal_acc")
elif args.scorer == "accuracy":
    run_method("accuracy", "acc")
elif args.scorer == "recall":
    run_method(recall_scorer, "recall")
elif args.scorer == "f1":
    run_method(f1_scorer, "f1")
