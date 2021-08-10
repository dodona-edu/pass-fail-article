from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np
import random
import matplotlib.pyplot as plt
import csv

FIT_RND = 365112949
GSCV_RND = 34986
rnd = random.Random(1234)
rnd2 = random.Random(498327)

np.random.seed(84475)

weekly_names = ["01", "02", "03", "04", "05", "05_ev1", "06", "07", "08", "09", "10", "10_ev12"]
fea_weekly_names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "1_eval", "11", "12", "13",
                    "14", "15", "16", "17", "2_eval", "18", "19", "20"]


# Calculates the accuracy for each class
def class_accs(predicted_labels, test_labels, classes):
    acc = []
    for cl in classes:
        indices = [i for i, label in enumerate(test_labels) if label == cl]
        accuracy = metrics.accuracy_score([cl] * len(indices), predicted_labels[indices])
        acc.append(accuracy)

    return acc


def get_features(fac, sem, week_number, cpf, extractor):
    assert (fac == "we" or fac == "fea"), "Wrong faculty"

    if fac == "we":
        fts = np.genfromtxt(f'{cpf}feature_vectors/we_labels_{extractor}1617_series1.csv', dtype='str', delimiter=",")
        if 5 <= week_number < 11:
            fts = np.genfromtxt(f'{cpf}feature_vectors/we_labels_{extractor}1617_series5_eval.csv', dtype='str',
                                delimiter=",")
        elif week_number >= 11:
            fts = np.genfromtxt(f'{cpf}feature_vectors/we_labels_{extractor}1617_series10_eval.csv', dtype='str',
                                delimiter=",")

    if fac == "fea":
        sp = "sem_" if sem else ""
        fts = np.genfromtxt(f'{cpf}feature_vectors/fea_labels_{sp}1617_series1.csv', dtype='str', delimiter=",")
        if 10 <= week_number < 18:
            fts = np.genfromtxt(f'{cpf}feature_vectors/fea_labels_{sp}1617_series10_eval1.csv', dtype='str',
                                delimiter=",")
        if week_number >= 18:
            fts = np.genfromtxt(f'{cpf}feature_vectors/fea_labels_{sp}1617_series17_eval2.csv', dtype='str',
                                delimiter=",")

    return fts


# Prepare the data: shuffle the data (including the marks) and then separate them again
def prep_data(train, test, marks):
    test = np.column_stack((test, marks))

    np.random.shuffle(train)
    np.random.shuffle(test)

    train_labels, train_features = train[:, -1], train[:, :-1]
    marks, test_labels, test_features = test[:, -1], test[:, -2], test[:, :-2]

    return train_labels, train_features, test_labels, test_features, marks


# Plot histogram of scores and predicted labels
def plot_histogram_scores(plbls, albls, amarks, accuracy, name, cpf):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axisbelow(True)
    ax.yaxis.grid(linestyle='-')

    pred_lbl = []
    patch_handles, plots = [], []
    bottom = np.zeros(22)
    colors = ["red", "green"]
    # Iterate over all possible classes & calculate per mark the number of scores that were predicted to belong
    # to a certain class
    for i, l in enumerate(range(len(np.unique(albls)))):
        marks = [sum([1 if amarks[j] == m and plbls[j] == l else 0 for j in range(len(plbls))]) for m in range(-1, 21)]
        # marks = []
        # for m in range(-1, 21):
        #     tmp = sum([1 if int(amarks[j]) == m and plbls[j] == l else 0
        #                for j in range(len(plbls))])
        #     marks.append(tmp)
        pred_lbl.append(marks)
        patch_handle = plt.bar(np.arange(22), marks, 0.8, bottom=bottom, color=colors[i], linewidth=10)
        bottom += marks
        plots.append(patch_handle[0])

        for j, patch in enumerate(patch_handle.get_children()):
            h = patch.get_height()
            if h != 0 and h != 1:
                plt.text(patch.get_x() + patch.get_width() / 2, patch.get_y() + h / 2, '%s' % h,
                         ha='center', va='center', color="white")

    # Calculate max height, rounded to the nearest 5 above the value
    max_vals = max(np.add(pred_lbl[0], pred_lbl[1]))
    max_y = max_vals + (5 - (max_vals % 5))

    ax.set_xlabel('Actual marks of students')
    ax.set_ylabel('Nr of students')

    xlbls = ["no exam", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
             "17", "18", "19", "20"]
    plt.xticks(list(range(0, 22, 1)), xlbls, fontsize=14)
    plt.yticks(np.arange(0, max_y, 5), fontsize=14)
    ax.get_xticklabels()[0].set_rotation(90)

    plt.legend(tuple(plots), ("Predicted to fail", "Predicted to pass"))
    plt.title(f'Actual student marks vs pass/fail prediction \nAccuracy: {accuracy / 1:.2%}\n')

    fig.savefig(f"{cpf}{name}")

    plt.close(fig)


def plot_histogram(plbls, albls, amarks, accuracy, name, cpf):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axisbelow(True)
    ax.yaxis.grid(linestyle='-')

    pred_lbl = []
    patch_handles, plots = [], []
    bottom = np.zeros(22)
    colors = ["grey", "grey"]
    # Iterate over all possible classes & calculate per mark the number of scores that were predicted to belong
    # to a certain class
    for i, l in enumerate(range(len(np.unique(albls)))):
        marks = [sum([1 if amarks[j] == m and plbls[j] == l else 0 for j in range(len(plbls))]) for m in range(-1, 21)]
        # marks = []
        # for m in range(-1, 21):
        #     tmp = sum([1 if int(amarks[j]) == m and plbls[j] == l else 0
        #                for j in range(len(plbls))])
        #     marks.append(tmp)
        pred_lbl.append(marks)
        patch_handle = plt.bar(np.arange(22), marks, 0.8, bottom=bottom, color=colors[i], linewidth=10)
        bottom += marks
        plots.append(patch_handle[0])

        # for j, patch in enumerate(patch_handle.get_children()):
        # h = patch.get_height()
        # if h != 0 and h != 1:
        # plt.text(patch.get_x() + patch.get_width() / 2, patch.get_y() + h / 2, '%s' % h,
        # ha='center', va='center', color="white")

    # Calculate max height, rounded to the nearest 5 above the value
    max_vals = max(np.add(pred_lbl[0], pred_lbl[1]))
    max_y = max_vals + (5 - (max_vals % 5))

    ax.set_xlabel('Actual marks of students')
    ax.set_ylabel('Nr of students')

    xlbls = ["no exam", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
             "17", "18", "19", "20"]
    plt.xticks(list(range(0, 22, 1)), xlbls, fontsize=14)
    plt.yticks(np.arange(0, max_y, 5), fontsize=14)
    ax.get_xticklabels()[0].set_rotation(90)

    # plt.legend(tuple(plots), ("Predicted to fail", "Predicted to pass"))
    plt.title(f'Point distribution of students in 2018-2019\n')

    fig.savefig(f"{cpf}{name}")

    plt.close(fig)


def plot_percentages(probs, marks, name):
    fig, ax = plt.subplots(figsize=(10, 10))

    fig.tight_layout()
    plt.margins(x=0)
    plt.margins(y=0)

    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    successperc = np.array(probs)[:, -1]
    marks = [-1 if m == 21 else m for m in marks]

    plt.plot([9.5, 9.5], [-0.05, 1.05], linewidth=3, color='black')
    plt.plot([-2, 21], [0.5, 0.5], linewidth=3, color='black')

    plt.scatter(marks, successperc, c="black")
    #     plt.xticks(np.arange(0, 21))

    ax.set_xlabel('Actual marks of students')
    ax.set_ylabel('Success rate')

    xticks = list(range(-1, 21, 1))
    xlbls = ["no exam", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
             "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
    plt.xticks(xticks, xlbls, fontsize=14)
    plt.yticks(np.arange(0, 1.05, 0.1))

    ax.get_xticklabels()[0].set_rotation(90)
    plt.title(f'Actual student marks vs predicted success rate')

    plt.show()
    fig.savefig(name)


class TestModel:

    def __init__(self, model, feature_method, params, cv, scoring, feature_labels, cpf):
        self.model = model
        self.feature_method = feature_method
        self.params = params
        self.cv = cv
        self.scoring = scoring
        self.feature_labels = feature_labels
        self.cpf = cpf

        self.chosen_params = None
        self.fitted_model = None
        self.predicted_labels = None
        self.scores = None
        self.feature_importances = None

        self.probs = []

    def __find_params(self, train_features, train_labels):
        gs = GridSearchCV(self.model(random_state=GSCV_RND), self.params, cv=self.cv, scoring=self.scoring)
        gs.fit(train_features, train_labels)

        self.chosen_params = gs.best_params_

    def __fit_model(self, train_features, train_labels):
        forest = self.model(random_state=FIT_RND, **self.chosen_params)
        forest.fit(train_features, train_labels)
        self.fitted_model = forest
        # std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

        return self.feature_method(forest)

    def __score(self, test_labels, test_features):
        # tf = test_features[:, self.best_features]
        print(len(test_features))
        classes = sorted(np.unique(test_labels))
        self.predicted_labels = self.fitted_model.predict(test_features)

        class_acc = class_accs(self.predicted_labels, test_labels, classes)
        bal_acc = np.mean(class_acc)
        f1 = metrics.f1_score(test_labels, self.predicted_labels, pos_label=0)

        recall = metrics.recall_score(test_labels, self.predicted_labels, pos_label=0)

        tn, fp, fn, tp = metrics.confusion_matrix(test_labels, self.predicted_labels).ravel()

        self.scores = [class_acc[0], class_acc[1], bal_acc, f1, recall, tn, fp, fn, tp]

        # probs = self.fitted_model.predict_proba(test_features)
        # self.probs = probs

    def test_model(self, train_labels, train_features, test_labels, test_features):
        self.__find_params(train_features, train_labels)
        imp = self.__fit_model(train_features, train_labels)
        self.__score(test_labels, test_features)

        return self.scores, imp


class WeeklyTestModel(TestModel):

    def __init__(self, model, feature_method, params, cv, scoring, feature_labels, cpf):
        super().__init__(model, feature_method, params, cv, scoring, feature_labels, cpf)

    def test_weekly_we(self, train_data, test_data, marks, name1, epf):
        print("WEEKLY WE")
        fp_params = open(f"{self.cpf}plots/we/params_{name1}.csv", "w+")
        fp = open(f"{self.cpf}plots/we/{name1}.csv", "w+")
        all_scores = []
        i = 0
        for train_week, test_week in zip(train_data, test_data):
            print(i)

            fts = get_features("we", True, i, self.cpf, epf)

            trl, trf, tel, tef, act_marks = prep_data(train_week, test_week, marks)
            week_score, imps = self.test_model(trl, trf, tel, tef)
            all_scores.append(week_score)

            fp2 = open(f"{self.cpf}plots/we/{epf}feature_importances_{name1}_s{weekly_names[i]}.csv", "w")
            w1 = csv.writer(fp2, delimiter=",")

            imps = imps.tolist()
            if len(imps) < 10:
                imps = imps[0]

            name = f"plots/we/{epf}{name1}_s{weekly_names[i]}.png"
            # name2 = f"plots/we/scatter_{name1}_s{weekly_names[i]}.png"
            name3 = f"plots/we/histogramall_{name1}_s{weekly_names[i]}.png"
            plot_histogram(self.predicted_labels, tel, act_marks, week_score[2], name3, self.cpf)
            plot_histogram_scores(self.predicted_labels, tel, act_marks, week_score[2], name, self.cpf)
            # plot_percentages(self.probs, act_marks, name2)
            print(week_score)
            print(str(len(fts)) + " " + str(len(imps)))
            w1.writerow(["feature", "importance"])
            for x, y in zip(fts, imps):
                w1.writerow([x, y])

            fp.write(f"series_{weekly_names[i]}, {week_score}")
            tmp_params = [f"{k}: {v}" for k, v in self.chosen_params.items()]
            fp_params.write(f"series_{weekly_names[i]}, {','.join(tmp_params)}\n")
            i += 1

        return all_scores

    def test_weekly(self, train_data, test_data, marks, name1, sp, epf):
        fp = open(f"{self.cpf}plots/fea/{name1}.csv", "a+")
        fp_params = open(f"{self.cpf}plots/fea/params_{name1}.csv", "w+")

        all_scores = []
        i = 0
        for train_week, test_week in zip(train_data, test_data):
            print(i)

            sem = True if sp else False
            fts = get_features("fea", sem, i, self.cpf, epf)

            trl, trf, tel, tef, act_marks = prep_data(train_week, test_week, marks)
            week_score, imps = self.test_model(trl, trf, tel, tef)
            all_scores.append(week_score)

            imps = imps.tolist()
            if len(imps) < 10:
                imps = imps[0]

            fp2 = open(f"{self.cpf}plots/fea/{epf}{sp}feature_importances_{name1}_s{fea_weekly_names[i]}.csv", "w")
            w1 = csv.writer(fp2, delimiter=",")

            name = f"plots/fea/{epf}{name1}_s{fea_weekly_names[i]}.png"
            name2 = f"plots/fea/{epf}scatter_{name1}_s{fea_weekly_names[i]}.png"
            plot_histogram_scores(self.predicted_labels, tel, act_marks, week_score[2], name, self.cpf)
            # plot_percentages(self.probs, act_marks, name2)
            print(week_score)
            print(str(len(fts)) + " " + str(len(imps)))
            w1.writerow(["feature", "importance"])
            for x, y in zip(fts, imps):
                w1.writerow([x, y])

            fp.write(f"series_{fea_weekly_names[i]}, {week_score}")
            tmp_params = [f"{k}: {v}" for k, v in self.chosen_params.items()]
            fp_params.write(f"series_{fea_weekly_names[i]}, {','.join(tmp_params)}\n")
            i += 1

        return all_scores


class TestModel1Year(TestModel):

    def __init__(self, model, feature_method, params, cv, scoring, feature_labels, cpf):
        super().__init__(model, feature_method, params, cv, scoring, feature_labels, cpf)

    def test_model1year_weekly(self, all_data, marks, name, fc="we"):
        fp = open(f"{self.cpf}plots/{fc}/{name}.csv", "w")
        if fc == "fea":
            fp_params = open(f"{self.cpf}plots/fea/params_{name}.csv", "w+")
        else:
            fp_params = open(f"{self.cpf}plots/we/params_{name}.csv", "w+")

        all_scores = []
        i = 0

        # Calculate weekly scores
        # for data in all_data:
        # while i == 0:
        for data in all_data:
            # data = all_data[0]
            print(i)
            scores = []
            data = np.column_stack((data, marks))
            np.random.shuffle(data)
            marks, data_labels, data_features = data[:, -1], data[:, -2], data[:, :-2]

            skf = StratifiedKFold(n_splits=5)

            predicted_values, split_marks = [], []

            # 5-fold cross validation in split
            j = 0
            for train_index, test_index in skf.split(data_features, data_labels):
                x_train, x_test = data_features[train_index], data_features[test_index]
                y_train, y_test = data_labels[train_index], data_labels[test_index]
                marks_cur_split = marks[test_index]

                sc, imp = self.test_model(y_train, x_train, y_test, x_test)
                scores.append(sc)

                predicted_values.append(self.predicted_labels)
                split_marks.append(marks_cur_split)
                if fc == "fea":
                    tmp_params = [f"{k}: {v}" for k, v in self.chosen_params.items()]
                    fp_params.write(f"series_{fea_weekly_names[i]}_{j}, {','.join(tmp_params)}\n")
                else:
                    tmp_params = [f"{k}: {v}" for k, v in self.chosen_params.items()]
                    fp_params.write(f"series_{weekly_names[i]}_{j}, {','.join(tmp_params)}\n")

                j += 1

            scores.append(np.mean(scores, axis=0))
            all_scores.append(scores)

            if fc == "fea":
                plot_name = f"plots/{fc}/{name}_s{fea_weekly_names[i]}.png"
            else:
                plot_name = f"plots/{fc}/{name}_s{weekly_names[i]}.png"
            predicted_labels = [elem for sublist in predicted_values for elem in sublist]
            act_marks = [elem for sublist in split_marks for elem in sublist]
            plot_histogram_scores(predicted_labels, [0, 1], act_marks, scores[-1][2], plot_name, self.cpf)
            i += 1

        writer1 = csv.writer(fp, delimiter=",")
        formatted_scores = []
        for i, x in enumerate(all_scores):

            it_scores = []
            for iteration_score in x:
                it_score = list(map(lambda e: str(round(e, 3)), iteration_score))
                it_scores.append(it_score)
            formatted_scores.append(it_scores)
            # scs = [elem for sublist in x for elem in sublist]

            # sc = [f"{name} series_{i:02}"] + list(map(lambda e: str(round(e, 3)), scs))

            writer1.writerows(it_scores)

        return formatted_scores
