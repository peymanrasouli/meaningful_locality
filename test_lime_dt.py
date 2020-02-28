import numpy as np
from load_data import LoadData
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.linear_model import Ridge
from meaningful_sampling import MeaningfulSampling
from random_sampling import RandomSampling
from interpretable_representation import InterpretableRepresentation
import warnings
warnings.filterwarnings('ignore')

class explanation_results():
    score = list()
    local_pred = list()

def kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

def forward_selection(data, labels, weights, num_features):
    clf = Ridge(alpha=0, fit_intercept=True)
    used_features = []
    for _ in range(min(num_features, data.shape[1])):
        max_ = -100000000
        best = 0
        for feature in range(data.shape[1]):
            if feature in used_features:
                continue
            clf.fit(data[:, used_features + [feature]], labels,
                    sample_weight=weights)
            score = clf.score(data[:, used_features + [feature]],
                              labels,
                              sample_weight=weights)
            if score > max_:
                best = feature
                max_ = score
        used_features.append(best)
    return np.array(used_features)

def data_explainer(instance2explain, training_data, blackbox, kernel_width, num_features = 5, meaningful_sampling=False):
    if meaningful_sampling:
        data, inverse = MeaningfulSampling(instance2explain, blackbox, training_data, N_samples=5000)
        labels = blackbox.predict(inverse)
        distances = pairwise_distances(
                data,
                data[0].reshape(1, -1),
        ).ravel()
        weights = kernel(distances,kernel_width)
        used_features = forward_selection(data, labels, weights, num_features)
        dt = tree.DecisionTreeClassifier()
        dt.fit(data[:,used_features], labels, sample_weight=weights)
        dt_labels = dt.predict(data[:,used_features])
        exp_res = explanation_results()
        exp_res.score = f1_score(labels, dt_labels, sample_weight=weights, average='weighted')
        exp_res.local_pred = int(dt.predict(data[0,used_features].reshape(1, -1)))

        return exp_res

    else:
        inverse = RandomSampling(instance2explain, training_data, num_samples=5000)
        data = InterpretableRepresentation(inverse)
        labels = blackbox.predict(inverse)
        distances = pairwise_distances(
                data,
                data[0].reshape(1, -1),
        ).ravel()
        weights = kernel(distances,kernel_width)
        used_features = forward_selection(data, labels, weights, num_features)
        dt = tree.DecisionTreeClassifier()
        dt.fit(data[:,used_features], labels, sample_weight=weights)
        dt_labels = dt.predict(data[:,used_features])
        exp_res = explanation_results()
        exp_res.score = f1_score(labels, dt_labels, sample_weight=weights, average='weighted')
        exp_res.local_pred = int(dt.predict(data[0,used_features].reshape(1, -1)))

        return exp_res

if __name__ == '__main__':

    # Defining the list of data sets
    datsets_list= {
        'ad': 'adult',
        'cp': 'compas',
        'gc': 'german_credit',
        'rc': 'recidivism',
        'bc': 'breast_cancer',
        'rw': 'red_wine_quality',
        'gl': 'glass',
        'vc': 'vehicle'
    }

    # Defining the list of black-boxes
    blackbox_list = {
         'lr': LogisticRegression,
         'rf': RandomForestClassifier,
         'gb': GradientBoostingClassifier,
         'ab': AdaBoostClassifier
    }

    for ds in datsets_list:
        # Loading data
        print("Data set = " + datsets_list[ds])
        X, y = LoadData(datsets_list[ds])

        # Kernel width
        kernel_width = float(np.sqrt(X.shape[1]) * .75)

        # Splitting the data set to train, test, and explain set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for bb in blackbox_list:
            # Creating and training black-box
            print("Black-box = " + bb)
            BlackBoxConstructor = blackbox_list[bb]
            blackbox = BlackBoxConstructor()
            blackbox.fit(X_train, y_train)

            # Generating explanations for the samples in the test set
            lime_score = []
            lime_pred = []
            slime_score = []
            slime_pred = []
            for i in range(X_test.shape[0]):
                # LIME
                lime_res = data_explainer(X_test[i], X, blackbox, kernel_width, num_features = 5, meaningful_sampling=False)
                lime_score.append(lime_res.score)
                lime_pred.append(lime_res.local_pred)

                # LIME with meaningful sampling
                slime_res = data_explainer( X_test[i], X, blackbox, kernel_width, num_features = 5, meaningful_sampling=True)
                slime_score.append(slime_res.score)
                slime_pred.append(slime_res.local_pred)

            bb_pred = blackbox.predict(X_test)
            lime_pred = np.asarray(lime_pred)
            slime_pred = np.asarray(slime_pred)

            # F1 Score
            lime_f1 = f1_score(bb_pred, lime_pred, average='weighted')
            slime_f1 = f1_score(bb_pred, slime_pred, average='weighted')
            print('LIME    F1 = ' + str(lime_f1))
            print('S-LIME  F1 = ' + str(slime_f1))
            print('\n')

            # Precision score
            lime_prec = precision_score(bb_pred, lime_pred, average='weighted')
            slime_prec = precision_score(bb_pred, slime_pred, average='weighted')
            print('LIME    Precision = ' + str(lime_prec))
            print('S-LIME  Precision = ' + str(slime_prec))
            print('\n')

            # Accuracy Score
            lime_acc = accuracy_score(bb_pred, lime_pred)
            slime_acc = accuracy_score(bb_pred, slime_pred)
            print('LIME    Accuracy = ' + str(lime_acc))
            print('S-LIME  Accuracy = ' + str(slime_acc))
            print('\n')

            # Average F1 Score of Neighborhood Data
            lime_neigh_f1 = np.mean(lime_score)
            slime_neigh_f1 = np.mean(slime_score)
            print('LIME    Neighborhood F1 = ' + str(lime_neigh_f1))
            print('S-LIME  Neighborhood F1 = ' + str(slime_neigh_f1))
            print('\n')