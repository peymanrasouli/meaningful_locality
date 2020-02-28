import numpy as np
from load_data import LoadData
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
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
        predictions = blackbox.predict_proba(inverse)
        label = np.argmax(predictions[0])
        predictions = predictions[:,label]
        distances = pairwise_distances(
                data,
                data[0].reshape(1, -1),
        ).ravel()
        weights = kernel(distances,kernel_width)
        used_features = forward_selection(data, predictions, weights, num_features)
        lr = Ridge(alpha=1, fit_intercept=True)
        lr.fit(data[:, used_features],predictions, sample_weight=weights)
        lr_preds = lr.predict(data[:,used_features])
        exp_res = explanation_results()
        exp_res.score = r2_score(predictions, lr_preds, sample_weight=weights)
        exp_res.local_pred = float(lr.predict(data[0,used_features].reshape(1, -1)))

        return exp_res

    else:
        inverse = RandomSampling(instance2explain, training_data, num_samples=5000)
        data = InterpretableRepresentation(inverse)
        predictions = blackbox.predict_proba(inverse)
        label = np.argmax(predictions[0])
        predictions = predictions[:,label]
        distances = pairwise_distances(
                data,
                data[0].reshape(1, -1),
        ).ravel()
        weights = kernel(distances,kernel_width)
        used_features = forward_selection(data, predictions, weights, num_features)
        lr = Ridge(alpha=1, fit_intercept=True)
        lr.fit(data[:, used_features],predictions, sample_weight=weights)
        lr_preds = lr.predict(data[:,used_features])
        exp_res = explanation_results()
        exp_res.score = r2_score(predictions, lr_preds, sample_weight=weights)
        exp_res.local_pred = float(lr.predict(data[0,used_features].reshape(1, -1)))

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
                slime_res = data_explainer(X_test[i], X, blackbox, kernel_width, num_features = 5, meaningful_sampling=True)
                slime_score.append(slime_res.score)
                slime_pred.append(slime_res.local_pred)

            prediction = blackbox.predict_proba(X_test)
            bb_pred = np.sort(prediction)[:, -1:]

            lime_pred = np.asarray(lime_pred)
            slime_pred = np.asarray(slime_pred)

            # Mean Absolute Error (MAE)
            lime_mae = mean_absolute_error(bb_pred, lime_pred)
            slime_mae = mean_absolute_error(bb_pred, slime_pred)
            print('LIME    MAE = ' + str(lime_mae))
            print('S-LIME  MAE = ' + str(slime_mae))
            print('\n')

            # Mean Squared Error (MSE)
            lime_mse = mean_squared_error(bb_pred, lime_pred)
            slime_mse = mean_squared_error(bb_pred, slime_pred)
            print('LIME    MSE = ' + str(lime_mse))
            print('S-LIME  MSE = ' + str(slime_mse))
            print('\n')

            # Average R² Score of Neighborhood Data
            lime_neigh_r2 = np.average(lime_score)
            slime_neigh_r2 = np.average(slime_score)
            print('LIME    Neighborhood R² = ' + str(lime_neigh_r2))
            print('S-LIME  Neighborhood R² = ' + str(slime_neigh_r2))
            print('\n')