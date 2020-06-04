import pickle
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_breast_cancer

def LoadData(dataset):
    if dataset == 'breast_cancer':
        # Breast Cancer Wisconsin
        data = load_breast_cancer()
        return data.data, data.target

    elif dataset == 'adult':
        # Adult
        pickle_in = open('datasets/adult.pkl', "rb")
        data = pickle.load(pickle_in)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        data['data'] = imp.fit_transform(data['data'])
        return data['data'], data['targets']

    elif dataset == 'compas':
        # Compas
        pickle_in = open('datasets/compas.pkl', "rb")
        data = pickle.load(pickle_in)
        return data['data'], data['targets']

    elif dataset == 'red_wine_quality':
        # Red Wine Quality
        pickle_in = open('datasets/red_wine.pkl', "rb")
        data = pickle.load(pickle_in)
        return data['data'], data['targets']

    elif dataset == 'vehicle':
        # Vehicle
        pickle_in = open('datasets/vehicle.pkl', "rb")
        data = pickle.load(pickle_in)
        return data['data'], data['targets']

    elif dataset == 'recidivism':
        # Recidivism
        pickle_in = open('datasets/recd.pkl', "rb")
        data = pickle.load(pickle_in)
        return data['data'], data['targets']

    elif dataset == 'german_credit':
        # German Credit
        pickle_in = open('datasets/german.pkl', "rb")
        data = pickle.load(pickle_in)
        return data['data'], data['targets']

    elif dataset == 'glass':
        # Glass
        pickle_in = open('datasets/glass.pkl', "rb")
        data = pickle.load(pickle_in)
        return data['data'], data['targets']

