from sklearn.ensemble import RandomForestClassifier
from treeinterpreter.treeinterpreter import treeinterpreter as ti
from random_sampling import RandomSampling
from quartile_discretizer import QuartileDiscretization
from sturges_discretizer import SturgesDiscretization
from sample_manipulation import SampleManipulation
from interpretable_representation import InterpretableRepresentation

def MeaningfulSampling(instance2explain, blackbox, training_data, N_samples):
    """
    This function performs dense data generation for the instance2explain.
    It starts by randomly generating data points using the distribution of
    training data, and then making them closer to the instance2explain
    by considering similarities between feature values and feature importance.
    """

    # Generating random data using the distribution of training data
    # Discretizing random data for comparison of feature values
    random_samples = RandomSampling(instance2explain, training_data, N_samples)
    random_samples_dc = QuartileDiscretization(random_samples)

    # Constructing a random forest classifier as surrogate model
    surrogate_model = RandomForestClassifier(n_estimators=10)
    surrogate_model.fit(random_samples, blackbox.predict(random_samples))

    # Extracting feature contributions using TreeIntepreter
    # Discretizing contributions for comparison of feature importance
    prediction, bias, contributions = ti.predict(surrogate_model, random_samples)
    contributions_dc = SturgesDiscretization(contributions)

    # Making a dense neighborhood w.r.t instance2explain
    dense_samples = SampleManipulation(prediction, random_samples, random_samples_dc, contributions_dc)

    # Creating a sparse interpretable representation of data
    interpretable_dense_samples = InterpretableRepresentation(dense_samples)

    return interpretable_dense_samples, dense_samples