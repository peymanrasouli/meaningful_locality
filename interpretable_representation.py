import numpy as np
from quartile_discretizer import QuartileDiscretization

def InterpretableRepresentation (dense_samples):
    """
    This function creates an interpretable (binary) representation
    of the data to be used by LIME explanation method. The original
    feature values are used by the blackbox model.

    :param dense_samples: data with original feature values
    :return: data with binary feature values
    """

    categorical_features = list(range(dense_samples.shape[1]))
    dense_samples_dc = QuartileDiscretization(dense_samples)
    interpretable_dense_samples = np.zeros(dense_samples.shape)

    for column in categorical_features:
        dense_column = dense_samples_dc[:,column]
        binary_column = np.array([1 if x == dense_column[0] else 0 for x in dense_column])
        binary_column[0] = 1
        interpretable_dense_samples[:, column] = binary_column

    return interpretable_dense_samples