import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)


def sigmoid(x):
    """
    Applies sigmoid function
    """
    return 1 / (1 + np.exp(-x))


def update_df(df, top_features):
    """
    Updates expression matrix by handling duplicates
    :param df: expression matrix
    :param top_features: top important features from the model
    :return: updated matrix
    """
    df = df[top_features]
    rem, new = {}, []
    for i in df.columns:
        if i not in rem:
            rem[i] = 1
            new.append(i)
        else:
            new.append(i + '-' + str(rem[i]))
            rem[i] += 1
    df.columns = new
    return df
