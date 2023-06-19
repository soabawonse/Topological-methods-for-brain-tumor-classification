import pickle
import typing
import glob
import time
import os

from teaspoon.SP.adaptivePart import Partitions
from teaspoon.TDA import Persistence as pP
import teaspoon.ML.PD_Classification
import teaspoon.ML.feature_functions
import teaspoon.ML.Base

import sklearn.model_selection
import sklearn.metrics
import sklearn.svm

import pandas as pd
import numpy as np
import tqdm

import machine_learning.ml_utils
import machine_learning
import utils


def create_df(data: typing.Dict, pts: typing.List) -> pd.DataFrame:
    """Helper function which creates DataFrame of features data dictionary"""
    dgm_0 = []
    dgm_1 = []
    labels = []
    for pt, label in pts:
        pt_data = data[pt]

        for seq in pt_data:
            dgm_0.append(seq["all"][0])
            dgm_1.append(seq["all"][1])
            labels.append(label)
    df = pd.DataFrame(
        data={
            "dgm_0": dgm_0,
            "dgm_1": dgm_1,
            "label": labels,
        }
    )
    return df


def featurize_template(train_df, test_df, dgm_col, params):
    """Helper function to featurize persistence diagrams"""
    # Get the portions of the test data frame with diagrams and concatenate into giant series:
    allDgms = pd.concat((train_df[label] for label in dgm_col))

    if params.useAdaptivePart == True:
        # Hand the series to the makeAdaptivePartition function
        params.makeAdaptivePartition(allDgms, meshingScheme="DV")
    else:
        # TODO this should work for both interp and tents but doesn't yet
        params.makeAdaptivePartition(allDgms, meshingScheme=None)

    #  delta = min([x.min() for x in train_df[dgm_col[0]]])
    #  params.delta = delta
    #  params.epsilon = delta * 0.9

    listOfG_train = []
    for dgmColLabel in dgm_col:
        G_train = teaspoon.ML.Base.build_G(train_df[dgmColLabel], params)
        listOfG_train.append(G_train)

    # feature matrix for training set
    X_train = np.concatenate(listOfG_train, axis=1)

    listOfG_test = []
    for dgmColLabel in dgm_col:
        G_test = teaspoon.ML.Base.build_G(test_df[dgmColLabel], params)
        listOfG_test.append(G_test)

    # feature matrix for test set
    X_test = np.concatenate(listOfG_test, axis=1)

    return X_train, X_test


def load_data(
    data_dir: str,
    sleep_type: str,
    filter_func: typing.Callable,
) -> typing.Tuple[typing.Dict, typing.List]:
    """Function which loads persistence diagram data and labels based on filter function passed
    to function

    Args:
        data_dir (str): String representation of directory containing data
        sleep_type (str): String which specifies sleep type. Can be one of
            "W_before", "N1", "N2", "N3", "R"
        filter_func (callable): Filter function to distinguish groups

    Returns:
        data (dict): Dictionary containing data
        pt_label (list): List containing tuples with patient ID and integer (1
            or 0) representing group
    """
    fnames = utils.get_pd_subdirectories(data_dir, sleep_type, feature_type="takens")
    data = utils.load_pkl_data_from_dir(fnames)

    pts = list(data.keys())
    pt_label = []

    for k_pt in tqdm.tqdm(pts, desc="Labelling Patients"):
        metadata_fname = glob.glob(os.path.join(data_dir, k_pt, "metadata", "*.pkl"))[0]

        with open(metadata_fname, "rb") as f:
            metadata = pickle.load(f)

        metadata["k_pt"] = k_pt
        group = filter_func(metadata)

        if group == 1:
            pt_label.append((k_pt, 1))
        elif group == -1:
            pt_label.append((k_pt, 0))

    return data, pt_label


def featurize_split(
    data: typing.Dict,
    train_pts: typing.List[str],
    test_pts: typing.List[str],
) -> typing.Tuple[np.ndarray, typing.List, np.ndarray, typing.List]:
    """Helper function to featurize training and testing splits independently

    Args:
        data (dict): Dictionary contaiing data
        train_pts (list): List containing patient IDs of patients to put in
            training set
        test_pts (list): List containing patient IDs of patients to put in
            testing set

    Returns:
        X_train (np.ndarray): Numpy array of training data features
        y_train (list): List of labels of training set
        X_test (np.ndarray): Numpy array of test data features
        y_test (list): List of labels of test set
    """
    t_start = time.time()

    train_df = create_df(data, train_pts)
    test_df = create_df(data, test_pts)

    params = teaspoon.ML.Base.ParameterBucket()
    params.feature_function = teaspoon.ML.feature_functions.interp_polynomial
    params.d = 20
    params.jacobi_poly = "cheb1"  # choose the interpolating polynomial
    params.useAdaptivePart = False
    params.clf_model = sklearn.svm.SVC
    params.TF_Learning = True

    dgm_cols = ["dgm_0"]
    print(f"Feature Columns: {dgm_cols}")
    X_train, X_test = featurize_template(
        train_df,
        test_df,
        dgm_cols,
        params,
    )
    t_end = time.time()
    print(f"Data featurized! Runtime: {t_end-t_start:.3f}")

    y_train = train_df["label"].tolist()
    y_test = test_df["label"].tolist()

    return X_train, y_train, X_test, y_test


def train(data_dir, sleep_type, filter_func):
    """Function to train model from the `machine_learning.models` subpackage on
    persistence diagrams featurized using template functions based on
    stratification from passed filter_func

    Args:
        data_dir (str): String representation of directory containing data
        sleep_type (str): String which specifies sleep type. Can be one of
            "W_before", "N1", "N2", "N3", "R"
        filter_func (callable): Filter function to distinguish groups

    """
    data, pt_label = load_data(data_dir, sleep_type, filter_func)

    kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=2023)

    for idx, (train_idx, test_idx) in enumerate(kf.split(pt_label)):
        print(f"Training fold {idx}")
        train_pts = [pt_label[t_idx] for t_idx in train_idx]
        test_pts = [pt_label[t_idx] for t_idx in test_idx]

        X_train, y_train, X_test, y_test = featurize_split(data, train_pts, test_pts)

        print("Train:")
        machine_learning.utils.get_label_distribution(y_train)

        print("Test:")
        machine_learning.utils.get_label_distribution(y_test)
        print()

        model = machine_learning.models.FCNN(
            num_features=X_train.shape[1], num_classes=2
        )
        trainer = machine_learning.models.NN_Trainer(model)
        machine_learning.utils.train_model(X_train, y_train, X_test, y_test, trainer)
