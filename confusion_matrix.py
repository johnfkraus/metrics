# author: John Kraus
# The sklearn.metrics confusion_matrix method (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) (1) is deprecated; and (2) produces unexpected results for single-class edge cases.
# This confusion_matrix.py package is a substitute for sklearn.metrics.confusion_matrix for binary classification.
# Returns ndarray of shape (2, 2)
# Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
import numpy as np
# from fractions import Fraction
# reference regarding formatting:  https://stackoverflow.com/questions/8234445/format-output-string-right-alignment
# TODO: expand functionality to 

def get_binary_confusion_matrix(y_true, y_pred, verbose=False):
    # given two numpy integer arrays of zeros and ones
    # returns a 2x2 numpy array representing a confusion matrix (cm)
    # cm is eq to [[tn , fp],
    #             [fn , tp]], a numpy array
    # reference https://kawahara.ca/how-to-compute-truefalse-positives-and-truefalse-negatives-in-python-for-binary-classification-problems/
    if verbose:
        print_array_data(y_true, y_pred)
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    if verbose:
        print('TN: %i, FP: %i, FN: %i, TP: %i' % (TN,FP,FN,TP))
    cm = np.array([[TN,FP],[FN,TP]])
    assert(cm.size == 4)
    assert(TP + TN == get_correct(y_true, y_pred).sum())
    return cm


def make_random_test_array(array_len=20, num_classes=2):
    # return an numpy array filled randomly with ones and zeros
    rand_generator = np.random.default_rng()
    np_array = rand_generator.integers(0, num_classes, array_len)
    return np_array


def test_make_cm_from_edge_case_zeros(array_len=20):
    y_true = np.zeros(array_len, dtype=int)
    y_pred = np.zeros(array_len, dtype=int)
    cm = get_binary_confusion_matrix(y_true, y_pred)
    make_cm_assertions(cm, array_len)
    return True


def test_make_cm_from_edge_case_ones(array_len=20):
    y_true = np.ones(array_len, dtype=int)
    y_pred = np.ones(array_len, dtype=int)
    cm = get_binary_confusion_matrix(y_true, y_pred)
    make_cm_assertions(cm, array_len)
    return True


def make_cm_assertions(cm, array_len):
    assert(cm.sum() == array_len)
    assert(cm.size == 4)


def test_make_cm_from_arbitrary_input():
    # sklearn.metrics import confusion_matrix breaks in the following cases, so we return a custom-made confusion matrix.
    y_true = np.array(list([0,0,0,0,1,0,1]))
    y_pred = np.array(list([1,0,0,0,0,0,1]))
    print_array_data(y_true, y_pred)
    cm = get_binary_confusion_matrix(y_true, y_pred)
    make_cm_assertions(cm, y_true.size)
    return True


def get_correct(y_true, y_pred):
    # return an numpy array showing which predicted values were correct?
    return np.where(y_true==y_pred, 1, 0)


def print_array_data(y_true, y_pred):
    correct = get_correct(y_true, y_pred)
    print("y_true :", y_true, y_true.sum())
    print("y_pred :", y_pred, y_pred.sum())
    print("correct:", correct, correct.sum())


def test_make_cm_from_random_input(array_len=20):
    # return a confusion matrix (cm) from two numpy arrays; the arrays contain random ones or zeros
    y_true = make_random_test_array(array_len)
    y_pred = make_random_test_array(array_len)
    cm = get_binary_confusion_matrix(y_true, y_pred)
    make_cm_assertions(cm, array_len)
    return True


if __name__ == "__main__":

    array_length = np.random.randint(3, 31)
    print('>>>> running: test_make_cm_from_random_input() PASS = ', test_make_cm_from_random_input(array_length))

    print('>>>> running: test_make_cm_from_cm_edge_case_zeros() PASS = ', test_make_cm_from_edge_case_zeros(array_len=np.random.randint(3,31)))
    print('>>>> running: test_make_cm_from_cm_edge_case_ones() PASS = ',  test_make_cm_from_edge_case_ones(array_len=np.random.randint(3,31)))
