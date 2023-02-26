# author: John Kraus
# The sklearn.metrics confusion_matrix method (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) (1) is deprecated; and (2) produces unexpected results for single-class edge cases.
# This confusion_matrix.py package is a replacement for sklearn.metrics.confusion_matrix for binary classification.
# Cndarray of shape (n_classes, n_classes)
# Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
import numpy as np
# from fractions import Fraction
# reference regarding formatting:  https://stackoverflow.com/questions/8234445/format-output-string-right-alignment
# TODO: expand functionality to 

def get_confusion_matrix_binary(y_true, y_pred, verbose=False):
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
    cm = make_cm(y_true, y_pred)
    make_cm_assertions(cm, array_len)
    return True


def test_make_cm_from_edge_case_ones(array_len=20):
    y_true = np.ones(array_len, dtype=int)
    y_pred = np.ones(array_len, dtype=int)
    cm = make_cm(y_true, y_pred)
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
    cm = make_cm(y_true, y_pred)
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


# def print_metrics_from_cm(cm):
#     # cm eq [[tn , fp],
#     #       [fn , tp]], a numpy array
#     # print('cm:', cm)
#     true_negative, false_positive, false_negative, true_positive = cm.ravel()

#     # Recall metric: What proportion of actual positives was identified correctly?
#     if (true_positive + false_negative) > 0:
#         recall = true_positive / (true_positive + false_negative)
#     else:
#         recall = -1  # or undefined?

#     if not ((true_positive is None) or (false_positive is None)):
#         # Precision: TP/(TP+FP); What proportion of positive identifications was actually correct?
#         if (true_positive + false_positive) > 0:
#             precision =  true_positive / (true_positive + false_positive)
#         else:
#             precision = -1  # undefined?

#     try:
#         total_obs = (true_positive + true_negative + false_positive + false_negative)
#     except ValueError as e:
#         print('error 118', e)

#     # Accuracy: (TP+TN)/(TP+TN+FP+FN)
#     # Informally, accuracy is the fraction of predictions our model got right.
#     if total_obs > 0:
#         accuracy = (true_positive + true_negative) / total_obs
#     else:
#         accuracy = 0

#     # Accuracy obtained with naive strategy of predicting all records are benign = 99.96%, better than obtained with C.S./LIME.
#     # This demonstrates why accuracy is not a good metric data set with unbalanced classes.
#     # accuracy_naive_strategy = (0 + (75000-30))/(0 + (75000-30) + 0 + 30)
#     # naive_accuracy = max((true_negative + false_positive)/ total_obs, (true_positive  )/ total_obs)
#     y_true_sum = true_positive + false_negative
#     y_true_len = cm.sum()  # expect cm.sum() == y_true_nparray.size
#     majority_class_acc_benchmark = max((y_true_sum / y_true_len), ((y_true_len - y_true_sum) / y_true_len))

#     accuracy_frac = str(Fraction(accuracy).limit_denominator())
#     # https://www.educative.io/answers/what-is-the-fractionslimitdenominator-method-in-python
#     print(f'{"Accuracy:":<43}{accuracy * 100:8.2f}% {accuracy_frac:>6}')

#     bench_frac = str(Fraction(majority_class_acc_benchmark).limit_denominator())
#     print(f'{"Majority class benchmark for accuracy:":<43}{majority_class_acc_benchmark * 100:8.2f}% {bench_frac:>6}')

#     # If accuracy % is less than benchmark %, model is less accurate than naively predicting the majority class value.
#     acc_minus_benchmark = accuracy - majority_class_acc_benchmark
#     acc_minus_benchmark_frac = str(Fraction(acc_minus_benchmark).limit_denominator())
#     print(f'{"Accuracy minus majority class benchmark:":<43}{acc_minus_benchmark * 100:8.2f}% {acc_minus_benchmark_frac:>6}')

#     if recall < 0:
#         print(f'{"Recall:":<43}{"Undefined":8} {"-":>6}')
#     else:
#         recall_frac = str(Fraction(recall).limit_denominator())
#         print(f'{"Recall:":<43}{recall*100:8.2f}% {recall_frac:>6}')

#     if precision < 0:
#         print(f'{"Precision:":<43}{"Undefined":8} {"-":>6}')
#     else:
#         precision_frac = str(Fraction(precision).limit_denominator())
#         print(f'{"Precision:":<43}{precision * 100:8.2f}% {precision_frac:>6}')

#     # F1 score equals 2 * ( precision * recall )/( precision + recall )
#     # F1 score is a machine learning evaluation metric that measures a model's accuracy. F1  combines the precision and recall scores of a model.
#     if (precision + recall) > 0:
#         f1 = 2*(precision * recall)/(precision + recall)
#     else:
#         f1 = 0

#     f1_frac = str(Fraction(f1).limit_denominator())
#     print(f'{"F1:":<43}{f1 * 100:8.2f}% {f1_frac:>6}')

#     print("C.M.   Pred 0 |      Pred 1|   totals")
#     print(f"{'T0 tn:':<7}{true_negative:>6} | fp: {false_positive:>6} |   {false_positive + true_negative:>6}")
#     print(f'T1 fn: {false_negative:>6} | tp: {true_positive:>6} |   {true_positive + false_negative:>6}')
#     print(f'total: {true_negative+false_negative:>6} |     {false_positive+true_positive:>6} |   {false_negative+false_positive+true_positive+true_negative:>6}')


def test_make_cm_from_random_input(array_len=20):
    # return a confusion matrix (cm) from two numpy arrays; the arrays contain random ones or zeros
    y_true = make_random_test_array(array_len)
    y_pred = make_random_test_array(array_len)
    cm = make_cm(y_true, y_pred)
    make_cm_assertions(cm, array_len)
    return True


if __name__ == "__main__":

    array_length = np.random.randint(3, 31)
    print('>>>> running: test_make_cm_from_random_input()')
    test_make_cm_from_random_input(array_length)

    print('>>>> running: test_make_cm_from_cm_edge_case_zeros()')
    test_make_cm_from_edge_case_zeros(array_len=np.random.randint(3,31))
    print('>>>> running: test_make_cm_from_cm_edge_case_ones()')
    test_make_cm_from_edge_case_ones(array_len=np.random.randint(3,31))
