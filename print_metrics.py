# author: John Kraus
# Print all the metrics, not just the cherry-picked, favorable-looking metric(s).
import numpy as np
from fractions import Fraction
import confusion_matrix
# reference regarding formatting:  https://stackoverflow.com/questions/8234445/format-output-string-right-alignment

TODO: implement additional metrics; from tabnet, consider adding: A few classic evaluation metrics are implemented (see further below for custom ones): binary classification metrics : 'auc', 'accuracy', 'balanced_accuracy', 'logloss'



def print_metrics_from_cm(cm):
    # cm eq [[tn , fp],
    #       [fn , tp]], a numpy array
    print('cm:', cm) 
    true_negative, false_positive, false_negative, true_positive = cm.ravel()
    
    # Recall metric: What proportion of actual positives was identified correctly?
    if (true_positive + false_negative) > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = -1  # or undefined?

    if not ((true_positive is None) or (false_positive is None)):         
        # Precision: TP/(TP+FP)
        if (true_positive + false_positive) > 0:
            precision =  true_positive / (true_positive + false_positive)
        else:
            precision = -1  # undefined?

    try:
        total_obs = (true_positive + true_negative + false_positive + false_negative)
    except ValueError as e:
        print('error 77', e)

    # Accuracy: (TP+TN)/(TP+TN+FP+FN) = 97.96%      
    # Informally, accuracy is the fraction of predictions our model got right. 
    if total_obs > 0: # (true_positive + true_negative + false_positive + false_negative) > 0:
        accuracy = (true_positive + true_negative) / total_obs
    else:
        accuracy = 0

    # Accuracy obtained with naive strategy of predicting all records are benign = 99.96%, better than obtained with C.S./LIME.
    # This demonstrates why accuracy is not a good metric data set with unbalanced classes.
    # accuracy_naive_strategy = (0 + (75000-30))/(0 + (75000-30) + 0 + 30)
    # naive_accuracy = max((true_negative + false_positive)/ total_obs, (true_positive  )/ total_obs)
    y_true_sum = true_positive + false_negative
    y_true_len = cm.sum()  # expect cm.sum() == y_true_nparray.size
    majority_class_acc_benchmark = max((y_true_sum / y_true_len), ((y_true_len - y_true_sum) / y_true_len))

    accuracy_frac = str(Fraction(accuracy).limit_denominator())
    # https://www.educative.io/answers/what-is-the-fractionslimitdenominator-method-in-python
    print(f'{"Accuracy:":<43}{accuracy * 100:8.2f}% {accuracy_frac:>6}')

    bench_frac = str(Fraction(majority_class_acc_benchmark).limit_denominator())
    print(f'{"Majority class benchmark for accuracy:":<43}{majority_class_acc_benchmark * 100:8.2f}% {bench_frac:>6}') 
    
    acc_minus_benchmark = accuracy - majority_class_acc_benchmark
    acc_minus_benchmark_frac = str(Fraction(acc_minus_benchmark).limit_denominator())
    print(f'{"Accuracy minus majority class benchmark:":<43}{acc_minus_benchmark * 100:8.2f}% {acc_minus_benchmark_frac:>6}')

    if recall < 0:
        print(f'{"Recall:":<43}{"Undefined":8} {"-":>6}')
    else:    
        recall_frac = str(Fraction(recall).limit_denominator())
        print(f'{"Recall:":<43}{recall*100:8.2f}% {recall_frac:>6}')

    if precision < 0:
        print(f'{"Precision:":<43}{"Undefined":8} {"-":>6}')
        #print(f'{"Precision:":<43}{precision * 100:8.2f}% {precision_frac:>6}')
    else:
        precision_frac = str(Fraction(precision).limit_denominator())        
        print(f'{"Precision:":<43}{precision * 100:8.2f}% {precision_frac:>6}')

    # F1 equals 2 * ( precision * recall )/( precision + recall )
    if (precision + recall) > 0:
        f1 = 2*(precision * recall)/(precision + recall)
    else:
        f1 = 0

    f1_frac = str(Fraction(f1).limit_denominator())
    print(f'{"F1:":<43}{f1 * 100:8.2f}% {f1_frac:>6}')
        
    print("C.M.   Pred 0 |      Pred 1|   totals")

    print(f"{'T0 tn:':<7}{true_negative:>6} | fp: {false_positive:>6} |   {false_positive + true_negative:>6}")
    print(f'T1 fn: {false_negative:>6} | tp: {true_positive:>6} |   {true_positive + false_negative:>6}')
    print(f'total: {true_negative+false_negative:>6} |     {false_positive+true_positive:>6} |   {false_negative+false_positive+true_positive+true_negative:>6}')


def make_random_test_array(array_len=20, num_classes=2):
    # return an numpy array filled randomly with with ones and zeros
    rand_generator = np.random.default_rng()
    np_array = rand_generator.integers(0, num_classes, array_len)
    return np_array


def test_print_metrics_from_random_data():
    print('running test_print_metrics_cm()')
    array_len = 20
    num_classes = 2
    rand_generator = np.random.default_rng()
    y_true = rand_generator.integers(0, num_classes, array_len)
    y_pred = rand_generator.integers(0, num_classes, array_len)
    print_array_data(y_true, y_pred)
    cm = confusion_matrix.get_binary_confusion_matrix(y_true, y_pred)
    print_metrics_from_cm(cm)


def test_print_metrics_from_edge_case_zeros(array_len=20):
    # test value = 1 or 0; all values in the y_true and y_false will equal the test_val
    y_true = np.zeros(array_len, dtype=int)
    y_pred = np.zeros(array_len, dtype=int)
    cm = confusion_matrix.get_binary_confusion_matrix(y_true, y_pred)
    make_cm_assertions(cm, array_len)
    print_metrics_from_cm(cm)   


def test_print_metrics_from_cm_edge_case_ones(array_len=20):
    y_true = np.ones(array_len, dtype=int)
    y_pred = np.ones(array_len, dtype=int)
    cm = confusion_matrix.get_binary_confusion_matrix(y_true, y_pred)
    make_cm_assertions(cm, array_len)
    print_metrics_from_cm(cm)      


def make_cm_assertions(cm, array_len):
    assert(cm.sum() == array_len)
    assert(cm.size == 4)


def test_print_metrics_from_arbitrary_input():
    y_true = np.array(list([0,0,0,0,1,0]))
    y_pred = np.array(list([1,0,0,0,0,0]))
    print_array_data(y_true, y_pred)
    cm = confusion_matrix.get_binary_confusion_matrix(y_true, y_pred)
    make_cm_assertions(cm, y_true.size)
    print_metrics_from_cm(cm) 

def test_print_metrics_from_arbitrary_input(y_true, y_pred):
    # y_true = np.array(list([0,0,0,0,1,0]))
    # y_pred = np.array(list([1,0,0,0,0,0]))
    print_array_data(y_true, y_pred)
    cm = confusion_matrix.get_binary_confusion_matrix(y_true, y_pred)
    make_cm_assertions(cm, y_true.size)
    print_metrics_from_cm(cm) 
               

def get_correct(y_true, y_pred):
    # how many predicted values are correct?
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
    cm = confusion_matrix.get_binary_confusion_matrix(y_true, y_pred)
    make_cm_assertions(cm, array_len)
    return cm


if __name__ == "__main__":
    # run tests
    print('>>>> running: test_make_cm_from_random_input()')
    cm = test_make_cm_from_random_input(20)
    print_metrics_from_cm(cm)
    print('>>>> running: test_print_metrics_from_cm_edge_case_zeros(array_len=20)')
    test_print_metrics_from_edge_case_zeros(array_len=20)
    print('>>>> running: test_print_metrics_from_cm_edge_case_ones(array_len=20)')
    test_print_metrics_from_cm_edge_case_ones(array_len=20)

    # All false positives scenario
    y_true = np.array(list([0,0,0,0,0,0]))
    y_pred = np.array(list([1,1,1,1,1,1]))
    test_print_metrics_from_arbitrary_input(y_true, y_pred)

    # All false negatives scenario
    y_pred = np.array(list([0,0,0,0,0,0]))
    y_true = np.array(list([1,1,1,1,1,1]))
    test_print_metrics_from_arbitrary_input(y_true, y_pred)



# compute and log classification model result metrics; work in process; TODO: evaluate
from datetime import datetime

def log_metrics_from_cm2(cm, random_seed_int, strat_rand_sample_duration_sum, clf_type="?", seed="?"):
    # precision = TP / (TP + FP)  What proportion of positive identifications was actually correct?
    tp = cm[1,1]
    fp = cm[0,1]
    precision = 0
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    # recall = TP / (TP + FN)  What proportion of actual positives was identified correctly?
    fn = cm[1,0]
    recall = 0
    if (tp + fn) > 0:    
        recall = tp / (tp + fn)

    tn = cm[0,0]
    count = tp + fp + fn + tn  # total number of predictions
    f1 = 0    
    if (precision + recall) > 0:
        f1 = 2 * ((precision * recall)/(precision + recall))
    else:
        f1 = 0

    # Informally, accuracy is the fraction of predictions our model got right. Formally, accuracy has the following definition:
    accuracy = (tp + tn) / count
    true_benign = tn + fp
    naive_accuracy = true_benign / count  # accuracy achieved if we always predicted "BENIGN".
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d-%H:%M:%S")
    metrics = str(date_time) + ", clf_type: "+ clf_type + ", seed: " + str(seed) + ", f1: " + str(f1) + ", precision: " + str(precision) + ", recall = " + str(recall) + ", accuracy = " + str(accuracy) + ", naive_accuracy = " + str(naive_accuracy) + ", cm = " + str(cm.tolist()) + "\n"
    # one-time manual initialization chore: manually create a file named "result.csv" and enter the following string on the first line: "datetime,clf_type,seed,f1,precision,recall,accuracy,naive_accuracy,cm,strat_sample_seed,sum_duration"
    metrics_csv = str(date_time) + ","+ clf_type + "," + str(seed) + "," + str(f1) + "," + str(precision) + "," + str(recall) + "," + str(accuracy) + "," + str(naive_accuracy) + ",\"" + str(cm.tolist()) + "," + str(random_seed_int) + "," + str(strat_rand_sample_duration_sum[0])  + "\"\n"

    with open("results.log", "a") as file1:
        file1.writelines(metrics)

    with open("results.csv", "a") as file1:
        print("metrics_csv = ", metrics_csv)
        file1.writelines(metrics_csv)

    return metrics, metrics_csv

