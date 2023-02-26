# checking/demonstrating the problematic behavior of the sklearn confusion_matrix function
from sklearn.metrics import confusion_matrix
import numpy as np
from fractions import Fraction

cm1 = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0])
print(cm1, cm1.size)

cm2 = confusion_matrix([1, 0, 0, 0], [0, 0, 0, 0])
assert(cm2.size == 4)

cm3 = confusion_matrix([0, 0, 0, 0], [0, 0, 0, 0])
print(cm3, cm3.size)
assert(cm3.size == 4)  # expect error
# when the comfusion matrix size is 1, we get an error when calling revel()
# tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()

cm4 = confusion_matrix([1, 1, 1, 1], [1, 1, 1, 1])
print(cm4, cm4.size)
assert(cm4.size == 4)  # expect error
# when the comfusion matrix size is 1, we get an error when calling revel()
# tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
