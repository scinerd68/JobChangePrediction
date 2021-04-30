import numpy as np

# X_train = np.genfromtxt("Data/X_train_under.csv", delimiter=',')
# y_train = np.genfromtxt("Data/y_train_under.csv", delimiter=',')

X_train = np.genfromtxt("Data/X_train.csv", delimiter=',')
y_train = np.genfromtxt("Data/y_train.csv", delimiter=',')

X_cv = np.genfromtxt("Data/X_cv.csv", delimiter=',')
y_cv = np.genfromtxt("Data/y_cv.csv", delimiter=',')


# X_train = np.genfromtxt("Data/X_train_strat.csv", delimiter=',')
# y_train = np.genfromtxt("Data/y_train_strat.csv", delimiter=',')

# X_cv = np.genfromtxt("Data/X_cv_strat.csv", delimiter=',')
# y_cv = np.genfromtxt("Data/y_cv_strat.csv", delimiter=',')

from LogisticRegression.LogReg import *

#Add one extra column for the bias.
X_train = add_bias(pd.DataFrame(X_train))
X_cv = add_bias(pd.DataFrame(X_cv))

#Model
J, theta = fit(X_train, y_train, 0.1, 500, 0.1, theta =None, balanced = True)
# Learning curve
import matplotlib.pyplot as plt
plt.figure(figsize = (6,4))
plt.scatter(range(0, len(J)), J)
plt.show()

#Train set evaluation
h, acc = predict(X_train, y_train, theta, threshold =0.5)
recall, precision, f1 = recall_precision_f1(h, y_train)
print("Train set Accuracy:",acc)
print("Train set Recall: ",recall)
print("Train set Precision:", precision)
print("Train set f1: ", f1)

#CV set evaluation
h, acc = predict(X_cv, y_cv, theta, threshold =0.5)
recall, precision, f1 = recall_precision_f1(h, y_cv)
print("CV set Accuracy:",acc)
print("CV set Recall: ",recall)
print("CV set Precision:", precision)
print("CV set f1: ", f1)

# Train set Accuracy: 0.7723159909518009
# Train set Recall:  0.7561312607944732
# Train set Precision: 0.5339024390243903
# Train set f1:  0.6258756254467477
# CV set Accuracy: 0.755741127348643
# CV set Recall:  0.6956989247311828
# CV set Precision: 0.4976923076923077
# CV set f1:  0.5802690582959641
