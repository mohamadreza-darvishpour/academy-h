'''

bagging parallel

boosting  sequential

http://scikit-learn.org/stable/auto_examples/ensemble/index.html

Docstring for 15_bagging_boositing
'''

import pandas as pd 
import numpy as np

from sklearn.svm  import SVC 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier






X = pd.read_csv("./blood/transfusion.data").set_axis(labels=list(range(5)),axis=1)
Y= np.array(X.pop(4))
X = np.array(X)

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

x_train , x_test , y_train , y_test = train_test_split(X ,Y , test_size=.2)
print(x_train.shape , x_test.shape)



clf = RandomForestClassifier()
clf.fit(x_train , y_train) 
score = clf.score(x_test , y_test)
print(score)



clf = AdaBoostClassifier()
clf.fit(x_train , y_train) 
score = clf.score(x_test , y_test)
print(score)


