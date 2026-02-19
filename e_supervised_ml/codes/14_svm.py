'''
استفاده از 
svm 
برای دسته بندی

خط هایی که برای جداسازی مجموعه داده ها پیش بینی میشودند


https://archive.ics.uci.edu/ml/datasets/Blood_Transfusion_Service_Center


'''
import pandas as pd 
import numpy as np

from sklearn.svm  import SVC 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

X = pd.read_csv("./blood/transfusion.data").set_axis(labels=list(range(5)),axis=1)
Y= np.array(X.pop(4))
X = np.array(X)

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

x_train , x_test , y_train , y_test = train_test_split(X ,Y , test_size=.2)
print(x_train.shape , x_test.shape)

# svc_linear = SVC(kernel='linear'    )
svc_linear = SVC(kernel='poly' ,degree=3   )
svc_linear = SVC(kernel='rbf',C=.1 , gamma=100  )

svc_linear.fit(x_train , y_train)


print(svc_linear.score(x_test , y_test))






