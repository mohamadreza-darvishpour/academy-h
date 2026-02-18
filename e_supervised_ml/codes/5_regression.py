'''
Docstring for 5_regression
رگرسیون . معادله ی نرمال
 we want to approximate or predict an amount.

teta = (x^T . X)^-1 . Xy     #معادله ی نورمال


در گام اول ایجاد مجموعه داده ی مصنوعی

'''


import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns        # use to alter chart 

#sue sikitlearn to make artificial dataset
#data set        http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression


from sklearn.datasets import make_regression 
from sklearn.model_selection import train_test_split



X,Y = make_regression(n_samples=400 , n_features=1 , n_informative=1 , noise=3.75 )
x_train , x_test , y_train ,y_test = train_test_split(X , Y , test_size=0.2)
#we use noise 0 to see the changes with changing in parameters.

#draw the chart 
plt.scatter(X,Y)
plt.title('made dataset')
plt.show()

plt.scatter(x_train,y_train)
plt.title('train dataset')
plt.show()

plt.scatter(x_test , y_test)
plt.title('test dataset')
plt.show()


