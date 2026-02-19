

'''
Docstring for 

p(a|b) = p(b|a)*p(a) /  p(b)    
p(a|b) > posterior
p(b|a) > likelihood 
p(a)   > prior 
p(b)   > marginalization


http://scikit-learn.org/stable/modules/naive_bayes.html
'''

from sklearn.datasets import load_iris
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB , MultinomialNB
from sklearn.metrics import accuracy_score



X,Y = load_iris(return_X_y=True)


x_train , x_test , y_train , y_test = train_test_split(X,Y , test_size=0.2)
print(x_train.shape , x_test.shape)

gnb = GaussianNB()
gnb.fit(x_train,y_train)
acc = accuracy_score(y_test , gnb.predict(x_test))
print(acc)


mnb = MultinomialNB()
mnb.fit(x_train , y_train)
acc_mnb = accuracy_score(y_test , mnb.predict(x_test))
print(acc_mnb)









