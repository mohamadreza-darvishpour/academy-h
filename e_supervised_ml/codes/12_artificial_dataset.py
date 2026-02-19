

'''
Docstring for 

p(a|b) = p(b|a)*p(a) /  p(b)    
p(a|b) > posterior
p(b|a) > likelihood 
p(a)   > prior 
p(b)   > marginalization


http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html




'''

from sklearn.datasets import load_iris
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB , MultinomialNB
from sklearn.metrics import accuracy_score , roc_curve , confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

X,Y = make_classification(  1000 ,n_features=20 ,  n_informative=15 , n_classes=5  )
min_max_scaler = MinMaxScaler() 
X = min_max_scaler.fit_transform(X)

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


cm = confusion_matrix(y_test , gnb.predict(x_test))
sns.heatmap(cm , cmap='Reds',annot=True)
plt.show()

'''
fpr_g,tpr_g, _ = roc_curve(y_test , gnb.predict(x_test))
fpr_m,tpr_m, _ = roc_curve(y_test , mnb.predict(x_test))


plt.plot(fpr_g , tpr_g , label='gnb')
plt.plot(fpr_m , tpr_m , label='mnb')

plt.show()

'''