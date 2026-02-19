'''
Docstring for 13_logisitic_regression

لاس در اینجا با کراس انتروپی محاسبه میشود.







'''
from sklearn.linear_model  import LogisticRegression
from sklearn.datasets import make_classification 


from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,confusion_matrix     
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

X,Y = make_classification(500 , n_features=30 , n_informative=10 , n_redundant=5,n_classes=4)
x_train , x_test , y_train,y_test = train_test_split(X,Y , test_size=.2)

log_r = LogisticRegression()
log_r.fit(x_train , y_train)


acc = accuracy_score(y_test , log_r.predict(x_test))
print(acc)

cm = confusion_matrix(y_test , log_r.predict(x_test))
sns.heatmap(cm , annot=True , cmap='Blues')
plt.show()



