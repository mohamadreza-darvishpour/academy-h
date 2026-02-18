'''
Docstring for 8_knn
target is classifying

یک روش دسته بندی دااده بر اساس فاصله تا جند همسایه بود.


'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#read the csv data in panda
#header=None means in first row we has not name of features.
#na_values means the sign of missing data here. 
raw_data = pd.read_csv('./dataset_credit_approval/crx.data' , header=None , na_values='?')

#concrete attr and discrete attrs
cont_attr = [1,2,7,10,13,14]
dis_attr = []
for i in range(15):
    if not i in cont_attr:
        dis_attr.append(i)

print(raw_data)


import math
print("before" , raw_data.isna().sum())
for index in range(15):
    if index in cont_attr:
        _mean = raw_data[index].mean()
        raw_data[index]  = raw_data[index].replace(math.nan , _mean)
        
    else:
        _mode = raw_data[index].mode()[0]
        raw_data[index] = raw_data[index].replace(math.nan , _mode)
        
print("after" , raw_data.isna().sum())








#normalization 
from sklearn.preprocessing import MinMaxScaler 

min_max_scaler = MinMaxScaler()
x = raw_data[cont_attr].values
x_scaled = min_max_scaler.fit_transform(x)

normalized = pd.DataFrame(x_scaled , columns=cont_attr , index=raw_data.index)
raw_data[cont_attr] = normalized 
print(raw_data)



#  روش اوردینال انکودینگ برای عددی کردن مقادیر حرفی
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder(dtype=np.int32)
raw_data[15] = ord_enc.fit_transform(raw_data[[15]])
print(raw_data)

one_hot = pd.get_dummies(raw_data[dis_attr])
raw_data = raw_data.drop(dis_attr , axis=1)
X = raw_data.join(one_hot)




Y = np.array(X.pop(15))
X = np.array(X) 

print(X.shape , Y.shape)
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,Y, test_size=0.2)
print(x_test.shape ,x_train.shape)


#************************
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train , y_train)


score = knn.score(x_test , y_test)
print('\\nn\n\n\n\n\nscore :     ' , score)


