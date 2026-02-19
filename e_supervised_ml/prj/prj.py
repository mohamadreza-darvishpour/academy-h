
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split


path = "prj/project/Dataset/CSM_dataset.xlsx"
dataset = pd.read_excel(path)

print(dataset.head(2))

#حذف سطر های خالی
dset = dataset.dropna(how='all')
dset.columns = range(dset.shape[1]) # تغییر نام ستون‌ها به اعداد   

#delete name col
dset = dset.drop(columns=[0])

Y = dset.pop(4)
X = dset       
 

minmaxscaler = MinMaxScaler()
X = minmaxscaler.fit_transform(X) 
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.2)

print(x_train.shape , x_test.shape)     





print(dset.head(2))    # 231*14





