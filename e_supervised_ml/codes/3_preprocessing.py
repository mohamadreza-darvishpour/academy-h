'''
تشخیص داده ها پرت چند جلسه ی دیگه اس.
Docstring for 2_preprocessing

alter discrete label to numeric ones .
example : 
a:2,b:3,c:4,a:7    >>>>    0:2,1:3,:2:4,0:7


we can use one hot encoded data **** its important to learn.
*******      ONE HOT ENCODED DATA    ********** 
example : 
a:2,b:3,c:4,a:7 
to below 
a_ones  ,b_ones  ,c_ones   , points 
  1     ,   0     ,    0     , 2
  0     ,   1     ,    0     , 3
  0     ,   0     ,    1     , 4
  1     ,   0     ,    0     , 7


وقتی از این روش استفاده کنیم و داده هارا به اعداد نگاشت کنیم در واقع برای آنها معمولا ترتیب
در نظر گرفته ایم . 
در حالی که برای تیم ها این کار مناسب نیست.


در روش 
one hot encoded data  
هم تعداد زیادی ویژگی اضاف میشود که مطلوب ما نیست.
اما در حالتی که داده چند متغیر دارد میتواند مفید باشد.


معمولا label encoding 
در جایی استفاده میشود که مدرک ترتیب هم دارد. مثل مدرک کارشناسی. ارشد.دکترا

'''


import numpy as np 
import pandas as pd 
import math 
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
'''
ordinal:ترتیبی
کار 
labelencoder , ordinalencoder 
تقریبا شبیه همدیگه است و هر دو به آرایه ی عددی نگاشت میکنن
اما احتمالا با تفاوت هایی.



'''






#read the csv data in panda
#header=None means in first row we has not name of features.
#na_values means the sign of missing data here. 
raw_data = pd.read_csv('./dataset_credit_approval/crx.data' , header=None , na_values='?')

#handling missing values              **************************
#continuous indexes *** its very noticable
cont_attr = [1,2,7,10,13,14]
#discrete attribute index
disc_attr = [i for i in range(16) if i not in cont_attr]


#we use mode for discrete one and mean for continuous one 
for ind in range(16):
    # print("______________________\n\n\n\n____________",ind,"\n__________________________\nn\n\nn\n")
    if ind in cont_attr:
        _mean = raw_data[ind].mean()
        raw_data[ind] = raw_data[ind].replace(math.nan , _mean)
    
    else:
        _mode = raw_data[ind].mode()[0]
        raw_data[ind]  = raw_data[ind].replace(math.nan , _mode)


# print(raw_data)

'''
#use ordianal encoder 
ord_enc = OrdinalEncoder(dtype=np.int32)
#its important to get to fit transform 2-dim array.
raw_data[15] = ord_enc.fit_transform(raw_data[[15]])
'''


#first way
'''
one_enc = OneHotEncoder(handle_unknown="ignore" , sparse_output=False )
print('\n\n\n____________________\n\n\n')
new_raw_data = one_enc.fit_transform(raw_data[disc_attr])
new_raw_data_df = pd.DataFrame(new_raw_data , index=raw_data.index)

# new_raw_data_df.index = raw_data.index 
raw_data = raw_data.drop(disc_attr , axis=1)

# X = pd.concat([new_raw_data , raw_data ]   , axis=1)     wrong
X = pd.concat([new_raw_data_df , raw_data ]   , axis=1)
print(X)
'''

# Y=raw_data[15].copy()

# second way 
one_hot = pd.get_dummies(raw_data[disc_attr])
raw_data = raw_data.drop(disc_attr , axis=1)
X= raw_data.join(one_hot) 
# print(X.isna().sum())




# print(raw_data)

'''
has some problem here ****************

# divide to test and train
Y = np.array(X.pop(15))
X = np.array(X) 
print(X.shape , Y.shape) 
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.2)




'''





