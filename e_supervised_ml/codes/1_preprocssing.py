'''

all the works on datas that makes it ready to use by model.

dataset from here: 
https://archive.ics.uci.edu/dataset/27/credit+approval

missed data : datas that one or more feature not exists.

3 way to correct missed datas : 1.delete   2.استفاده از ویژگی های آماری       
         3.use predictor model 
         4.use different type of data
         5.we should do that and have no  other way


'''


'''
missing values , missing datas 

some data may missed in gathering,saving,moving

way to correct: 
1.delete sample
2.use statistic features like mean,max,min,....
3.use predictor model to approximate missed data


data preprocessing:
> data cleaning : missing data , noisy data
> data transformation : normalization , attribute selection , discretization , concept hierarchy generation
> data reduction  : data cube aggregation , attribute subset selection , numerosity reduction , dimensionality reduction

missing data may show with '?' or 'nan' or "none" .... in datas.
'''


#libs 
import numpy as np
import pandas as pd 
import math 

#read the csv data in panda
#header=None means in first row we has not name of features.
#na_values means the sign of missing data here. 
raw_data = pd.read_csv('./dataset_credit_approval/crx.data' , header=None , na_values='?')

#handling missing values              **************************
#continuous indexes *** its very noticable
cont_attr = [1,2,7,10,13,14]
#discrete attribute index
disc_attr = [i for i in range(16) if i not in cont_attr]

print(raw_data)

#we use mode for discrete one and mean for continuous one 
for ind in range(16):
    # print("______________________\n\n\n\n____________",ind,"\n__________________________\nn\n\nn\n")
    if ind in cont_attr:
        _mean = raw_data[ind].mean()
        raw_data[ind] = raw_data[ind].replace(math.nan , _mean)
    
    else:
        _mode = raw_data[ind].mode()[0]
        raw_data[ind]  = raw_data[ind].replace(math.nan , _mode)





# normalization              ********************************
'''
در نرمال سازی مقیاس ها خیلی مهم هستند.
معنی روتین نرمال سازی این است که رنج داده ها هر چی که هست به رنج 0 تا 1 برسونیم.
از رنج های -1 تا 1 و رنج های دیگر هم استفاده میشود.



mini max way of normalization 
x(scaled) = ( x(data) - x(min) ) / ( x(max) - x(min) )

normalization helps to scaling and correlation 


'''






#return the matris where missing data show 'true' else 'false'
print(raw_data.isna().sum())












