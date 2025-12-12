import numpy as np 
import pandas as pd 
import math 
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
'''   didn't find out ...   '''
#-------------------------------------------------------------
# خواندن داده
raw_data = pd.read_csv('./dataset_credit_approval/crx.data',
                       header=None, 
                       na_values='?')

#-------------------------------------------------------------
# ستون‌های پیوسته و گسسته
cont_attr = [1,2,7,10,13,14]                # continuous
disc_attr = [i for i in range(16) if i not in cont_attr]   # discrete (0-15 ولی شامل 15 هم هست)

#-------------------------------------------------------------
# جدا کردن ستون هدف قبل از هر پردازشی
Y = raw_data[15].copy()

# حذف ستون هدف از دیتافریم ویژگی‌ها
raw_data = raw_data.drop(columns=[15])

# حالا ستون‌های گسسته را به‌روز می‌کنیم (بدون 15)
disc_attr = [i for i in disc_attr if i != 15]

#-------------------------------------------------------------
# پر کردن مقادیر گمشده
for ind in range(15):   # دیگر ستون 15 نداریم
    if ind in cont_attr:
        raw_data[ind] = raw_data[ind].fillna(raw_data[ind].mean())
    else:
        raw_data[ind] = raw_data[ind].fillna(raw_data[ind].mode()[0])

#-------------------------------------------------------------
# One-hot encoding فقط روی ستون‌های گسسته
one_hot = pd.get_dummies(raw_data[disc_attr])

# حذف ستون‌های گسسته و اتصال one-hot
raw_data = raw_data.drop(columns=disc_attr)
X = raw_data.join(one_hot)

#-------------------------------------------------------------
# تبدیل به numpy
X = np.array(X)
Y = np.array(Y)

print("Shapes:", X.shape, Y.shape)

#-------------------------------------------------------------
# تقسیم داده
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("Train:", x_train.shape, y_train.shape)
print("Test :", x_test.shape , y_test.shape)
