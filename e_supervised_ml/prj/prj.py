
import numpy as np 
import pandas as pd 




path = "prj/project/Dataset/CSM_dataset.xlsx"
dataset = pd.read_excel(path)

#حذف سطر های زوج خالی
dset = dataset.dropna(how='all')
dset.columns = range(dset.shape[1]) # تغییر نام ستون‌ها به اعداد   






print(dset)    # 231*14





