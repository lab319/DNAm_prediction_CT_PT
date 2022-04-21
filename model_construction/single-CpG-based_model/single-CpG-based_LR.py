

import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import pandas as pd 
from sklearn.model_selection import KFold,LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.linear_model import LinearRegression
#from numpy import *

X=pd.read_csv('surrogate_tissue.txt','\t',index_col=0).T
y=pd.read_csv('target_tissue.txt','\t',index_col=0).T

Out_pre=pd.DataFrame(index=list(y.index),columns=list(y.columns))

leng=len(list(y.columns))

start_time=datetime.datetime.now()

print("Started at: "+str(start_time))

for i in range(0,leng):

   y1=y.iloc[:,i].values.ravel()
   
   x1=X.iloc[:,i].values.reshape(-1,1)
   
   kf = KFold(n_splits=10, shuffle=False)
   
   Out=pd.DataFrame()
   for train_index, test_index in kf.split(x1,y1):
#       print( 'test_index', test_index)
       train_X, train_y = x1[train_index], y1[train_index]
       
       test_X, test_y = x1[test_index], y1[test_index]
       
       LR=LinearRegression()
       
       LR.fit(train_X,train_y)
      
       a=LR.predict(test_X)
       
       Out=Out.append(pd.DataFrame(a),ignore_index=True) 
       
   Out_pre[list(y.columns)[i]]=Out.values 
   
end_time=datetime.datetime.now()

print("Ended at: "+str(end_time))

Out_pre.T.to_csv("LR-single_target.txt",sep='\t')    

