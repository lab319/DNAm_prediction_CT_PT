# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:04:28 2021

@author: Lab319
"""

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
#from numpy import *

X=pd.read_csv("surrogate_tissue.txt",sep='\t',index_col=0).T#.iloc[0:500,]
y=pd.read_csv("target_tissue.txt",sep='\t',index_col=0).T#.iloc[0:500,]

Out_pre=pd.DataFrame(index=list(y.index),columns=list(y.columns))

leng=len(list(y.columns))

start_time=datetime.datetime.now()

print("Started at: "+str(start_time))
j=500
for i in range(0,leng):
   if i==j:
      print("The cpg number is",i)
      j=i+300
      
   y1=y.iloc[:,i].values.ravel()
   
   x1=X.iloc[:,i].values.reshape(-1,1)
   
   kf = KFold(n_splits=10, shuffle=False)
   
   Out=pd.DataFrame()
  
   for train_index, test_index in kf.split(x1,y1):
      
#       print( 'test_index', test_index)
       train_X, train_y = x1[train_index], y1[train_index]
       
       test_X, test_y = x1[test_index], y1[test_index]
       
       svr=SVR(kernel ='rbf',C=10,gamma=1)
       
       svr.fit(train_X,train_y)
       
       a=svr.predict(test_X)
       
       Out=Out.append(pd.DataFrame(a),ignore_index=True) 
       
   Out_pre[list(y.columns)[i]]=Out.values 
   
end_time=datetime.datetime.now()

print("Ended at: "+str(end_time))

Out_pre.T.to_csv('SVM_target.txt',sep='\t')

