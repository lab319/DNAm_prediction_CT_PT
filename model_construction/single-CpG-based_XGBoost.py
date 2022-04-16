# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:18:45 2020

@author: Administrator
"""
import xgboost as xgb
import datetime
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import KFold,GridSearchCV,LeaveOneOut,train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

X=pd.read_csv("surrogate_tissue.txt",sep='\t',index_col=0).T
y=pd.read_csv("target_tissue.txt",sep='\t',index_col=0).T



Out_pre=pd.DataFrame(index=list(y.index),columns=list(y.columns))
leng=len(list(y.columns))

start_time=datetime.datetime.now()

print("Started at: "+str(start_time))
j=3000
loo = LeaveOneOut()
for i in range(0,leng):
   if i==j:
       print("The cpg number is",i)
       j=i+2000
   y1=y.iloc[:,i].values.ravel()
   x1=X.iloc[:,i].values.reshape(-1,1)
   kf = KFold(n_splits=10, shuffle=False)
   Out=pd.DataFrame()
   for train_index, test_index in kf.split(x1,y1):
     # print('train_index', train_index, 'test_index', test_index)
       train_X, train_y = x1[train_index], y1[train_index]
       test_X, test_y = x1[test_index], y1[test_index]
       
       params={
               'eta':0.13,
                 'objective':'reg:squarederror',
                 'max_depth':5,
                 #'subsample':0.85,
                 'booster':'gbtree',
                 #'n_estimators':200,
                 'min_child_weight':5,
                 #'colsample_bytree':0.065,
                 'gamma':1e-3,
                 #'reg_lambda':1,
                 'reg_alpha':0.5,
                 'eval_metric':'mae'
                  }
       
       dtrain = xgb.DMatrix(train_X, train_y)
       dtest = xgb.DMatrix(test_X,test_y)
       #dval=xgb.DMatrix(X_val,y_val)
       watchlist = [(dtrain, 'train')]
       num_boost_round = 100
       model=xgb.train(params,dtrain,num_boost_round=200,early_stopping_rounds=30,evals=watchlist,verbose_eval = 0)
       #print("best_n_estimators=",model.best_iteration)
       a=model.predict(dtest)      
       Out=Out.append(pd.DataFrame(a),ignore_index=True)   
      
   Out_pre[list(y.columns)[i]]=Out.values 
   
end_time=datetime.datetime.now()

print("Ended at: "+str(end_time))

Out_pre.T.to_csv('XGBoost_target.txt',sep='\t')
