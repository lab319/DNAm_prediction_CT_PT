

import xgboost as xgb
import datetime
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import KFold,GridSearchCV,LeaveOneOut
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.feature_selection import SelectKBest,f_regression
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.svm import SVR

def P_values(train_X,train_y,test_X):
    
     F,pvalues_f = f_regression(train_X,train_y)
     k = F.shape[0]-(pvalues_f>0.01).sum()
     #print("Significantly correlated Cpgs: ",k)
     sel_p=SelectKBest(f_regression,k=30)
     sel_p.fit(train_X,train_y)
     #X_sel_f=sel_p.get_support(indices=True)
     #print(X_sel_f)
     X_sel_train=sel_p.transform(train_X)
     X_sel_test=sel_p.transform(test_X)
     return X_sel_train,X_sel_test
 
    
def XGB_find_base(scoring,train_X,train_y,f):
    
       params = {}      
       ##3,3
       param_test1 ={'max_depth': np.arange(5, 20, 3),'min_child_weight':np.arange(3,6,1)}
       gsearch1 = GridSearchCV(f, param_grid = param_test1,scoring=scoring,cv=3)
       gsearch1.fit(train_X,train_y)
       f.max_depth=gsearch1.best_params_['max_depth']
       f.min_child_weight=gsearch1.best_params_['min_child_weight']
       params.update({'max_depth': gsearch1.best_params_["max_depth"]})
       params.update({'min_child_weight': gsearch1.best_params_["min_child_weight"]})
       
       
       return f,params    
 

X=pd.read_csv('surrogate_feature.txt','\t',index_col=0).T

X_y=pd.read_csv('surrogate_tissue.txt','\t',index_col=0).T
y=pd.read_csv('target_tissue.txt','\t',index_col=0).T

Out_pre=pd.DataFrame(index=list(y.index),columns=list(y.columns))

leng=len(list(y.columns))#
start_time=datetime.datetime.now()
print("Started at: "+str(start_time))
loo=LeaveOneOut()
for i in range(0,leng):
   print(i)
   y1=y.iloc[:,i].values.ravel()
   x1=X_y.iloc[:,i].values.reshape(-1,1)
   kf = KFold(n_splits=10, shuffle=False)
   cpg_n=list(y.columns)[i]
   X_find=X.drop([cpg_n],axis=1)
   X_find_ch= X_find.values
   Out=pd.DataFrame()
#   Out_single=pd.DataFrame()
   for train_index, test_index in kf.split(X_find_ch,y1):
       X1_train=x1[train_index]
       X1_test=x1[test_index]
       train_X, train_y = X_find_ch[train_index], y1[train_index]
       test_X, test_y = X_find_ch[test_index], y1[test_index]
       X_sel_train,X_sel_test=P_values(train_X,train_y,test_X)
       X_sel_self=np.column_stack((X1_train,X_sel_train))
       test_X_new=np.column_stack((X1_test,X_sel_test))
       
       
       f=XGBRegressor( objective ='reg:squarederror',
                   booster='gbtree',   #gbtree,gblinear
                   n_estimators=200,   #num_boosting_rounds
                   max_depth=10,
                   eval_metric='mae',
                   #subsample=0.95,
                   colsample_bytree=0.001,
                   gamma=1e-2,
                   reg_alpha=0.01,
                   reg_lambda=2,
                   random_state=123456,
                   eta=0.03
                   )     
       
       f,params = XGB_find_base('neg_mean_absolute_error',X_sel_self, train_y,f)
       
       params.update({'eta':0.03,'gamma':1e-2,'colsample_bytree':0.001,'reg_lambda':2,'reg_alpha':0.01,'random_state':123456})
       
       dtrain = xgb.DMatrix(X_sel_self,train_y)
       dtest = xgb.DMatrix(test_X_new,test_y)
       watchlist = [(dtrain, 'train')]
       num_boost_round = 200
       model=xgb.train(params,dtrain,num_boost_round=200,early_stopping_rounds=20,evals=watchlist,verbose_eval = 0)
       a=model.predict(dtest)   
       #print("best_n_estimators=",model.best_iteration)
       Out=Out.append(pd.DataFrame(a),ignore_index=True)   
       
   Out_pre[cpg_n]=Out.values 

end_time=datetime.datetime.now()

print("Ended at: "+str(end_time))

Out_pre.T.to_csv("XGBoost-multiple_target.txt",sep='\t')    











    

