# -*-coding: utf-8-*-
import sys
sys.path.append('/data/tinyv/kw/ppackage')
import pandas as pd
from lightgbm.sklearn import *
from sklearn import metrics


import traceback
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import *
import json
import matplotlib.pyplot as plt
import seaborn as sns
from project_demo.tools.evaluate import *
from copy import deepcopy
from sklearn.metrics import mean_squared_error
import numpy as np


def main():
    train_org = pd.read_csv('train_bu_weather.csv')
    train_org['meter_reading'] = train_org['meter_reading'].apply(lambda x: np.log1p(x))
    test_org = pd.read_csv('test_bu_weather.csv')


    train, test = train_test_split(train_org, test_size=0.33, random_state=42)

    features = [f for f in train if f not in ['building_id','site_id','meter_reading']]
    train_x = train[features]
    train_y = train['meter_reading']

    test_x = test[features]
    test_y = test['meter_reading']

    clf = LGBMRegressor(boosting_type='gbdt',
            colsample_bytree=0.3, drop_rate=0.1,
            importance_type='split',
            learning_rate=0.05,
            #max_bin=1000,
            max_depth=4,
            min_child_samples=1000,
            min_split_gain=0.2,
            n_estimators=500, n_jobs=-1,
            num_leaves=8, objective=None,
            random_state=24,
            reg_alpha=10, reg_lambda=1,
            sigmoid=0.3, silent=True,
            #class_weight={0:1,1:10},
            #subsample=0.5,
            subsample_for_bin=30000,
            is_unbalance=True,
            subsample_freq=1
            )
    clf.fit(train_x, train_y)
    train_y_pred = clf.predict(train_x)
    y_pred = clf.predict(test_x)
    print(mean_squared_error(train_y_pred,train_y), mean_squared_error(y_pred,test_y))

    test_org['meter_reading'] = clf.predict(test_org[clf._Booster.feature_name()])
    test_org['meter_reading'] = test_org['meter_reading'].apply(lambda x:np.expm1(x))
    test_org['row_id'] = test_org['row_id'].astype(np.int32)
    test_org[['row_id','meter_reading']].to_csv('submit_m2.csv',index=False)

if __name__ == '__main__':
    main()