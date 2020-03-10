# -*-coding: utf-8-*-
import sys
sys.path.append('/data/tinyv/kw/ppackage')
import pandas as pd
import lightgbm as lgb
from lightgbm.sklearn import *
from sklearn import metrics
from sklearn.model_selection import *
import traceback
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import *
import json
import matplotlib.pyplot as plt
import seaborn as sns
from project_demo.tools.evaluate import *
from project_demo.tools.optimize import *
from copy import deepcopy
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle


def main():
    train_org = pd.read_csv(r'../method3/train_bu_weather.csv')
    train_org['meter_reading'] = train_org['meter_reading'].apply(lambda x: np.log1p(x))
    test_org = pd.read_csv(r'../method3/test_bu_weather.csv')


    #train, test = train_test_split(train_org, test_size=0.33, random_state=42)
    train = train_org

    features = [f for f in train if f not in ['building_id','site_id','meter_reading','timestamp_aligned']]
    train_x = train[features]
    train_y = train['meter_reading']


    d_half_1 = lgb.Dataset(train_x[:int(train_x.shape[0] / 2)], label=train[:int(train.shape[0] / 2)]['meter_reading'], free_raw_data=False)
    d_half_2 = lgb.Dataset(train_x[int(train_x.shape[0] / 2):], label=train[int(train.shape[0] / 2):]['meter_reading'], free_raw_data=False)

    watchlist_1 = [d_half_1, d_half_2]
    watchlist_2 = [d_half_2, d_half_1]

    params = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.9, 'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': 5,
              'min_child_samples': 2825, 'min_child_weight': 0.001, 'min_split_gain': 0.38425225046382727, 'n_estimators': 2000, 'n_jobs': -1,
              'num_leaves': 25, 'objective': None, 'random_state': 24, 'reg_alpha': 464.18134018797014, 'reg_lambda': 1000.0, 'silent': True,
              'subsample': 0.9058845740485875, 'subsample_for_bin': 29730, 'subsample_freq': 1, 'drop_rate': 0.1, 'max_bin': 833, 'sigmoid': 0.824407683379107,
              'is_unbalance': True,"metric": "rmse"}


    print("Building model with first half and validating on second half:")
    model_half_1 = lgb.train(params, train_set=d_half_1, num_boost_round=1000, valid_sets=watchlist_1, verbose_eval=200,
                             early_stopping_rounds=200)

    print("Building model with second half and validating on first half:")
    model_half_2 = lgb.train(params, train_set=d_half_2, num_boost_round=1000, valid_sets=watchlist_2, verbose_eval=200,
                             early_stopping_rounds=200)

    pred = np.expm1(model_half_1.predict(test_org[model_half_1.feature_name()], num_iteration=model_half_1.best_iteration)) / 2

    pred += np.expm1(model_half_2.predict(test_org[model_half_2.feature_name()], num_iteration=model_half_2.best_iteration)) / 2

    test_org['meter_reading'] = pred
    #test_org['meter_reading'] = test_org['meter_reading'].apply(lambda x:np.expm1(x))
    test_org[['row_id','meter_reading']].to_csv('submitm4v1.csv',index=False)


if __name__ == '__main__':
    main()
