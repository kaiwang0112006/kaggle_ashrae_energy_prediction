# -*-coding: utf-8-*-
import sys
sys.path.append('/data/tinyv/kw/ppackage')
import pandas as pd
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
    train_org = pd.read_csv('train_bu_weather.csv')
    train_org['meter_reading'] = train_org['meter_reading'].apply(lambda x: np.log1p(x))
    test_org = pd.read_csv('test_bu_weather.csv')


    train, test = train_test_split(train_org, test_size=0.33, random_state=42)

    features = [f for f in train if f not in ['building_id','site_id','meter_reading']]
    train_x = train[features]
    train_y = train['meter_reading']

    test_x = test[features]
    test_y = test['meter_reading']

    parms = {
        # 'x_train':X_train,
        # 'y_train':y_train,
        'num_leaves': (5, 40),
        'colsample_bytree': (0.1, 0.9),
        'drop_rate': (0.1, 1),
        'learning_rate': (0.001, 0.1),
        'max_bin': (10, 1000),
        'max_depth': (2, 5),
        'min_split_gain': (0.1, 0.9),
        'min_child_samples': (2, 9000),
        'n_estimators': (50, 2000),
        'reg_alpha': (0.1, 1000),
        'reg_lambda': (0.1, 1000),
        'sigmoid': (0.1, 1),
        'subsample': (0.1, 1),
        'subsample_for_bin': (100, 50000),
        'subsample_freq': (0, 5)
    }

    # 参数整理格式，其实只需要提供parms里的参数即可
    intdeal = ['max_bin', 'max_depth', 'max_drop', 'min_child_samples',
               'min_child_weight', 'n_estimators', 'num_leaves', 'scale_pos_weight',
               'subsample_for_bin', 'subsample_freq']  # int类参数
    middledeal = ['colsample_bytree', 'drop_rate', 'learning_rate',
                  'min_split_gain', 'skip_drop', 'subsample', '']  # float， 只能在0，1之间
    maxdeal = ['reg_alpha', 'reg_lambda', 'sigmoid']  # float，且可以大于1

    others = {'is_unbalance': True, 'random_state': 24}

    bayesopsObj = bayes_ops(estimator=LGBMRegressor, param_grid=parms, cv=5, intdeal=intdeal, middledeal=middledeal,
                            maxdeal=maxdeal,
                            score_func='neg_mean_squared_error',
                            init_points=3, n_iter=10, acq="ucb", kappa=1, others=others
                            )
    bayesopsObj.run(X=train_x, Y=train_y)
    parms = bayesopsObj.baseparms

    clf = LGBMRegressor(**parms)
    clf.fit(train_x, train_y)
    train_y_pred = clf.predict(train_x)
    y_pred = clf.predict(test_x)
    print(mean_squared_error(train_y_pred,train_y), mean_squared_error(y_pred,test_y))

    test_org['meter_reading'] = clf.predict(test_org[clf._Booster.feature_name()])
    test_org['meter_reading'] = test_org['meter_reading'].apply(lambda x:np.expm1(x))
    test_org[['row_id','meter_reading']].to_csv('submitm2opt.csv',index=False)
    with open('m2opt.pkl', 'wb') as f:
        pickle.dump(clf, f)

if __name__ == '__main__':
    main()
