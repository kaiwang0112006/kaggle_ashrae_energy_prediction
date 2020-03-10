# -*-coding: utf-8-*-

import sys
sys.path.append(r'/data/tinyv/kw/ppackage')
import numpy as np
import pandas as pd
import os
from sklearn.pipeline import *
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model
from keras.losses import mean_squared_error as mse_loss
from keras import optimizers
from keras.optimizers import RMSprop, Adam
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from project_demo.tools.features_engine import *
from project_demo.tools.multi_apply import *
from project_demo.tools.evaluate import *
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
np.random.seed(2017)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def main():
    train_org = pd.read_csv(r'train_bu_weatherv1.csv')
    train_org['meter_reading'] = train_org['meter_reading'].apply(lambda x: np.log1p(x))
    test_org = pd.read_csv(r'test_bu_weatherv1.csv')

    dropcals = ['index','timestamp','building_id','site_id','meter_reading','timestamp_aligned']
    categoricals = ["primary_use", "hour_datetime", "weekday", "meter"]
    features = [f for f in train_org if f not in dropcals]
    numericals = [f for f in features if f not in categoricals]

    train, test = train_test_split(train_org, test_size=0.2, random_state=42)
    train_x = train[features]
    train_y = train['meter_reading']
    valid_x = test[features]
    valid_y = test['meter_reading']



    train_org = train_org.fillna(-1)
    step1 = ('MinMaxScaler', minmaxScalerClass(cols=numericals, target=""))

    pipeline = Pipeline(steps=[step1])
    train_x_new = pipeline.fit_transform(train_x)
    valid_x_new = pipeline.transform(valid_x)
    test_org_new = pipeline.transform(test_org)

    dense_dim_1=64
    dense_dim_2=32
    dense_dim_3=32
    dense_dim_4=16
    dropout1=0.2
    dropout2=0.1
    dropout3=0.1
    dropout4=0.1
    lr=0.001
    patience = 3
    batch_size = 1024
    epochs = 10

    meter = Input(shape=[1], name="meter")
    primary_use = Input(shape=[1], name="primary_use")
    hour_datetime = Input(shape=[1], name="hour_datetime")
    weekday = Input(shape=[1], name="weekday")
    other_in = Input(shape=[len(features)-4,], name="other_in")

    #Embeddings layers
    emb_meter = Embedding(4, 2)(meter)
    emb_primary_use = Embedding(16, 2)(primary_use)
    emb_hour = Embedding(24, 3)(hour_datetime)
    emb_weekday = Embedding(7, 2)(weekday)

    concat_emb = concatenate([
          Flatten()(emb_meter)
        , Flatten()(emb_primary_use)
        , Flatten()(emb_hour)
        , Flatten()(emb_weekday)
    ])

    categ = Dropout(dropout1)(Dense(dense_dim_1, activation='relu')(concat_emb))
    categ = BatchNormalization()(categ)
    categ = Dropout(dropout2)(Dense(dense_dim_2, activation='relu')(categ))

    main_l = concatenate([
        categ, other_in
    ])

    main_l = Dropout(dropout3)(Dense(dense_dim_3, activation='relu')(main_l))
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(dropout4)(Dense(dense_dim_4, activation='relu')(main_l))

    # output
    output = Dense(1)(main_l)

    model = Model([meter,
                   primary_use,
                   hour_datetime,
                   weekday,
                   other_in], output)
    #checkpoint = ModelCheckpoint('clf_weights.{epoch:03d}-{val_root_mean_squared_error:.4f}.hdf5', monitor='val_root_mean_squared_error', verbose=1,
    #                             save_best_only=True, mode='min')
    checkpoint = ModelCheckpoint('clf_weights.hdf5',
                                 monitor='val_root_mean_squared_error', verbose=1,
                                 save_best_only=True, mode='min')
    model.compile(optimizer="adam", loss=mse_loss, metrics=[root_mean_squared_error])
    early_stopping = EarlyStopping(patience=patience, verbose=1)
    model.fit([train_x_new['meter'],train_x_new['primary_use'],
               train_x_new['hour_datetime'],train_x_new['weekday'],
               train_x_new[[f for f in train_x_new if f not in ['meter','primary_use','hour_datetime','weekday']]]],
              train_y, batch_size=batch_size, epochs=epochs,verbose=1,
              callbacks=[early_stopping, checkpoint],
              validation_data=([valid_x_new['meter'], valid_x_new['primary_use'],
                                valid_x_new['hour_datetime'], valid_x_new['weekday'],
                                valid_x_new[[f for f in valid_x_new if
                                             f not in ['meter', 'primary_use', 'hour_datetime', 'weekday']]]],
                               valid_y))

    model.save_weights('modelv1.h5')
    json_string = model.to_json()
    joblib.dump(json_string, 'modelv1.pkl')

    test_org_new['meter_reading'] = model.predict(test_org_new[features])
    test_org_new['meter_reading'] = model.predict([test_org_new['meter'],test_org_new['primary_use'],
                                                   test_org_new['hour_datetime'],test_org_new['weekday'],
                                                   test_org_new[[f for f in test_org_new if f not in ['meter','primary_use','hour_datetime','weekday']]]])
    test_org_new['meter_reading'] = test_org_new['meter_reading'].apply(lambda x:np.expm1(x))
    test_org_new[['row_id','meter_reading']].to_csv('submitm5v1.csv',index=False)

if __name__ == '__main__':
    main()












