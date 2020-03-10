# -*-coding: utf-8-*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
import math

## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def main():
    building = pd.read_csv('../building_metadata.csv')
    weather_train = pd.read_csv('../weather_train.csv',parse_dates=['timestamp'])
    weather_train['split']=1
    weather_test = pd.read_csv('../weather_test.csv',parse_dates=['timestamp'])
    weather_test['split'] = 2
    train = pd.read_csv('../train.csv',parse_dates=['timestamp'])
    test = pd.read_csv('../test.csv',parse_dates=['timestamp'])

    weather = pd.concat(
        [
            weather_train,
            weather_test
        ],
        ignore_index=True
    )

    weather_key = ['site_id', 'timestamp']
    temp_skeleton = weather[weather_key + ['air_temperature']] \
        .drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()

    # calculate ranks of hourly temperatures within date/site_id chunks
    temp_skeleton['temp_rank'] = temp_skeleton.groupby(
        ['site_id', temp_skeleton.timestamp.dt.date],
    )['air_temperature'].rank('average')

    # create 2D dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)
    df_2d = temp_skeleton.groupby(
        ['site_id', temp_skeleton.timestamp.dt.hour]
    )['temp_rank'].mean().unstack(level=1)

    # align scale, so each value within row is in [0,1] range
    df_2d = df_2d / df_2d.max(axis=1).values.reshape((-1, 1))

    # sort by 'closeness' of hour with the highest temperature
    site_ids_argmax_maxtemp = pd.Series(np.argmax(df_2d.values, axis=1)).sort_values().index

    # assuming (1,5,12) tuple has the most correct temp peaks at 14:00
    site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)
    site_ids_offsets.index.name = 'site_id'
    site_ids_offsets.sort_values()

    #temp_skeleton['offset'] = temp_skeleton.site_id.map(site_ids_offsets)
    # add offset
    #temp_skeleton['timestamp_aligned'] = (
    #        temp_skeleton.timestamp
    #        - pd.to_timedelta(temp_skeleton.offset, unit='H')
    #)
    weather_train['offset'] = weather_train.site_id.map(site_ids_offsets)
    weather_train['timestamp_aligned'] = (weather_train.timestamp - pd.to_timedelta(weather_train.offset, unit='H'))
    weather_test['offset'] = weather_test.site_id.map(site_ids_offsets)
    weather_test['timestamp_aligned'] = (weather_test.timestamp - pd.to_timedelta(weather_test.offset, unit='H'))
    ## REducing memory
    train_df = reduce_mem_usage(train)
    test_df = reduce_mem_usage(test)

    weather_train_df = reduce_mem_usage(weather_train)
    weather_test_df = reduce_mem_usage(weather_test)
    building_meta_df = reduce_mem_usage(building)


    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])
    weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])

    building_meta_df['primary_use'] = building_meta_df['primary_use'].astype('category')

    temp_df = train_df[['building_id']]
    temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')
    del temp_df['building_id']
    train_df = pd.concat([train_df, temp_df], axis=1)

    temp_df = test_df[['building_id']]
    temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')

    del temp_df['building_id']
    test_df = pd.concat([test_df, temp_df], axis=1)
    del temp_df, building_meta_df

    temp_df = train_df[['site_id', 'timestamp']]
    temp_df = temp_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

    del temp_df['site_id'], temp_df['timestamp']
    train_df = pd.concat([train_df, temp_df], axis=1)
    train_df['timestamp'] = train_df['timestamp_aligned']
    del train_df['timestamp_aligned']

    temp_df = test_df[['site_id', 'timestamp']]
    temp_df = temp_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')

    del temp_df['site_id'], temp_df['timestamp']
    test_df = pd.concat([test_df, temp_df], axis=1)
    test_df['timestamp'] = test_df['timestamp_aligned']
    del test_df['timestamp_aligned']

    del temp_df, weather_train_df, weather_test_df

    train_df['age'] = train_df['year_built'].max() - train_df['year_built'] + 1
    test_df['age'] = test_df['year_built'].max() - test_df['year_built'] + 1

    le = LabelEncoder()
    le.fit(pd.concat([train_df[['primary_use']],test_df[['primary_use']]])['primary_use'])
    # train_df['primary_use'] = train_df['primary_use'].astype(str)
    train_df['primary_use'] = le.transform(train_df['primary_use']).astype(np.int8)

    # test_df['primary_use'] = test_df['primary_use'].astype(str)
    test_df['primary_use'] = le.transform(test_df['primary_use']).astype(np.int8)

    train_df['floor_count'] = train_df['floor_count'].fillna(-999).astype(np.int16)
    test_df['floor_count'] = test_df['floor_count'].fillna(-999).astype(np.int16)

    train_df['year_built'] = train_df['year_built'].fillna(-999).astype(np.int16)
    test_df['year_built'] = test_df['year_built'].fillna(-999).astype(np.int16)

    train_df['age'] = train_df['age'].fillna(-999).astype(np.int16)
    test_df['age'] = test_df['age'].fillna(-999).astype(np.int16)

    train_df['cloud_coverage'] = train_df['cloud_coverage'].fillna(-999).astype(np.int16)
    test_df['cloud_coverage'] = test_df['cloud_coverage'].fillna(-999).astype(np.int16)

    train_df['month_datetime'] = train_df['timestamp'].dt.month.astype(np.int8)
    train_df['weekofyear_datetime'] = train_df['timestamp'].dt.weekofyear.astype(np.int8)
    train_df['dayofyear_datetime'] = train_df['timestamp'].dt.dayofyear.astype(np.int16)

    train_df['hour_datetime'] = train_df['timestamp'].dt.hour.astype(np.int8)
    train_df['day_week'] = train_df['timestamp'].dt.dayofweek.astype(np.int8)
    train_df['day_month_datetime'] = train_df['timestamp'].dt.day.astype(np.int8)
    train_df['week_month_datetime'] = train_df['timestamp'].dt.day / 7
    train_df['week_month_datetime'] = train_df['week_month_datetime'].apply(lambda x: math.ceil(x)).astype(np.int8)

    train_df['year_built'] = train_df['year_built'] - 1900
    train_df['square_feet'] = np.log(train_df['square_feet'])

    test_df['month_datetime'] = test_df['timestamp'].dt.month.astype(np.int8)
    test_df['weekofyear_datetime'] = test_df['timestamp'].dt.weekofyear.astype(np.int8)
    test_df['dayofyear_datetime'] = test_df['timestamp'].dt.dayofyear.astype(np.int16)

    test_df['hour_datetime'] = test_df['timestamp'].dt.hour.astype(np.int8)
    test_df['day_week'] = test_df['timestamp'].dt.dayofweek.astype(np.int8)
    test_df['day_month_datetime'] = test_df['timestamp'].dt.day.astype(np.int8)
    test_df['week_month_datetime'] = test_df['timestamp'].dt.day / 7
    test_df['week_month_datetime'] = test_df['week_month_datetime'].apply(lambda x: math.ceil(x)).astype(np.int8)

    test_df['year_built'] = test_df['year_built'] - 1900
    test_df['square_feet'] = np.log(test_df['square_feet'])

    drop_cols = ["precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed", "timestamp"]
    #target = np.log1p(train_df["meter_reading"])
    #del train["meter_reading"]
    train_df = train_df.drop(drop_cols, axis=1)
    test_df = test_df.drop(drop_cols, axis=1)
    train_df.to_csv('train_bu_weather_m3.csv', index=False)
    test_df.to_csv('test_bu_weather_m3.csv', index=False)
    print(train_df.shape)
    print(test_df.shape)

if __name__ == '__main__':
    main()
