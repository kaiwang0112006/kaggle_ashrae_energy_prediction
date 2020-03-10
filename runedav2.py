# -*-coding: utf-8-*-
import pandas as pd
import featuretools as ft
import copy

def main():
    building = pd.read_csv('building_metadata.csv')
    weather_train = pd.read_csv('weather_train.csv')
    weather_test = pd.read_csv('weather_test.csv')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    building = building.fillna(-1)

    map_use = dict(zip(building['primary_use'].value_counts().sort_index().keys(),
                       range(1, len(building['primary_use'].value_counts()) + 1)))

    building['primary_use'] = building['primary_use'].replace(map_use)
    train_bu = pd.merge(train,building,on='building_id').reset_index()
    test_bu = pd.merge(test,building,on='building_id').reset_index()

    weather_feature = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',
                       'sea_level_pressure','wind_direction','wind_speed']

    weather_train = weather_train.fillna(0)
    weather_train['day'] = weather_train['timestamp'].apply(lambda x: x[:10])

    train_bu['day'] = train_bu['timestamp'].apply(lambda x: x[:10])
    train_bu_weather = copy.deepcopy(train_bu)
    for f in ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
              'wind_direction', 'wind_speed']:
        df = weather_train.groupby(['site_id', 'day'])[f].agg(['mean', 'max', 'min', pd.Series.mode]).reset_index(
            col_level=-1, drop=False)
        col = {'mean': f + '_mean', 'max': f + '_max', 'min': f + '_min', 'mode': f + '_mode'}
        df = df.rename(columns=col)
        train_bu_weather = pd.merge(train_bu_weather, df[list(col.values()) + ['site_id', 'day']],
                                    on=['site_id', 'day'], how='left')
    train_bu_weather.to_csv('train_bu_weather.csv',index=False)

    weather_test = weather_test.fillna(0)
    weather_test['day'] = weather_test['timestamp'].apply(lambda x: x[:10])
    test_bu['day'] = test_bu['timestamp'].apply(lambda x: x[:10])
    test_bu_weather = copy.deepcopy(test_bu)
    for f in ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
              'wind_direction', 'wind_speed']:
        df = weather_train.groupby(['site_id', 'day'])[f].agg(['mean', 'max', 'min', pd.Series.mode]).reset_index(
            col_level=-1, drop=False)
        col = {'mean': f + '_mean', 'max': f + '_max', 'min': f + '_min', 'mode': f + '_mode'}
        df = df.rename(columns=col)
        test_bu_weather = pd.merge(test_bu_weather, df[list(col.values()) + ['site_id', 'day']],
                                    on=['site_id', 'day'], how='left')
    test_bu_weather.to_csv('test_bu_weather.csv',index=False)

if __name__ == '__main__':
    main()
