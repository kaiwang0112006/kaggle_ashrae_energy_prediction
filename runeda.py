# -*-coding: utf-8-*-
import pandas as pd
import featuretools as ft

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

    weather_train = weather_train.fillna(-1).reset_index()
    weather_test = weather_test.fillna(-1).reset_index()

    weather_feature = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',
                       'sea_level_pressure','wind_direction','wind_speed']

    entities = {
        "data": (train_bu, 'index'),
        "weather": (weather_train, 'index')
    }
    relationships = [("data", "site_id", "weather", "site_id"), ("data", "timestamp", "weather", "timestamp")]
    feature_matrix_train, features_defs = ft.dfs(entities=entities, relationships=relationships,
                                                 trans_primitives=weather_feature,target_entity="data")
    feature_matrix_train.to_csv('train_dfs.csv',index=False)

    entities = {
        "data": (test_bu, "building_id", 'index'),
        "weather": (weather_test, "site_id", "index")
    }
    relationships = [("data", "site_id", "weather", "site_id"), ("data", "timestamp", "weather", "timestamp")]
    feature_matrix_test, features_defs = ft.dfs(entities=entities, relationships=relationships,
                                                trans_primitives=weather_feature,target_entity="data")
    feature_matrix_test.to_csv('test_dfs.csv',index=False)


if __name__ == '__main__':
    main()
