import pandas as pd
import numpy as np

import torch
from torch.nn import functional as F

import datetime
import pickle
import os

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder


PATH_DATA = os.environ["PATH_DATA"]
PATH = os.environ["PATH"]


class DST:
    """daylight saving time correction
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        for h in [datetime.datetime(2021, 10, 31, 3, 0, 0), datetime.datetime(2022, 3, 27, 3, 0, 0),
                  datetime.datetime(2022, 10, 30, 3, 0, 0), datetime.datetime(2023, 3, 26, 3, 0, 0)]:
            index_0 = df[df.datetime==h-datetime.timedelta(seconds=3600)].index
            index_1 = df[df.datetime==h].index
            index_2 = df[df.datetime==h+datetime.timedelta(seconds=3600)].index
            df.loc[index_1, 'target'] = (df.loc[index_0, 'target'].values + 
                                         df.loc[index_2, 'target'].values) / 2
        return df


def date_time_transform(data):
    """polar time transformation:
       year time (hours)
       weekdays (hours)
    """
    data.datetime = pd.to_datetime(data.datetime)
    # year time encoding
    df = data.loc[:, ['datetime', 'data_block_id']].groupby('datetime', as_index=False).data_block_id.first()
    df['time_year'] = (df.datetime.map(lambda x: (x - 
                                                  datetime.datetime(x.year, 1, 1, 0, 0)).total_seconds()
                                       ) / 1800)
    df['time_sin'] = (2 * np.pi * df['time_year'] / 8783).map(np.sin)
    df['time_cos'] = (2 * np.pi * df['time_year'] / 8783).map(np.cos)

    # week time encoding
    df['weekday'] = df.datetime.dt.weekday.astype(int)
    df['hour'] = df.datetime.dt.hour.astype(int)
    on = ['weekday', 'hour']
    df_week = df.loc[:, on].groupby(on, as_index=False)[on].first()
    df_week['time_week_sin'] = (2 * np.pi * (24 * df_week.weekday + df.hour) / 84).map(np.sin)
    df_week['time_week_cos'] = (2 * np.pi * (24 * df_week.weekday + df.hour) / 84).map(np.cos)
    df = df.merge(df_week, on=on, how='left')
    df.drop(['time_year', 'weekday', 'hour'], axis=1, inplace=True)
    df['time'] = df.data_block_id / 700
    return df


def _scaler_selection(_scaler):
    if _scaler == 'minmax':
        scaler = MinMaxScaler()
    elif _scaler == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("scaler belongs to ['minmax', 'standard']")
    return scaler


class DataNormalizer:
    """consumption / production values normalization:
       - standard scaler
       - minmax
       back transformation
    Arguments:
     _scaler belongs to ['minmax', 'standard']
    """
    def fit(self, df, y=None, _scaler='minmax'):
        self.scaler = {}
        self.id_s = df.prediction_unit_id.unique()
        for p in [0, 1]:
            # absolute normalization
            self.scaler[p] = _scaler_selection(_scaler)
            X = df[df.is_consumption==p].target.values.reshape(-1, 1)
            self.scaler[p].fit(X)
            filename = f'absolute_normalizer_{p}.pkl'
            filename = os.path.join(PATH, 'artifacts', filename)
            with open(filename, 'wb') as f:
                pickle.dump(self.scaler[p], f)
            # prediction unit id
            for _id in self.id_s:
                self.scaler[(p, _id)] = MinMaxScaler()
                X = df[(df.is_consumption==p)&
                       (df.prediction_unit_id==_id)].target.values.reshape(-1, 1)
                self.scaler[(p, _id)].fit(X)
                filename = f'scaler_{p}_{_id}.pkl'
                filename = os.path.join(PATH, 'artifacts', filename)
                with open(filename, 'wb') as f:
                    pickle.dump(self.scaler[(p, _id)], f)
                    
        # back transformation
        self.back_transformation = [[0]*69, [0]*69]
        for k in self.scaler.keys():
            if type(k) is tuple:
                self.back_transformation[k[0]][k[1]] = self.scaler[k].inverse_transform

    def transform(self, df):
        df['x'] = 0
        df['x_unit'] = 0
        for p in [0, 1]:
            # absolute normalization
            filename = f'absolute_normalizer_{p}.pkl'
            filename = os.path.join(PATH, 'artifacts', filename)
            with open(filename, 'rb') as f:
                self.scaler[p] = pickle.load(f)
            index = df[df.is_consumption==p].index
            X = df.loc[index, 'target'].values.reshape(-1, 1)
            df.loc[index, 'x'] = self.scaler[p].transform(X)
            # prediction unit id
            for _id in self.id_s:
                filename = f'scaler_{p}_{_id}.pkl'
                filename = os.path.join(PATH, 'artifacts', filename)
                with open(filename, 'rb') as f:
                    self.scaler[(p, _id)] = pickle.load(f)
                index = df[(df.is_consumption==p)&
                       (df.prediction_unit_id==_id)].index
                X = df.loc[index, 'target'].values.reshape(-1, 1)
                df.loc[index, 'x_unit'] = self.scaler[(p, _id)].transform(X)
        return df
    
    def back_transform(self, Y, units):
        Y_transformed = np.zeros(Y.shape)
        for i, k in enumerate(units):
            Y_transformed[i, :24] = self.back_transformation[1][k](Y[i, :24].reshape(-1, 1)).reshape(1, -1)
            Y_transformed[i, 24:] = self.back_transformation[0][k](Y[i, 24:].reshape(-1, 1)).reshape(1, -1)
        return torch.tensor(Y_transformed)


def client_data_transformer(_scaler='minmax'):
    """client data transformer: 1 kind of normalization
    """
    client_data = pd.read_csv(os.path.join(PATH_DATA, 'client.csv'))
    client_data['id'] = client_data.apply(lambda x: (x.county, x.product_type, x.is_business), axis=1)
    for b in [0, 1]:
        df = client_data[client_data.data_block_id==2]
        df.data_block_id = b
        client_data = pd.concat([client_data, df])

    # prediction unit id
    client_data['eic_count_unit_norm'] = 0
    client_data['installed_capacity_unit_norm'] = 0
    id_s = client_data['id'].unique()
    for _id in id_s:
        scaler = _scaler_selection(_scaler)
        index = client_data[client_data.id==_id].index
        for f in ['eic_count', 'installed_capacity']:
            X = client_data.loc[index, f].values.reshape(-1, 1)
            client_data.loc[index, f + '_unit_norm'] = scaler.fit_transform(X)
        
    # absolute normalization
    scaler = _scaler_selection(_scaler)
    for f in ['eic_count', 'installed_capacity']:
        client_data[f] = scaler.fit_transform(client_data[f].values.reshape(-1, 1))
    
    features = ['data_block_id', 'id', 'eic_count', 'installed_capacity', 'eic_count_unit_norm', 'installed_capacity_unit_norm']
    return client_data.loc[:, features]


def prices_transformer(_scaler='minmax'):
    """read, transform, scale and join electricity and gas prices
    """
    # electricity prices
    electricity_prices = pd.read_csv(os.path.join(PATH_DATA, 'electricity_prices.csv'))
    electricity_prices.forecast_date = pd.to_datetime(electricity_prices.forecast_date)

    for h in [datetime.datetime(2022, 3, 27, 2, 0, 0), datetime.datetime(2023, 3, 26, 2, 0, 0)]:
        df = electricity_prices[electricity_prices.forecast_date == h - datetime.timedelta(hours=1)]
        df['forecast_date'] = h
        df[['euros_per_mwh']] = (df[['euros_per_mwh']].values[0][0] +  
                                electricity_prices[electricity_prices.forecast_date == 
                                h + datetime.timedelta(hours=1)].euros_per_mwh.values) / 2
        electricity_prices = electricity_prices.append(df)

    # for joining: DA -> actual time
    electricity_prices.forecast_date = electricity_prices.forecast_date + datetime.timedelta(days=1)
    electricity_prices = electricity_prices.loc[:, ['forecast_date', 'euros_per_mwh']]
    electricity_prices.columns = ['datetime', 'euros_per_mwh']

    # gas prices
    gas_prices = pd.read_csv(os.path.join(PATH_DATA, 'gas_prices.csv'))
    gas_prices.forecast_date = pd.to_datetime(gas_prices.forecast_date)
    # for joining: DA -> actual time
    gas_prices.forecast_date = (gas_prices.forecast_date + datetime.timedelta(days=1)).dt.date
    
    # joining
    electricity_prices['forecast_date'] = electricity_prices.datetime.dt.date
    columns = ['forecast_date', 'lowest_price_per_mwh', 'highest_price_per_mwh']
    electricity_prices = electricity_prices.merge(gas_prices.loc[:, columns], on='forecast_date', how='left')
    electricity_prices.drop('forecast_date', axis=1, inplace=True)
    
    scaler = _scaler_selection(_scaler)
    for f in electricity_prices.columns[1:]:
        X = electricity_prices[f].values.reshape(-1, 1)
        electricity_prices[f] = scaler.fit_transform(X)
    
    return electricity_prices


def weather_transformer(agg=['mean'], _scaler='minmax'):
    """weather files transformation, scaling etc and aggregation 
       hour / country aggregation: min, max, mean
       agg is a subset of ['std', 'mean', 'min', 'max']
    """
    def ws_code(df):
        f = lambda x: str(round(x, 2))
        df['ws_code'] = df['latitude'].map(f) + df['longitude'].map(f)
        df.drop(['latitude', 'longitude'], axis=1, inplace=True)
        return df

    def add_country_id(df, countries):
        df = ws_code(df)
        df = df.merge(countries, on='ws_code', how='left')
        df.drop(['ws_code'], axis=1, inplace=True)
        return df

    # countries <-> weather stations
    countries = pd.read_csv(os.path.join(PATH_DATA, 'weather_station_to_county_mapping.csv'))
    countries.county = countries.county.fillna(12)
    countries.county = countries.county.astype(int)
    countries = ws_code(countries)
    countries.drop(['county_name'], axis=1, inplace=True)

    # historical_weather
    historical_weather = pd.read_csv(os.path.join(PATH_DATA, 'historical_weather.csv'))
    historical_weather.datetime = pd.to_datetime(historical_weather.datetime)
    #historical_weather.datetime = historical_weather.datetime + datetime.timedelta(hours=1)
    historical_weather = add_country_id(historical_weather, countries)

    columns = ['temperature', 'dewpoint', 'rain', 'snowfall', 'surface_pressure', 'cloudcover_total', 
             'cloudcover_low', 'cloudcover_mid', 'cloudcover_high', 'windspeed_10m',
             'winddirection_10m', 'shortwave_radiation', 'direct_solar_radiation',
             'diffuse_radiation']

    scaler = _scaler_selection(_scaler)

    for f in columns:
        X = historical_weather[f].values.reshape(-1, 1)
        historical_weather[f] = scaler.fit_transform(X)
        
    df_historical = historical_weather.groupby(['datetime', 'county'], 
                                               as_index=True)[columns].aggregate(agg).fillna(0)
    df_historical = df_historical.reset_index()
    
    df_historical.columns = ['datetime', 'county'] + ['w' + str(k) for k in range(df_historical.shape[1]-2)]
    df_historical.to_csv(os.path.join(PATH, 'data', 'historical_weather_transformed.csv'), index=False)

    del df_historical, historical_weather

    # forecast_weather
    forecast_weather = pd.read_csv(os.path.join(PATH_DATA, 'forecast_weather.csv'))
    forecast_weather.surface_solar_radiation_downwards.fillna(0, inplace=True)
    forecast_weather.forecast_datetime = pd.to_datetime(forecast_weather.forecast_datetime)
    
    forecast_weather = forecast_weather.query('21 < hours_ahead < 46')
    forecast_weather = forecast_weather.rename(columns={'datetime': 'forecast_datetime'})
    forecast_weather = add_country_id(forecast_weather, countries)
    
    columns = ['temperature', 'dewpoint', 'cloudcover_high', 'cloudcover_low', 'cloudcover_mid',
               'cloudcover_total', '10_metre_u_wind_component', '10_metre_v_wind_component', 
               'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall', 
               'total_precipitation']
    for f in columns:
        X = forecast_weather[f].values.reshape(-1, 1)
        forecast_weather[f] = scaler.fit_transform(X)
    df_forecast = forecast_weather.groupby(['forecast_datetime', 'county'] 
                                          )[columns].aggregate(agg).fillna(0)
    df_forecast = df_forecast.reset_index()
    df_forecast.columns = ['datetime', 'county'
                          ] + ['w' + str(k) for k in range(df_forecast.shape[1]-2)]
    
    # daylight saving time correction
    for h in [datetime.datetime(2022, 3, 27, 3, 0, 0), 
              datetime.datetime(2023, 3, 26, 3, 0, 0)]:
        df = df_forecast[df_forecast.datetime==h-datetime.timedelta(hours=1)]
        df.iloc[:, 2] = (df.iloc[:, 2].values + 
                         df_forecast[df_forecast.datetime==h+datetime.timedelta(hours=1)].iloc[:, 2].values
                        ) / 2
        df.datetime = h
        df_forecast = pd.concat([df_forecast, df])
    df_forecast.to_csv(os.path.join(PATH, 'data', 'forecast_weather_transformed.csv'), index=False)


class DataLoader:
    def __init__(self,
                 targets=['consumption', 'production'],
                 ):
        self.targets = targets

        production = pd.read_csv(os.path.join(PATH, 'data', 'train_production.csv'))
        consumption = pd.read_csv(os.path.join(PATH, 'data', 'train_consumption.csv'))

        # frame
        self.frame = production.loc[:, ['datetime', 'county', 'prediction_unit_id', 'data_block_id', 'row_id']]
        le = LabelEncoder()
        le.fit(self.frame.datetime)
        self.frame['hour'] = le.transform(self.frame.datetime)
        self.frame.drop('datetime', axis=1, inplace=True)
        self.hours = self.frame.groupby('hour', as_index=False).data_block_id.first()

        # weights
        self.weights = {}
        #f = lambda x: np.log10(5 + x)
        f = lambda x: x**0.3
        self.weights[0] = f(production.groupby('prediction_unit_id').target.mean().sort_index().values)
        self.weights[1] = f(consumption.groupby('prediction_unit_id').target.mean().sort_index().values)

        # mappers
        self.counties = self.frame.groupby('prediction_unit_id', as_index=False).county.first()
        self.all_units = set(self.frame.prediction_unit_id.unique())

        # consumption / production X
        values = ['x', 'x_unit']
        #values = ['x']
        n = len(values)

        df = production.pivot(index='datetime', columns='prediction_unit_id', values=values)
        columns = df.sort_index(axis=1,level=[1, 0], ascending=[True, True]).columns
        df = df.loc[:, columns]
        shape = df.shape[0], df.shape[1] // n,  n
        self.production = torch.tensor(df.to_numpy(), dtype=torch.float32).reshape(*shape).transpose(1, 0)

        df = consumption.pivot(index='datetime', columns='prediction_unit_id', values=values)
        columns = df.sort_index(axis=1,level=[1, 0], ascending=[True, True]).columns
        df = df.loc[:, columns]
        self.consumption = torch.tensor(df.to_numpy(), dtype=torch.float32).reshape(*shape).transpose(1, 0)
        
        # client data 
        client_data = pd.read_csv(os.path.join(PATH, 'data', 'client_data.csv'))
        values = ['eic_count', 'installed_capacity', 'eic_count_unit_norm', 'installed_capacity_unit_norm']
        #values = ['eic_count', 'installed_capacity']
        n = len(values)
        df = client_data.pivot(index='datetime', columns='prediction_unit_id', values=values)
        columns = df.sort_index(axis=1,level=[1, 0], ascending=[True, True]).columns
        df = df.loc[:, columns]
        shape = df.shape[0], df.shape[1] // n,  n
        self.client_data = torch.tensor(df.to_numpy(), dtype=torch.float32).reshape(*shape).transpose(1, 0)

        # consumption / production Y
        #values = ['target']
        values = ['x_unit']
        production = production.pivot(index='datetime', columns='prediction_unit_id', values=values)
        self.target_production = torch.tensor(production.to_numpy(), dtype=torch.float32)
        consumption = consumption.pivot(index='datetime', columns='prediction_unit_id', values=values)
        self.target_consumption = torch.tensor(consumption.to_numpy(), dtype=torch.float32)
        del production, consumption, client_data

        # prices
        self.prices = prices_transformer().sort_values('datetime')
        self.prices['hour'] = le.transform(self.prices.datetime.dt.strftime('%Y-%m-%d %H:%M:%S'))
        self.prices = self.frame.loc[:, ['hour']].merge(self.prices, on='hour', how='left')
        self.prices = self.prices.fillna(method='bfill')
        self.prices = self.prices.sort_values('hour')
        self.prices.drop(['datetime', 'hour'], axis=1, inplace=True)
        self.prices = torch.tensor(self.prices.values, dtype=torch.float32)
        
        # cat. profile
        self.prediction_units = pd.read_csv(os.path.join(PATH, 'data', 'prediction_units.csv'))
        self.prediction_units = pd.concat([self.prediction_units.iloc[:, -2],
        pd.get_dummies(self.prediction_units.iloc[:, -1])], axis=1)
        self.prediction_units= torch.tensor(self.prediction_units.values, dtype=torch.float32)

        # time profile
        self.date_time = pd.read_csv(os.path.join(PATH, 'data', 'date_time.csv'))
        self.date_time['hour'] = le.transform(self.date_time.datetime)
        self.date_time = self.date_time.sort_values('hour')
        self.date_time.datetime = pd.to_datetime(self.date_time.datetime)
        self.weekdays = self.date_time.loc[:, ['datetime']]
        self.weekdays['weekday'] = self.weekdays.datetime.dt.weekday
        self.weekdays = pd.get_dummies(self.weekdays.iloc[:, -1])
        self.weekdays = torch.tensor(self.weekdays.values, dtype=torch.float32)
        
        #self.date_time = torch.tensor(self.date_time.iloc[:, -4:].values, dtype=torch.float32)
        self.date_time = torch.tensor(self.date_time.iloc[:, -5:].values, dtype=torch.float32)
        
        # hist. weather
        df = pd.read_csv(os.path.join(PATH, 'data', 'historical_weather_transformed.csv'))
        df['hour'] = le.transform(df.datetime)
        values = df.columns[2: -1]
        n = len(values)
        df = df.pivot_table(index='hour', columns='county', values=values)
        columns = df.sort_index(axis=1,level=[1, 0], ascending=[True, True]).columns
        df = df.loc[:, columns]
        shape = df.shape[0], df.shape[1] // n,  n
        self.historical_weather = torch.tensor(df.to_numpy(), dtype=torch.float32).reshape(*shape).transpose(1, 0)
        
        # forecast weather
        self.forecast_weather = pd.read_csv(os.path.join(PATH, 'data', 'forecast_weather_transformed.csv'))
        self.forecast_weather['hour'] = le.transform(self.forecast_weather.datetime)
        df = self.forecast_weather[self.forecast_weather.hour < 48]
        df.hour = df.hour - 24
        self.forecast_weather = pd.concat([df, self.forecast_weather])
        values = self.forecast_weather.columns[2: -1]
        n = len(values)
        df = self.forecast_weather.pivot_table(index='hour', columns='county', values=values)
        columns = df.sort_index(axis=1,level=[1, 0], ascending=[True, True]).columns
        df = df.loc[:, columns]
        shape = df.shape[0], df.shape[1] // n,  n
        self.forecast_weather = torch.tensor(df.to_numpy(), dtype=torch.float32).reshape(*shape).transpose(1, 0)
        
    def _add_cat_profile(self, X, prediction_units):
        Z = torch.unsqueeze(self.prediction_units[prediction_units], dim=-1)
        Z = torch.unsqueeze(Z, dim=-1)
        Z = Z.transpose(3, 1)
        Z = Z.repeat(1, X.shape[1], X.shape[2], 1)
        X = torch.cat([X, Z], dim=-1)
        return X

    def _add_date_time_profile(self, X, prediction_units):
        Z = torch.unsqueeze(self.date_time[prediction_units], dim=-1)
        Z = torch.unsqueeze(Z, dim=-1)
        Z = Z.transpose(3, 1)
        Z = Z.repeat(1, X.shape[1], X.shape[2], 1)
        X = torch.cat([X, Z], dim=-1)
        return X

    def generate(self, target_data_block, hist_shifts=[2], mode='train'):
        """
        Arguments: 
           target_data_block - target data block
           hist_shifts - historical features (2 - last day with information, 7 - a week ago)
        """
        # prediction_units:
        prediction_units = self.all_units

        for b in [target_data_block] + [target_data_block - k for k in hist_shifts]:
            prediction_units = prediction_units & set(self.frame[self.frame.data_block_id==b].prediction_unit_id.unique())
        
        #if 28 in prediction_units:
        #    prediction_units.remove(28)

        prediction_units = sorted(list(prediction_units))

        # ---- encoder ----
        blocks_in = [target_data_block - k for k in hist_shifts]
        hours = self.hours[self.hours.data_block_id.isin(blocks_in)].hour.values

        X = []
        # X (production / consumption) - encoder
        X.append(self.consumption[prediction_units, :, :][:, hours, :])
        X.append(self.production[prediction_units, :, :][:, hours, :])
        
        # client data
        X.append(self.client_data[prediction_units, :, :][:, hours, :])

        # prices
        x = torch.unsqueeze(self.prices[hours], dim=0).repeat(X[0].shape[0], 1, 1)
        X.append(x)
        
        # time
        x = torch.unsqueeze(self.date_time[hours], dim=0)
        x = x.repeat(X[0].shape[0], 1, 1)
        X.append(x)

        # weather
        county_idx = self.counties[self.counties.prediction_unit_id.isin(prediction_units)].loc[:, 'county'].values
        x = self.historical_weather[county_idx, :, :]
        x = x[:, hours, :]
        X.append(x)
    
        X = torch.cat(X, dim=2)
        X = torch.unsqueeze(X, dim=-1)
        
        # add cat profile
        X = self._add_cat_profile(X, prediction_units)

        # add weekday profile
        #Z = torch.unsqueeze(self.weekdays[hours], dim=-1)
        #Z = torch.unsqueeze(Z, dim=0)
        #Z = Z.transpose(3, 2)
        #Z = Z.repeat(X.shape[0], 1, X.shape[2], 1)
        #X = torch.cat([X, Z], dim=-1)

        # ---- decoder ----
        H = []

        # current block
        blocks_in = [target_data_block]
        hours = self.hours[self.hours.data_block_id.isin(blocks_in)].hour.values

        n = len(prediction_units)
        
        # client data
        H.append(self.client_data[prediction_units, :, :][:, hours, :])

        # prices - decoder
        x = torch.unsqueeze(self.prices[hours], dim=0).repeat(n, 1, 1)
        H.append(x)
        
        # time - decoder
        x = torch.unsqueeze(self.date_time[hours], dim=0)
        x = x.repeat(n, 1, 1)
        H.append(x)
        
        # weather - decoder
        county_idx = self.counties[self.counties.prediction_unit_id.isin(prediction_units)].loc[:, 'county'].values
        x = self.forecast_weather[county_idx, :, :]
        x = x[:, hours, :]
        H.append(x)

        H = torch.cat(H, dim=2)
        H = torch.unsqueeze(H, dim=-1)

        # add cat profile
        #Z = torch.unsqueeze(self.date_time[prediction_units], dim=-1).transpose(2, 1)
        #Z = Z.repeat(1, H.shape[1], 1)
        #H = torch.cat([H, Z], dim=-1)
        H = self._add_cat_profile(H, prediction_units)

        # target
        Y = []
        if 'consumption' in self.targets:
            y = self.target_consumption[hours, :][:, prediction_units]
            Y.append(y)
        if 'production' in self.targets:
            y = self.target_production[hours, :][:, prediction_units]
            Y.append(y)
        Y = torch.cat(Y, dim=0).transpose(1, 0)

        return X, H, Y, prediction_units


def weather_loader():
    """weather files transformation (without aggregation)
    """
    def ws_code(df):
        f = lambda x: str(round(x, 2))
        df['ws_code'] = df['latitude'].map(f) + df['longitude'].map(f)
        df.drop(['latitude', 'longitude'], axis=1, inplace=True)
        return df
    
    # countries <-> weather stations
    countries = pd.read_csv(os.path.join(PATH_DATA, 'weather_station_to_county_mapping.csv'))
    countries.county = countries.county.fillna(12)
    countries.county = countries.county.astype(int)
    countries = ws_code(countries)
    countries.drop(['county_name'], axis=1, inplace=True)

    historical_weather = pd.read_csv(os.path.join(PATH_DATA, 'historical_weather.csv'))
    historical_weather.datetime = pd.to_datetime(historical_weather.datetime)
    historical_weather = ws_code(historical_weather)

    columns = ['temperature', 'dewpoint', 'rain', 'snowfall', 'surface_pressure', 'cloudcover_total', 
             'cloudcover_low', 'cloudcover_mid', 'cloudcover_high', 'windspeed_10m',
             'winddirection_10m', 'shortwave_radiation', 'direct_solar_radiation',
             'diffuse_radiation']
    
    scaler = _scaler_selection(_scaler)

    for f in columns:
        X = historical_weather[f].values.reshape(-1, 1)
        historical_weather[f] = scaler.fit_transform(X)
    historical_weather.drop('data_block_id', axis=1, inplace=True)
    historical_weather = historical_weather.groupby(['datetime', 'ws_code'], as_index=False).mean()
    historical_weather = historical_weather.pivot(index='datetime', columns='ws_code', values=columns)
    historical_weather = historical_weather.sort_index()
    historical_weather.columns = historical_weather.columns.droplevel(1) 
    historical_weather.to_csv(os.path.join(PATH, 'data', 'historical_weather_pivoted.csv'), index=False)
    
    # forecast_weather
    forecast_weather = pd.read_csv(os.path.join(PATH_DATA, 'forecast_weather.csv'))
    forecast_weather.surface_solar_radiation_downwards.fillna(0, inplace=True)
    forecast_weather.forecast_datetime = pd.to_datetime(forecast_weather.forecast_datetime)
    
    forecast_weather = forecast_weather.query('21 < hours_ahead < 46')
    forecast_weather = forecast_weather.rename(columns={'datetime': 'forecast_datetime'})
    forecast_weather = ws_code(forecast_weather)
    
    columns = ['temperature', 'dewpoint', 'cloudcover_high', 'cloudcover_low', 'cloudcover_mid',
               'cloudcover_total', '10_metre_u_wind_component', '10_metre_v_wind_component', 
               'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall', 
               'total_precipitation']

    for f in columns:
        X = forecast_weather[f].values.reshape(-1, 1)
        forecast_weather[f] = scaler.fit_transform(X)
    forecast_weather.drop('data_block_id', axis=1, inplace=True)
    forecast_weather = forecast_weather.groupby(['forecast_datetime', 'ws_code'], as_index=False).mean()
    forecast_weather = forecast_weather.pivot(index='forecast_datetime', columns='ws_code', values=columns)

    for h in [datetime.datetime(2022, 3, 27, 3, 0, 0), 
              datetime.datetime(2023, 3, 26, 3, 0, 0)]:
        df = forecast_weather[forecast_weather.index==h-datetime.timedelta(hours=1)]
        df.loc[:, :] = (df.values + 
                        forecast_weather[forecast_weather.index==h+datetime.timedelta(hours=1)].values) / 2
        df.index += datetime.timedelta(hours=1)
        forecast_weather = pd.concat([forecast_weather, df])

    df = forecast_weather.iloc[:24, :]
    df.index -= datetime.timedelta(days=1)
    forecast_weather = pd.concat([df, forecast_weather])
    forecast_weather = forecast_weather.sort_index()
    forecast_weather.columns = forecast_weather.columns.droplevel(1) 
    forecast_weather.to_csv(os.path.join(PATH, 'data', 'forecast_weather_pivoted.csv'), index=False)

