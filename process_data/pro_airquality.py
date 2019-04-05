import numpy as np
import pandas as pd

beijing_aq = pd.read_csv('../Data/beijing_aq.csv')
beijing_meo = pd.read_csv('../Data/beijing_meo.csv')
aq_station_region = pd.read_csv("../Data/station.csv")

beijing_aq = beijing_aq.fillna(method='pad')

beijing_aq.set_index(['station_id', 'time'], inplace=True)
dupes = beijing_aq[beijing_aq.index.duplicated()]
beijing_aq.drop(dupes.index, inplace=True)
beijing_aq = beijing_aq.reset_index()

beijing_aq = pd.merge(beijing_aq,aq_station_region,how='left',left_on=beijing_aq['station_id'],right_on=aq_station_region['station_id'])
del beijing_aq['station_id_x']
del beijing_aq['station_id_y']
del beijing_aq['key_0']
del beijing_aq['name_chinese']
del beijing_aq['latitude']
del beijing_aq['longitude']
del beijing_aq['district_id']
beijing_aq['station'] = beijing_aq.pop('name_english')

beijing_aq_agg = pd.DataFrame()
for i in beijing_aq['station'].unique():
    data = beijing_aq[beijing_aq['station']==i].reset_index(drop=True)
    del data['station']
    data.columns = ['time',i.split('_')[0]+'_PM2.5',i.split('_')[0]+'_NO2',i.split('_')[0]+'_CO',i.split('_')[0]+'_O3',i.split('_')[0]+'_SO2']
    if i == 'HaiDianBeiBuXinQu':
        beijing_aq_agg = data
    else:
        beijing_aq_agg = pd.merge(beijing_aq_agg,data,how='left',on='time')
print('beijing_aq_agg: ', beijing_aq_agg)

beijing_meo['district_id'] = beijing_meo.pop('id')
beijing_meo = beijing_meo.fillna(method='pad')


beijing_meo = pd.merge(aq_station_region,beijing_meo,how='inner',on='district_id')
del beijing_meo['name_chinese']
del beijing_meo['latitude']
del beijing_meo['longitude']
del beijing_meo['district_id']
del beijing_meo['station_id']
del beijing_meo['weather']
beijing_meo['station'] = beijing_meo.pop('name_english')
beijing_meo.to_csv('../Data/bjmeo.csv',index=0)


flag = 1
beijing_meo_agg = pd.DataFrame()
for i in beijing_meo['station'].unique():
    data = beijing_meo[beijing_meo['station']==i].reset_index(drop=True)
    del data['station']
    data.columns = ['time',i+'_temperature',i+'_pressure',i+'_humidity',i+'_wind_speed/kph',i+'_wind_direction']
    if flag==1:
        flag = 0
        beijing_meo_agg = data
    else:
        beijing_meo_agg = pd.merge(beijing_meo_agg,data,how='left',on='time')

beijing_meo_agg.to_csv('../Data/bj.csv',index=0)
print('beijing_meo_agg: ', beijing_meo_agg)

beijing_aqmeo_agg = pd.merge(beijing_aq_agg,beijing_meo_agg,how='left',on='time')
beijing_aqmeo_agg = beijing_aqmeo_agg.fillna( beijing_aqmeo_agg.median())
beijing_aqmeo_agg['utc_time'] = beijing_aqmeo_agg.pop('time')
beijing_aqmeo_agg.isnull().sum().sum()
beijing_aqmeo_agg.to_csv('../Data/beijing_aq_meo_agg.csv',index=0)
print(beijing_aqmeo_agg)
beijing_aqmeo_agg1 = beijing_aqmeo_agg[:6177]
beijing_aqmeo_agg2 = beijing_aqmeo_agg[6177:6912]
beijing_aqmeo_agg3 = beijing_aqmeo_agg[6912:]
beijing_aqmeo_agg1.to_csv('../Data/beijing_aq_meo_agg1.csv',index=0)
beijing_aqmeo_agg2.to_csv('../Data/beijing_aq_meo_agg2.csv',index=0)
beijing_aqmeo_agg3.to_csv('../Data/beijing_aq_meo_agg3.csv',index=0)
print('beijing_aqmeo_agg1: ', beijing_aqmeo_agg1)
print('beijing_aqmeo_agg2: ', beijing_aqmeo_agg2)
print('beijing_aqmeo_agg3: ', beijing_aqmeo_agg3)
