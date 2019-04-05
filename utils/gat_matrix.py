import math
import numpy as np
import pandas as pd

df = pd.read_csv('../data/beijing_aq_meo_agg.csv')
df1 = pd.read_csv('../data/region.csv')
df2 = df.head(1)


def getDistance(lng1, lat1, lng2, lat2):
    EARTH_RADIUS = 6378.137

    lngA = (lng1 * math.pi / 180.0)
    latA = (lat1 * math.pi / 180.0)
    lngB = (lng2 * math.pi / 180.0)
    latB = (lat2 * math.pi / 180.0)

    a = latA - latB
    b = lngA - lngB
    s = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(latA) * math.cos(latB) * math.pow(math.sin(b / 2), 2)))
    s = s * EARTH_RADIUS
    distanceAB = s * 1000

    return distanceAB


def gen_adj_mx(epsilon=0.5):
    aq_station_region = pd.read_csv("../data/region.csv")
    # print(aq_station_region)
    for index_t in aq_station_region.index:
        print('index_t: ',index_t)
        row_t = aq_station_region.loc[index_t]
        print('row_t: ',row_t)
        long_t = row_t["longitude"]
        lati_t = row_t["latitude"]

        station_name = row_t["station_id"]
        print('station_name: ',station_name)
        all_dis = []
        for index in aq_station_region.index:
            print('index: ',index)
            row = aq_station_region.loc[index]
            long = row['longitude']
            lati = row['latitude']
            dis = getDistance(long, lati, long_t, lati_t)
            all_dis.append(dis)
            print('all_dis: ',all_dis)
        aq_station_region[station_name] = all_dis

    distance_matrix = aq_station_region.drop(['longitude', 'latitude', 'location_type', 'category'], axis=1)
    distance_matrix.set_index(['station_id'], inplace=True)
    distances = distance_matrix.values.flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(distance_matrix / std))
    adj_mx[adj_mx < epsilon] = 0
    # adj_mx[adj_mx == 1] = 0

    return adj_mx

print('gen_adj_mx(epsilon=0.5): ', gen_adj_mx(epsilon=0.5))