# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import gc

pm25 = ['utc_time','HaiDianBeiBuXinQu_PM2.5','HaiDianBeiJingZhiWuYuan_PM2.5','ShiJingShanGuCheng_PM2.5',
        'FengTaiYunGang_PM2.5','FangShanLiangXiang_PM2.5','HaiDianWanLiu_PM2.5',
        'ChaoYangAoTiZhongXin_PM2.5','XiZhiMenBeiDaJie_PM2.5','XiChengWanShouXiGong_PM2.5',
        'YongDingMenNeiDaJie_PM2.5','NanSanHuanXiLu_PM2.5','ChaoYangNongZhanGuan_PM2.5',
        'DongChengDongSi_PM2.5','DongSiHuanBeiLu_PM2.5','XiChengGuanYuan_PM2.5',
        'FengTaiHuaYuan_PM2.5','DaXingHuangCunZhen_PM2.5','YiZhuangKaiFaQu_PM2.5',
        'TongZhouXinCheng_PM2.5','DongChengTianTan_PM2.5','QianMenDongDaJie_PM2.5',
        'BeiJingMeiGuoDaShiGuan_PM2.5','ShunYiXinCheng_PM2.5','ChangPingZhen_PM2.5',
        'MenTouGouLongQuanZhen_PM2.5','PingGuZhen_PM2.5','HuaiRouZhen_PM2.5',
        'MiYunZhen_PM2.5','YanQingZhen_PM2.5','ChangPingDingLing_PM2.5',
        'JingXiBeiBaDaLing_PM2.5','JingDongBeiMiYunShuiKu_PM2.5','JingDongDongGaoCun_PM2.5',
        'JingDongNanYongLeDian_PM2.5','JingNanYu_PM2.5','JingXiNanLiuLiHe_PM2.5']
pm10 = ['utc_time','aotizhongxin_PM10', 'badaling_PM10', 'beibuxinqu_PM10', 'daxing_PM10', 'dingling_PM10', 'donggaocun_PM10',
        'dongsi_PM10', 'dongsihuan_PM10', 'fangshan_PM10', 'fengtaihuayuan_PM10', 'guanyuan_PM10', 'gucheng_PM10',
        'huairou_PM10', 'liulihe_PM10', 'mentougou_PM10', 'miyun_PM10', 'miyunshuiku_PM10', 'nansanhuan_PM10',
        'nongzhanguan_PM10', 'pingchang_PM10', 'pinggu_PM10', 'qianmen_PM10', 'shunyi_PM10', 'tiantan_PM10',
        'tongzhou_PM10', 'wanliu_PM10', 'wanshouxigong_PM10', 'xizhimenbei_PM10', 'yanqin_PM10', 'yizhuang_PM10',
        'yongdingmennei_PM10', 'yongledian_PM10', 'yufa_PM10', 'yungang_PM10', 'zhiwuyuan_PM10']
o3 = ['utc_time','aotizhongxin_O3', 'badaling_O3', 'beibuxinqu_O3', 'daxing_O3', 'dingling_O3', 'donggaocun_O3', 'dongsi_O3',
      'dongsihuan_O3', 'fangshan_O3', 'fengtaihuayuan_O3', 'guanyuan_O3', 'gucheng_O3', 'huairou_O3', 'liulihe_O3',
      'mentougou_O3', 'miyun_O3', 'miyunshuiku_O3', 'nansanhuan_O3', 'nongzhanguan_O3', 'pingchang_O3', 'pinggu_O3',
      'qianmen_O3', 'shunyi_O3', 'tiantan_O3', 'tongzhou_O3', 'wanliu_O3', 'wanshouxigong_O3', 'xizhimenbei_O3',
      'yanqin_O3', 'yizhuang_O3', 'yongdingmennei_O3', 'yongledian_O3', 'yufa_O3', 'yungang_O3', 'zhiwuyuan_O3']
co = ['utc_time','aotizhongxin_CO', 'badaling_CO', 'beibuxinqu_CO', 'daxing_CO', 'dingling_CO', 'donggaocun_CO', 'dongsi_CO',
      'dongsihuan_CO', 'fangshan_CO', 'fengtaihuayuan_CO', 'guanyuan_CO', 'gucheng_CO', 'huairou_CO', 'liulihe_CO',
      'mentougou_CO', 'miyun_CO', 'miyunshuiku_CO', 'nansanhuan_CO', 'nongzhanguan_CO', 'pingchang_CO', 'pinggu_CO',
      'qianmen_CO', 'shunyi_CO', 'tiantan_CO', 'tongzhou_CO', 'wanliu_CO', 'wanshouxigong_CO', 'xizhimenbei_CO',
      'yanqin_CO', 'yizhuang_CO', 'yongdingmennei_CO', 'yongledian_CO', 'yufa_CO', 'yungang_CO', 'zhiwuyuan_CO']
so2 = ['utc_time','aotizhongxin_SO2', 'badaling_SO2', 'beibuxinqu_SO2', 'daxing_SO2', 'dingling_SO2', 'donggaocun_SO2',
       'dongsi_SO2', 'dongsihuan_SO2', 'fangshan_SO2', 'fengtaihuayuan_SO2', 'guanyuan_SO2', 'gucheng_SO2',
       'huairou_SO2', 'liulihe_SO2', 'mentougou_SO2', 'miyun_SO2', 'miyunshuiku_SO2', 'nansanhuan_SO2',
       'nongzhanguan_SO2', 'pingchang_SO2', 'pinggu_SO2', 'qianmen_SO2', 'shunyi_SO2', 'tiantan_SO2',
       'tongzhou_SO2', 'wanliu_SO2', 'wanshouxigong_SO2', 'xizhimenbei_SO2', 'yanqin_SO2', 'yizhuang_SO2',
       'yongdingmennei_SO2', 'yongledian_SO2', 'yufa_SO2', 'yungang_SO2', 'zhiwuyuan_SO2']
no2 = ['utc_time','aotizhongxin_NO2', 'badaling_NO2', 'beibuxinqu_NO2', 'daxing_NO2', 'dingling_NO2', 'donggaocun_NO2',
       'dongsi_NO2', 'dongsihuan_NO2', 'fangshan_NO2', 'fengtaihuayuan_NO2', 'guanyuan_NO2', 'gucheng_NO2',
       'huairou_NO2', 'liulihe_NO2', 'mentougou_NO2', 'miyun_NO2', 'miyunshuiku_NO2', 'nansanhuan_NO2',
       'nongzhanguan_NO2', 'pingchang_NO2', 'pinggu_NO2', 'qianmen_NO2', 'shunyi_NO2', 'tiantan_NO2',
       'tongzhou_NO2', 'wanliu_NO2', 'wanshouxigong_NO2', 'xizhimenbei_NO2', 'yanqin_NO2', 'yizhuang_NO2',
       'yongdingmennei_NO2', 'yongledian_NO2', 'yufa_NO2', 'yungang_NO2', 'zhiwuyuan_NO2']
temperature = ['HaiDianBeiBuXinQu_temperature','HaiDianBeiJingZhiWuYuan_temperature','ShiJingShanGuCheng_temperature',
               'FengTaiYunGang_temperature','FangShanLiangXiang_temperature','HaiDianWanLiu_temperature',
               'ChaoYangAoTiZhongXin_temperature','XiZhiMenBeiDaJie_temperature','XiChengWanShouXiGong_temperature',
               'YongDingMenNeiDaJie_temperature','NanSanHuanXiLu_temperature','ChaoYangNongZhanGuan_temperature',
               'DongChengDongSi_temperature','DongSiHuanBeiLu_temperature','XiChengGuanYuan_temperature',
               'FengTaiHuaYuan_temperature','DaXingHuangCunZhen_temperature','YiZhuangKaiFaQu_temperature',
               'TongZhouXinCheng_temperature','DongChengTianTan_temperature','QianMenDongDaJie_temperature',
               'BeiJingMeiGuoDaShiGuan_temperature','ShunYiXinCheng_temperature','ChangPingZhen_temperature',
               'MenTouGouLongQuanZhen_temperature','PingGuZhen_temperature','HuaiRouZhen_temperature',
               'MiYunZhen_temperature','YanQingZhen_temperature','ChangPingDingLing_temperature',
               'JingXiBeiBaDaLing_temperature','JingDongBeiMiYunShuiKu_temperature','JingDongDongGaoCun_temperature',
               'JingDongNanYongLeDian_temperature','JingNanYu_temperature','JingXiNanLiuLiHe_temperature']
pressure = ['HaiDianBeiBuXinQu_pressure','HaiDianBeiJingZhiWuYuan_pressure','ShiJingShanGuCheng_pressure',
            'FengTaiYunGang_pressure','FangShanLiangXiang_pressure','HaiDianWanLiu_pressure',
            'ChaoYangAoTiZhongXin_pressure','XiZhiMenBeiDaJie_pressure','XiChengWanShouXiGong_pressure',
            'YongDingMenNeiDaJie_pressure','NanSanHuanXiLu_pressure','ChaoYangNongZhanGuan_pressure',
            'DongChengDongSi_pressure','DongSiHuanBeiLu_pressure','XiChengGuanYuan_pressure',
            'FengTaiHuaYuan_pressure','DaXingHuangCunZhen_pressure','YiZhuangKaiFaQu_pressure',
            'TongZhouXinCheng_pressure','DongChengTianTan_pressure','QianMenDongDaJie_pressure',
            'BeiJingMeiGuoDaShiGuan_pressure','ShunYiXinCheng_pressure','ChangPingZhen_pressure',
            'MenTouGouLongQuanZhen_pressure','PingGuZhen_pressure','HuaiRouZhen_pressure',
            'MiYunZhen_pressure','YanQingZhen_pressure','ChangPingDingLing_pressure',
            'JingXiBeiBaDaLing_pressure','JingDongBeiMiYunShuiKu_pressure','JingDongDongGaoCun_pressure',
            'JingDongNanYongLeDian_pressure','JingNanYu_pressure','JingXiNanLiuLiHe_pressure']
humidity = ['HaiDianBeiBuXinQu_humidity','HaiDianBeiJingZhiWuYuan_humidity','ShiJingShanGuCheng_humidity',
            'FengTaiYunGang_humidity','FangShanLiangXiang_humidity','HaiDianWanLiu_humidity',
            'ChaoYangAoTiZhongXin_humidity','XiZhiMenBeiDaJie_humidity','XiChengWanShouXiGong_humidity',
            'YongDingMenNeiDaJie_humidity','NanSanHuanXiLu_humidity','ChaoYangNongZhanGuan_humidity',
            'DongChengDongSi_humidity','DongSiHuanBeiLu_humidity','XiChengGuanYuan_humidity',
            'FengTaiHuaYuan_humidity','DaXingHuangCunZhen_humidity','YiZhuangKaiFaQu_humidity',
            'TongZhouXinCheng_humidity','DongChengTianTan_humidity','QianMenDongDaJie_humidity',
            'BeiJingMeiGuoDaShiGuan_humidity','ShunYiXinCheng_humidity','ChangPingZhen_humidity',
            'MenTouGouLongQuanZhen_humidity','PingGuZhen_humidity','HuaiRouZhen_humidity',
            'MiYunZhen_humidity','YanQingZhen_humidity','ChangPingDingLing_humidity',
            'JingXiBeiBaDaLing_humidity','JingDongBeiMiYunShuiKu_humidity','JingDongDongGaoCun_humidity',
            'JingDongNanYongLeDian_humidity','JingNanYu_humidity','JingXiNanLiuLiHe_humidity']
wind_direction = ['HaiDianBeiBuXinQu_wind_direction','HaiDianBeiJingZhiWuYuan_wind_direction','ShiJingShanGuCheng_wind_direction',
                  'FengTaiYunGang_wind_direction','FangShanLiangXiang_wind_direction','HaiDianWanLiu_wind_direction',
                  'ChaoYangAoTiZhongXin_wind_direction','XiZhiMenBeiDaJie_wind_direction','XiChengWanShouXiGong_wind_direction',
                  'YongDingMenNeiDaJie_wind_direction','NanSanHuanXiLu_wind_direction','ChaoYangNongZhanGuan_wind_direction',
                  'DongChengDongSi_wind_direction','DongSiHuanBeiLu_wind_direction','XiChengGuanYuan_wind_direction',
                  'FengTaiHuaYuan_wind_direction','DaXingHuangCunZhen_wind_direction','YiZhuangKaiFaQu_wind_direction',
                  'TongZhouXinCheng_wind_direction','DongChengTianTan_wind_direction','QianMenDongDaJie_wind_direction',
                  'BeiJingMeiGuoDaShiGuan_wind_direction','ShunYiXinCheng_wind_direction','ChangPingZhen_wind_direction',
                  'MenTouGouLongQuanZhen_wind_direction','PingGuZhen_wind_direction','HuaiRouZhen_wind_direction',
                  'MiYunZhen_wind_direction','YanQingZhen_wind_direction','ChangPingDingLing_wind_direction',
                  'JingXiBeiBaDaLing_wind_direction','JingDongBeiMiYunShuiKu_wind_direction','JingDongDongGaoCun_wind_direction',
                  'JingDongNanYongLeDian_wind_direction','JingNanYu_wind_direction','JingXiNanLiuLiHe_wind_direction']
wind_speed = ['HaiDianBeiBuXinQu_wind_speed/kph','HaiDianBeiJingZhiWuYuan_wind_speed/kph','ShiJingShanGuCheng_wind_speed/kph',
              'FengTaiYunGang_wind_speed/kph','FangShanLiangXiang_wind_speed/kph','HaiDianWanLiu_wind_speed/kph',
              'ChaoYangAoTiZhongXin_wind_speed/kph','XiZhiMenBeiDaJie_wind_speed/kph','XiChengWanShouXiGong_wind_speed/kph',
              'YongDingMenNeiDaJie_wind_speed/kph','NanSanHuanXiLu_wind_speed/kph','ChaoYangNongZhanGuan_wind_speed/kph',
              'DongChengDongSi_wind_speed/kph','DongSiHuanBeiLu_wind_speed/kph','XiChengGuanYuan_wind_speed/kph',
              'FengTaiHuaYuan_wind_speed/kph','DaXingHuangCunZhen_wind_speed/kph','YiZhuangKaiFaQu_wind_speed/kph',
              'TongZhouXinCheng_wind_speed/kph','DongChengTianTan_wind_speed/kph','QianMenDongDaJie_wind_speed/kph',
              'BeiJingMeiGuoDaShiGuan_wind_speed/kph','ShunYiXinCheng_wind_speed/kph','ChangPingZhen_wind_speed/kph',
              'MenTouGouLongQuanZhen_wind_speed/kph','PingGuZhen_wind_speed/kph','HuaiRouZhen_wind_speed/kph',
              'MiYunZhen_wind_speed/kph','YanQingZhen_wind_speed/kph','ChangPingDingLing_wind_speed/kph',
              'JingXiBeiBaDaLing_wind_speed/kph','JingDongBeiMiYunShuiKu_wind_speed/kph','JingDongDongGaoCun_wind_speed/kph',
              'JingDongNanYongLeDian_wind_speed/kph','JingNanYu_wind_speed/kph','JingXiNanLiuLiHe_wind_speed/kph']


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    df_temperature = df[temperature]
    df_pressure = df[pressure]
    df_humidity = df[humidity]
    df_wind_direction = df[wind_direction]
    df_wind_speed = df[wind_speed]
    df = df[pm25]
    df.set_index(['utc_time'], inplace=True)
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_temperature = np.expand_dims(df_temperature.values,axis=-1)
    data_pressure = np.expand_dims(df_pressure.values, axis=-1)
    data_humidity = np.expand_dims(df_humidity.values, axis=-1)
    data_wind_direction = np.expand_dims(df_wind_direction.values, axis=-1)
    data_wind_speed = np.expand_dims(df_wind_speed.values, axis=-1)
    data_list = [data]
    data_list.append(data_temperature)
    data_list.append(data_pressure)
    data_list.append(data_humidity)
    data_list.append(data_wind_direction)
    data_list.append(data_wind_speed)
    # print('data_list: ', data_list)
    if add_time_in_day:    #
        time_ind = (df.index.values.astype("datetime64[h]") - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)
    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))  # 11
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive 34249-13
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):               # 数据格式转换成（num_samples, input_length, num_nodes, input_dim）

    # Data preparation
    df = pd.read_csv('../Data/beijing_aq_meo_agg1.csv')
    df_val = pd.read_csv('../Data/beijing_aq_meo_agg2.csv')
    df_test = pd.read_csv('../Data/beijing_aq_meo_agg3.csv')
    # 0 is the latest observed sample. 输入长度
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-71, 1, 1),))
    )
    # Predict the next one hour 输出长度
    y_offsets = np.sort(np.arange(1, 49, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    # train
    x_train, y_train = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )
    # val
    x_val, y_val = generate_graph_seq2seq_io_data(
        df_val,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )
    print('y_val: ', y_val)
    # test
    x_test, y_test = generate_graph_seq2seq_io_data(
        df_test,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x val: ", x_val, ", y_val: ", y_val)


    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y: ", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

def main(args):
    print("Generating training data")          # 划分数据集，将原始数据转换成图数据格式
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()             # 创建命令行与参数解析对象
    parser.add_argument(
        "--output_dir", type=str, default="../Data/beijing/", help="Output directory."
    )                                              # 添加参数
    parser.add_argument(
        "--air_df_filename",
        type=str,
        default="../Data/beijing_aq_meo_agg.csv",
        help="Raw air readings.",
    )
    args = parser.parse_args()                     # 解析参数
    main(args)
