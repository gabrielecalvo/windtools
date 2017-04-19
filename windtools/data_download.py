import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from urllib.error import URLError

from .util import download_file, get_path, timer

#todo: using flags in ncdc data to filter
#todo: extra filter for bad data (999s etc)

STATION_LIST_PATH = get_path(__file__, 'isd-history.txt')
DOWNLOAD_DATA_SUBFOLDER_NAME = 'downloaded_data'


class NCDC_Downloader():
    def __init__(self, lat=None, long=None, radius_km=50, date_from=None, date_to=None):
        self.latitude = lat
        self.longitude = long
        self.distance = radius_km
        self.date_from = datetime(*date_from) if isinstance(date_from, tuple) else date_from
        self.date_to = datetime(*date_to) if isinstance(date_to, tuple) else date_to
        self.daterange_tollerance = 0  # days
        self.station_id_list = []
        self.station_info = None
        self.station_data = None

    @staticmethod
    def update_station_list():
        url = 'ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.txt'
        download_file(url=url, save_to=STATION_LIST_PATH)

    @staticmethod
    def _convert_between_id_and_usaf_wban(id=None, usaf=None, wban=None):
        if id:
            usaf, wban = [float(i) for i in id.split('-')]
            return usaf, wban
        else:
            id = '{:0>6.0f}-{:0>5.0f}'.format(usaf, wban)
            return id

    @staticmethod
    def _load_stations_metadata():
        colspecs = [(0, 7), (7, 13), (13, 43), (43, 48), (48, 51), (51, 57),
                    (57, 65), (65, 74), (74, 82), (82, 91), (91, 999)]
        df = pd.read_fwf(STATION_LIST_PATH, skiprows=range(20), colspecs=colspecs, parse_dates=[9, 10])
        # df.index = [self._convert_between_id_and_usaf_wban(usaf=r['USAF'], wban=r['WBAN']) for i, r in df.iterrows()]
        return df

    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points on the earth (specified in decimal degrees).
        Inputs in degrees. The result is in km.
        """
        # todo speedup with pd.map
        # convert decimal degrees to radians
        lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(np.radians, [lon1, lat1, lon2, lat2])
        # haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.power(np.sin(dlat / 2), 2) + np.cos(lat1_rad) * np.cos(lat2_rad) * np.power(np.sin(dlon / 2), 2)
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c
        return km

    @staticmethod
    def rh_from_dew_temperature(t, t_dew, simple=False):
        if simple:
            rh = 5*(t_dew-t)+100
        else:
            T = np.add(t, 273.15)
            Td = np.add(t_dew, 273.15)
            L_Rv = 5423
            rh = np.exp(L_Rv*(1/T-1/Td))*100
        return rh

    def find_stations_nearby(self):
        assert self.latitude and self.longitude and self.distance
        stations = self._load_stations_metadata()
        stations = stations[pd.notnull(stations['LAT']) & pd.notnull(stations['LON'])]

        # calculate distance from target
        apply_haversine = lambda r: self.haversine(lon1=self.longitude, lat1=self.latitude,
                                                   lon2=r['LON'], lat2=r['LAT'])
        stations['Distance'] = stations.apply(apply_haversine, axis=1)

        # filter for valid stations
        distance_mask = stations['Distance'] <= self.distance
        timerange_mask = [True] * len(stations)
        if self.date_from:
            timerange_mask &= stations['BEGIN'] <= self.date_from + timedelta(days=self.daterange_tollerance)
        if self.date_to:
            timerange_mask &= stations['END'] >= self.date_to - timedelta(days=self.daterange_tollerance)
        stations_nearby = stations[distance_mask & timerange_mask]

        # adding ids as index
        get_ids = lambda r: self._convert_between_id_and_usaf_wban(usaf=r['USAF'], wban=r['WBAN'])
        id_list = stations_nearby.apply(get_ids, axis=1).tolist()
        stations_nearby.index = id_list

        # storing valid stations
        self.station_info = stations_nearby
        self.station_id_list = id_list
        return self.station_id_list

    def download_ncdc_station_data(self, station_id, date_from=None, date_to=None):
        # defining data range
        if date_from is None or date_to is None:
            info = self.station_info.ix[station_id, :]
            date_from = info['BEGIN'] if date_from is None else date_from
            date_to = info['END'] if date_to is None else date_to

        # downloading data
        downloaded_file_list = []
        for year in range(date_from.year, date_to.year + 1):
            file_name = '{id}-{year}.gz'.format(id=station_id, year=year)
            url = 'ftp://ftp.ncdc.noaa.gov/pub/data/noaa/{year}/{file_name}'.format(year=year, file_name=file_name)
            save_to = os.path.join(DOWNLOAD_DATA_SUBFOLDER_NAME, file_name)
            try:
                download_file(url=url, save_to=save_to)
                downloaded_file_list.append(save_to)
            except URLError:
                print('!!! File {} not found. download skipped. !!!'.format(file_name))
        return downloaded_file_list

    def _parse_ncdc_data(self, fpath):
        header_names = ['total', 'USAF', 'WBAN', 'datetime', 'source', 'latitude', 'longitude', 'report_type',
                        'elevation',
                        'call_letter_id', 'quality_control', 'direction', 'direction_quality', 'observation',
                        'speed_times_10',
                        'speed_quality', 'sky', 'sky_quality', 'sky_determination', 'sky_cavok_code', 'visibility',
                        'visibility_quality', 'visibility_variability', 'visibility_variability_quality', 'temperature',
                        'temperature_quality', 'temperature_dew', 'temperature_dew_quality',
                        'pressure_sea_level', 'pressure_quality']
        colspecs = [(0, 4), (4, 10), (10, 15), (15, 27), (27, 28), (28, 34), (34, 41), (41, 46), (46, 51),
                    (51, 56), (56, 60), (60, 63), (63, 64), (64, 65), (65, 69), (69, 70), (70, 75), (75, 76), (76, 77),
                    (77, 78), (78, 84), (84, 85), (85, 86), (86, 87), (87, 92), (92, 93), (93, 98), (98, 99), (99, 104),
                    (104, 105)]
        compression = 'gzip' if fpath.endswith('.gz') else 'infer'
        df = pd.read_fwf(fpath, colspecs=colspecs, header=None, names=header_names, index_col=3,
                         parse_dates=True, compression=compression)
        return df

    def merge_ncdc_data(self, downloaded_file_list):
        # collating all data into one parsed dataframe
        df_collated = pd.DataFrame()
        for zipped_fpath in downloaded_file_list:
            df_file = self._parse_ncdc_data(zipped_fpath)
            df_collated = pd.concat([df_collated, df_file], axis=0)
        return df_collated

    def export_data_from_selected_stations(self, station_ids, process_data=True):
        df_dict = {}
        for station_id in station_ids:
            raw_station_name = self.station_info.ix[station_id, 'STATION NAME']
            station_name = re.sub(r'[\\\/\*\[\]\:\?]', '', raw_station_name)[:31]
            file_list = self.download_ncdc_station_data(station_id, self.date_from, self.date_to)
            df = self.merge_ncdc_data(file_list)
            if process_data:
                df = self.process_ncdc_data(df)
            df_dict[station_name] = df
        self.station_data = df_dict
        return self.station_data

    def process_ncdc_data(self, df):
        # rescaling of values
        df['windspeed'] = df['speed_times_10']/10
        df['temperature'] /= 10
        df['temperature_dew'] /= 10
        df['pressure_sea_level'] /= 10

        # calculation of humidity
        df['humidity'] = self.rh_from_dew_temperature(df['temperature'], df['temperature_dew'])

        # trimming to useful data columns only
        useful_cols = ['windspeed', 'direction', 'temperature', 'pressure_sea_level', 'humidity']
        return df.loc[:, useful_cols]

    def save_all_downloaded_data(self, excel_name='all_data.xlsx'):
        dfs = self.station_data
        writer = pd.ExcelWriter(os.path.join(DOWNLOAD_DATA_SUBFOLDER_NAME, excel_name))
        for station, data in dfs.items():
            data.to_excel(writer, station)
        writer.save()

    def download_and_save_data(self, station_ids=None, process_data=True, excel_name='all_data.xlsx'):
        station_ids = self.station_id_list if station_ids is None else station_ids
        self.export_data_from_selected_stations(station_ids, process_data=process_data)
        self.save_all_downloaded_data(excel_name=excel_name)
        print('Downloaded data is located here: {}'.format(os.path.abspath(DOWNLOAD_DATA_SUBFOLDER_NAME, excel_name)))

    @classmethod
    def get_all(cls, *args, **kwargs):
        inst = cls(*args, **kwargs)
        inst.find_stations_nearby()
        inst.export_data_from_selected_stations(inst.station_id_list)
        inst.save_all_downloaded_data()

if __name__ == '__main__':
    test_dict = dict(lat=45, long=7.7, radius_km=10, date_from=(2015, 12, 20))

    # with timer('get_all'):
    #     NCDC_downloader.get_all(test_dict)

    downloader = NCDC_Downloader(test_dict)
    nearby_stations = downloader.find_stations_nearby()
    wanted_stations = nearby_stations[0:2]
    downloader.download_and_save_data(wanted_stations)
