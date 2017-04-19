import pytest
from windtools.data_download import NCDC_Downloader, Downloader
from datetime import datetime


def test_ncdc_downloader_integrity_test():
    test_dict = dict(lat=45, long=7.7, radius_km=10, date_from=(2015, 12, 20))
    downloader = NCDC_Downloader(**test_dict)
    nearby_stations = downloader.find_stations_nearby()
    wanted_stations = nearby_stations[0:2]
    #downloader.download_and_save_data(wanted_stations)
    #todo: finish testing


def test_downloader_init():
    d = Downloader(lat=10, long=20, radius_km=30, date_from=(2000, 1, 1), date_to=datetime(2001, 1, 1))
    assert d.latitude == 10
    assert d.longitude == 20
    assert d.distance == 30
    assert d.date_from == datetime(2000, 1, 1)
    assert d.date_to == datetime(2001, 1, 1)
    assert d.station_id_list == []
    assert d.station_info is None
    assert d.station_data is None
