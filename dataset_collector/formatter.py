import numpy as np
from abc import abstractmethod, abstractproperty
class BaseFormatter:

    def __init__(self, raw_data: dict, meta_data: dict):
        self.raw_data = raw_data
        self.meta_data = meta_data

    def get_raw_data(self):
        return self.raw_data

    @abstractmethod
    def get_dataset_columns(self):
        raise NotImplemented

    @abstractproperty
    def label_mapping(self):
        raise NotImplemented

class ShiftDatasetFormatter(BaseFormatter):

    @staticmethod
    def get_u_component(tetha, v_speed):
        """
        cmc_0_2_2_10: uv∣V→∣=−∣V→∣sinϕ=−∣V→∣cosϕ=u2+v2−−−−−−√
        :param tetha:
        :return:
        """
        return -1 * v_speed * np.sin(tetha)

    @staticmethod
    def get_v_component(tetha, v_speed):
        """
        cmc_0_2_2_10: uv∣V→∣=−∣V→∣sinϕ=−∣V→∣cosϕ=u2+v2−−−−−−√
        :param tetha:
        :return:
        """
        return -1 * v_speed * np.cos(tetha)

    def get_dataset_columns_adaptor(self):
        return {
            "fact_time": self.raw_data["dt"],
            "fact_latitude": self.meta_data["fact_latitude"],
            "fact_longitude": self.meta_data["fact_longitude"],
            "fact_temperature": self.raw_data["main"]["temp"],
            "fact_cwsm_class": self.raw_data["weather"]["id"],
            "climate": self.raw_data["weather"]["main"],
            "cmc_0_3_1_0": self.raw_data["main"]["sea_level"],
            "cmc_0_3_0_0": self.raw_data["main"]["grnd_level"],
            "gfs_wind_speed": self.raw_data["wind"]["speed"],
            "climate_pressure": self.raw_data["main"]["sea_level"],
            "cmc_0_2_2_10": self.get_u_component(
                tetha=self.raw_data["wind"]["deg"],
                v_speed=self.raw_data["wind"]["speed"]
            ),
            "cmc_0_2_3_10": self.get_v_component(
                tetha=self.raw_data["wind"]["deg"],
                v_speed=self.raw_data["wind"]["speed"]
            ),
            "cmc_0_6_1_0": self.raw_data["clouds"]["all"]
        }