import json
from pathlib import Path

import aiofiles
from aiohttp import ClientSession


class AsyncRequest:
    OUTPUT_RESULT_DIR = '/home/seyed/PycharmProjects/step/STEP/weather_data/'

    async def get_url(self, lat, lang):
        return (f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lang}&'
                'start_date=2012-03-01&end_date=2012-07-31&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,'
                'apparent_temperature,precipitation,rain,snowfall,weathercode,pressure_msl,surface_pressure,'
                'cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,'
                'et0_fao_evapotranspiration,vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m'
                ',winddirection_100m,windgusts_10m,soil_temperature_0_to_7cm,soil_temperature_7_to_28cm,'
                'soil_temperature_28_to_100cm,soil_temperature_100_to_255cm,soil_moisture_0_to_7cm,'
                'soil_moisture_7_to_28cm,soil_moisture_28_to_100cm,soil_moisture_100_to_255cm&daily=temperature_2m_max'
                ',temperature_2m_min,apparent_temperature_max,apparent_temperature_min,apparent_temperature_mean,'
                'precipitation_sum,rain_sum,snowfall_sum,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant'
                '&timeformat=unixtime&timezone=GMT&format=json')

    async def get_headers(self):
        return {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/116.0',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://open-meteo.com/',
            'Origin': 'https://open-meteo.com',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site'
        }

    async def get_historical_data(self, lat, lang, city_id):
        async with ClientSession() as session:
            print(f"Start requesting for city id {city_id}.....")
            async with session.get(url=(await self.get_url(lat, lang)), headers=(await self.get_headers())) as request:
                request.raise_for_status()
                response = await request.json()
                print("response is:", type(response), "[...]")
                path = Path(self.OUTPUT_RESULT_DIR) / str(city_id)
                path.mkdir(parents=True, exist_ok=True)
                file = path / f"{city_id}.json"
                async with aiofiles.open(file.as_posix(), "w") as fp:
                    await fp.write(json.dumps(response))
        return
