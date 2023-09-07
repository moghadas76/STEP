import json
import asyncio
import networkx
from ast import literal_eval
from typing import Tuple, List
from asyncio import get_event_loop
from STEP.dataset_collector.request_util import AsyncRequest

def process_nodes(nodes: dict):
    city_ids = []
    cords= []
    for city_id, value in nodes.items():
        city_ids.append(int(city_id))
        cords.append(literal_eval(value["label"]))
    return city_ids, cords

async def main(cordinates: List[Tuple[float]], city_ids: List[int]):
    futures = [
        asyncio.ensure_future(AsyncRequest().get_historical_data(lat, lang, city_id), loop=get_event_loop())
        for (lat, lang), city_id in zip(cordinates, city_ids)
    ]
    results = await asyncio.gather(*futures)
    return results


if __name__ == '__main__':
    nodes_dump = None
    city_ids, cords = None, None
    with open("/home/seyed/PycharmProjects/step/STEP/checkpoints/labels.json", "r") as file:
        nodes_dump = json.load(file)
    city_ids, cords = process_nodes(nodes_dump)
    print("city_ids, cords", city_ids[:5], cords[:5])
    asyncio.run(main(cords, city_ids))

