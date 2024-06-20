import datetime
import io
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
from easytorch import launch_runner, Runner

import os
import sys
from STEP.basicts import launch_training
import json

class DateList(BaseModel):
    date: datetime.datetime


import torch
from torchvision.transforms import transforms


# class StepInfrence:

#     def __init__(self):
#         self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#         print(f"Loading model for device {self.device}")
#         self.model = STEP()
#         self.model.load_state_dict(torch.load("mnist_cnn.pt"))
#         self.model = self.model.eval()
#         self.model = self.model.to(self.device)

#     def infer(self, image_data):
#         preprocessed_image_data = self._preprocess(image_data)
#         prediction = self._predict(preprocessed_image_data)

#         return prediction

#     def _preprocess(self, image_data):
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Grayscale(),
#             transforms.Resize(28),
#             transforms.CenterCrop(28),
#             transforms.Normalize((0.1307,), (0.3081,)),
#         ])

#         tensor = transform(image_data)

#         return torch.unsqueeze(tensor, dim=0)

#     def _predict(self, image_data):
#         with torch.inference_mode():
#             data = image_data.to(self.device)
#             output = self.model(data)
#             pred = output.argmax(dim=1, keepdim=True)
#             return pred.item()

# Initialize the FastAPI app
app = FastAPI()
print(sys.path)
# def get_context_timeseries(date):
#     context = pd.read_csv('data/context.csv')
#     context['date'] = pd.to_datetime(context['date'])
#     reset_index = context.reset_index()
#     date_index = reset_index[reset_index['index'] == date].index[0]
#     window_len = 2016
#     short_window_len = 12
#     future_window_len = 12
#     context_timeseries = context.iloc[date_index-window_len:date_index]
#     short_timeseries = context.iloc[date_index-short_window_len:date_index]
#     future_timeseries = context.iloc[date_index:date_index+future_window_len]
#     context_timeseries, short_timeseries, future_timeseries = [torch.tensor(df.values).to('cuda:1') for df in [context_timeseries, short_timeseries, future_timeseries]]
    
#     return context

# Load the PyTorch model


# Define the path operation function for handling image uploads and predictions
@app.post("/predict")
async def predict(dates: DateList):
    date: datetime.datetime = dates.date
    try:
        result = launch_training("STEP/step/STEP_Bru.py", '0, 1', inference=True, date_inference=date)
        res = {i: result[i] if not isinstance(result[i], torch.Tensor) else result[i].cpu().numpy().tolist()
            for i in range(len(result))}
        # Return the predictions
        return res[0], res[2]
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/sensors")
async def get_sensors():
    try:
        sensors = pd.read_pickle("/home/seyed/PycharmProjects/step/STEP/datasets/Bru/columns_index_data_dataset.pkl")
        return sensors.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))