import torch
import torch.nn as nn
from typing import List, Optional
from gluonts.model.forecast_generator import ForecastGenerator, SampleForecastGenerator
from gluonts.transform import Transformation
from gluonts.torch.model.predictor import OutputTransform

from gluonts.torch.model.estimator import PyTorchLightningEstimator, PyTorchPredictor, TrainOutput


class LlamaEstimator(PyTorchLightningEstimator):
    def __init__(
        self,
        prediction_length: int,
        freq: str,
        input_names: List[str],
        prediction_net: nn.Module,
        batch_size: int,
        input_transform: Transformation,
        forecast_generator: ForecastGenerator = SampleForecastGenerator(),
        output_transform: Optional[OutputTransform] = None,
        lead_time: int = 0,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__(prediction_length, freq, lead_time=lead_time)
        self.input_names = input_names
        self.prediction_net = prediction_net.to(device)
        self.batch_size = batch_size
        self.input_transform = input_transform
        self.forecast_generator = forecast_generator
        self.output_transform = output_transform
        self.device = device
        self.required_fields = ["forecast_start", "item_id", "info"]

    def create_predictor(
        self, transformation: Transformation, trained_network: nn.Module
    ) -> PyTorchPredictor:
        return PyTorchPredictor(
            input_names=self.input_names,
            prediction_net=trained_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            