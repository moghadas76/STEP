import json
import torch
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, AnyStr
from dataclasses import dataclass
from scripts.var_model.dcrnn import load_var_model
from step.step_arch.step import STEP
app = FastAPI()

def _torch_initializer():
    return {
    "dataset_name": "Brussels",
    "pre_trained_tsformer_path": "tsformer_ckpt/TSFormer_Brussels.pt",
    "tsformer_args": {
                    "patch_size":12,
                    "in_channel":1,
                    "embed_dim":96,
                    "num_heads":4,
                    "mlp_ratio":4,
                    "dropout":0.1,
                    "num_token":288 * 7 / 12,
                    "mask_ratio":0.75,
                    "encoder_depth":4,
                    "decoder_depth":1,
                    "mode":"forecasting"
    },
    "backend_args": {
                    "num_nodes" : 269,
                    "support_len" : 2,
                    "dropout"   : 0.3,
                    "gcn_bool"  : True,
                    "addaptadj" : True,
                    "aptinit"   : None,
                    "in_dim"    : 2,
                    "out_dim"   : 12,
                    "residual_channels" : 32,
                    "dilation_channels" : 32,
                    "skip_channels"     : 256,
                    "end_channels"      : 512,
                    "kernel_size"       : 2,
                    "blocks"            : 4,
                    "layers"            : 2
    },
    "dgl_args": {
                "dataset_name": "Brussels",
                "k": 10,
                "input_seq_len": 12,
                "output_seq_len": 12
    }
}

def _get_var_model(dataframe,):

    return load_var_model("Brussels")


@dataclass
class ModelMeta:
    instance: object
    is_torch: bool

def _get_model(model_name):

    return {
        # "var": ModelMeta(instance=, is_torch=False),
        "transformer": ModelMeta(instance=STEP(**_torch_initializer()), is_torch=True),
    }[model_name]

def inference(model, payload):
    input_to_model = torch.tensor(payload["input"], dtype=torch.float32)
    with torch.no_grad():
        input_to_model = input_to_model.unsqueeze(0)
        output = model(input_to_model)
    np_output = output.numpy()
    output_as_pandas = pd.DataFrame(np_output)
    json.loads(output_as_pandas.to_json(orient="records"))


class TimeseriesPayload(BaseModel):
    history: list

class TimeseriesResponse(BaseModel):
    chart: dict
    columns: List[AnyStr]

# Endpoint that uses the dependency
@app.get("/predict/transformer", response_model=TimeseriesResponse)
async def predict_transformer(body: TimeseriesPayload):
    response = inference(_get_model(), body)
    return TimeseriesResponse(chart=response, columns=["sum"])

# Running Uvicorn with FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)