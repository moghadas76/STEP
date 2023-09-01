import os
import torch
import numpy as np
from uuid import uuid4

def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss: torch.Tensor = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss_name = f"mae_{uuid4()}"
    np.savez(
    os.path.join("/home/seyed/PycharmProjects/step/STEP/loss/", "%s.npz" % loss_name),
    mae=loss.cpu().data.numpy()
    )
    return torch.mean(loss)
