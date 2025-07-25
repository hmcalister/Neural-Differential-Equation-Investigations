# A script to test various overparameterizations of the network input, e.g. including (y**1, y**2, y**3...) when only y**3 is required.

from datetime import datetime
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from typing import Type

import torch
import torch.nn as nn
# pyright: reportPrivateImportUsage=false
import torch.optim as optim 

SAVED_DATA_PATH = "../data/optimizer"
os.makedirs(SAVED_DATA_PATH, exist_ok=True)

TORCH_DEVICE = "cuda"

USE_ADJOINT = True
DATA_STEPS = 1000
BATCH_SIZE = 20
BATCH_TIME = 10
NUM_EPOCHS = 2000
TEST_FREQUENCY = 10
HIDDEN_LAYER_SIZE = 50

if USE_ADJOINT:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def get_batch(t: torch.Tensor, true_y: torch.Tensor):
    s = torch.from_numpy(np.random.choice(np.arange(DATA_STEPS - BATCH_TIME, dtype=np.int64), BATCH_SIZE, replace=False))
    batch_y0 = true_y[s]
    batch_t = t[:BATCH_TIME]
    batch_y = torch.stack([true_y[s + i] for i in range(BATCH_TIME)], dim=0)
    return batch_y0.to(TORCH_DEVICE), batch_t.to(TORCH_DEVICE), batch_y.to(TORCH_DEVICE)

class Lambda(nn.Module):
    """
    A simulation of the "true" differential equation
    """

    def __init__(self, dimension: int, exponentCount: int, coefficientMatrix: torch.Tensor):
        super(Lambda, self).__init__()
        assert coefficientMatrix.shape[1] == dimension

        self.dimension = dimension
        self.exponentCount = exponentCount
        self.exponentVector = torch.arange(1,self.exponentCount+1).repeat_interleave(self.dimension).to(TORCH_DEVICE)
        self.coefficientMatrix = coefficientMatrix
    
    def forward(self, t, y):
        return torch.mm(torch.pow(y.tile(self.exponentCount), self.exponentVector), self.coefficientMatrix)


class ODEFunc(nn.Module):
    def __init__(self, dimension: int, exponentCount: int):
        super(ODEFunc, self).__init__()
        self.dimension = dimension
        self.exponentCount = exponentCount
        self.exponentVector = torch.arange(1,self.exponentCount+1).repeat_interleave(dimension).to(TORCH_DEVICE)
        
        self.net = nn.Sequential(
            nn.Linear(self.dimension * self.exponentCount, HIDDEN_LAYER_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYER_SIZE, self.dimension),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(torch.pow(y.tile(self.exponentCount), self.exponentVector))


def performOptimizerExperiment(optimizerClass: Type[optim.Optimizer], y0: torch.Tensor, trueCoefficientMatrix: torch.Tensor) -> pd.DataFrame:
    """
    Perform an experiment by giving a number of exponents corresponding to exponentCount to the network.

    Parameters
    ---
    optimizerClass: The optimizer function to use to train the ODEFunc
    y0: The initial condition of the differential equation. Shape (1, dimension)
    trueCoefficientMatrix: The coefficient matrix of the differential equation. Shape (dimension*exponentCount, dimension)
    """

    experimentDataRows = []

    DIMENSION = y0.shape[1]
    y0 = y0.to(TORCH_DEVICE)
    trueCoefficientMatrix = trueCoefficientMatrix.to(TORCH_DEVICE)
    t = torch.linspace(0., 25., DATA_STEPS).to(TORCH_DEVICE)

    with torch.no_grad():
        l = Lambda(DIMENSION, exponentCount, trueCoefficientMatrix)
        true_y: torch.Tensor = odeint(l, y0, t, method="dopri5") # type: ignore

    func = ODEFunc(DIMENSION, exponentCount).to(TORCH_DEVICE)
    optimizer: optim.Optimizer = optimizerClass(func.parameters(), lr=1e-3) # pyright: ignore[reportCallIssue]

    for epoch in tqdm(range(1,NUM_EPOCHS+1), desc="Epoch", leave=False):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(t, true_y)
        pred_y: torch.Tensor = odeint(func, batch_y0, batch_t).to(TORCH_DEVICE) # type: ignore
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        if epoch % TEST_FREQUENCY == 0:
            with torch.no_grad():
                pred_y: torch.Tensor = odeint(func, y0, t) # type: ignore
                loss = torch.mean(torch.abs(pred_y - true_y))
                predCoefficientMatrix: torch.Tensor = func.net(torch.eye(DIMENSION*exponentCount).to(TORCH_DEVICE))
                testRow = {
                    "epoch": epoch,
                    "time": time.time_ns(),
                    "loss": loss.cpu(),
                    "predCoefficientMatrix": predCoefficientMatrix.cpu(),
                }
                experimentDataRows.append(testRow)
    return pd.DataFrame(experimentDataRows)

if __name__ == "__main__":

    metadataDfFilepath = os.path.join(SAVED_DATA_PATH, "metadata.pkl")
    if os.path.exists(metadataDfFilepath):
        METADATA_DF = pd.read_pickle(metadataDfFilepath)
    else:
        METADATA_DF = pd.DataFrame(columns=[
            "fileTimestamp",
            "taskName",
            "useAdjoint",
            "dataSteps",
            "batchSize",
            "batchTime",
            "numEpochs",
            "hiddenLayerSize",
            "optimizer",
            "y0",
            "trueCoefficientMatrix"
        ])

    def addRowAndSaveMetadataDf(fileTimestamp: str, taskName: str, optimizerClass: Type[optim.Optimizer], y0: torch.Tensor, trueCoefficientMatrix: torch.Tensor):
        global METADATA_DF
        METADATA_DF = pd.concat([METADATA_DF, pd.DataFrame([[
            fileTimestamp,
            taskName,
            USE_ADJOINT,
            DATA_STEPS,
            BATCH_SIZE,
            BATCH_TIME,
            NUM_EPOCHS,
            HIDDEN_LAYER_SIZE,
            optimizerClass.__name__,
            y0.cpu(),
            trueCoefficientMatrix.cpu(),
        ]], columns=METADATA_DF.columns)])
        METADATA_DF.to_pickle(metadataDfFilepath)

    OPTIMIZER_LIST = [optim.RMSprop, optim.Adam, optim.AdamW, optim.SGD]

    taskName = "Original"
    for repeat in range(10):
        for optimizer in tqdm(OPTIMIZER_LIST, desc=f"Original Example Exponent Count Loop (Repeat: {repeat})"):
            dimension = 2
            exponentCount = 3
            y0 = torch.tensor([[2., 0.]]).to(TORCH_DEVICE)
            trueCoefficientMatrix = torch.zeros(size=(dimension*exponentCount, dimension)).to(TORCH_DEVICE)
            trueCoefficientMatrix[4,:] = torch.tensor([-0.1, 2.0])
            trueCoefficientMatrix[5,:] = torch.tensor([-2.0, -0.1])

            df = performOptimizerExperiment(optimizer, y0, trueCoefficientMatrix)
            fileTimestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            df.to_pickle(os.path.join(SAVED_DATA_PATH, fileTimestamp+".pkl"))
            addRowAndSaveMetadataDf(fileTimestamp, taskName, optimizer, y0.cpu(), trueCoefficientMatrix.cpu())

    taskName = "10-Dimensional"
    for repeat in range(10):     
        for optimizer in tqdm(OPTIMIZER_LIST, desc=f"10-Dimensional (Repeat: {repeat})"):
            dimension = 10
            exponentCount = 1
            y0 = torch.normal(mean=torch.zeros((1, dimension)), std=torch.ones((1, dimension))).to(TORCH_DEVICE)
            trueCoefficientMatrix = torch.zeros(size=(dimension*exponentCount, dimension)).to(TORCH_DEVICE)
            # In theory this should be a stable system? Each variable is negatively associated with all others dy/dt = -Ay for some A...
            trueCoefficientMatrix[:dimension, :] = -torch.rand((dimension, dimension))
            trueCoefficientMatrix.fill_diagonal_(0.0)
            
            try:
                df = performOptimizerExperiment(optimizer, y0, trueCoefficientMatrix)
            except AssertionError:
                continue
            fileTimestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            df.to_pickle(os.path.join(SAVED_DATA_PATH, fileTimestamp+".pkl"))
            addRowAndSaveMetadataDf(fileTimestamp, taskName, optimizer, y0.cpu(), trueCoefficientMatrix.cpu())
            
    taskName = "100-Dimensional"
    for repeat in range(10):     
        for optimizer in tqdm(OPTIMIZER_LIST, desc=f"100-Dimensional (Repeat: {repeat})"):
            dimension = 100
            exponentCount = 1
            y0 = torch.normal(mean=torch.zeros((1, dimension)), std=torch.ones((1, dimension))).to(TORCH_DEVICE)
            trueCoefficientMatrix = torch.zeros(size=(dimension*exponentCount, dimension)).to(TORCH_DEVICE)
            # In theory this should be a stable system? Each variable is negatively associated with all others dy/dt = -Ay for some A...
            trueCoefficientMatrix[:dimension, :] = -torch.rand((dimension, dimension))
            trueCoefficientMatrix.fill_diagonal_(0.0)

            try:
                df = performOptimizerExperiment(optimizer, y0, trueCoefficientMatrix)
            except AssertionError:
                continue
            fileTimestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            df.to_pickle(os.path.join(SAVED_DATA_PATH, fileTimestamp+".pkl"))
            addRowAndSaveMetadataDf(fileTimestamp, taskName, optimizer, y0.cpu(), trueCoefficientMatrix.cpu())
            