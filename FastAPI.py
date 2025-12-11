from fastapi import FastAPI
import torch
import pandas as pd
from pydantic import BaseModel

mlp_model = torch.load("mlp_model.pth")
ae_model = torch.load("autoencoder_model.pth")
vae_model = torch.load("vae_model.pth")
mlp_model.eval()
ae_model.eval()
vae_model.eval()

app = FastAPI(title="Fraud Detection API")

class Transaction(BaseModel):
  features: list

@app.post("/predict")
def predict(transaction:Transaction):
  x = torch.tensor([transaction.features], dtype=torch.float32)

  mlp_out = torch.argmax(mlp_model(x), dim=1).item()

  ae_recon = ae_model(x)
