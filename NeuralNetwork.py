import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy score

class MLP(nn.Module):
  def __init__(self, input_dim):
    super(MLP, self).__init__()
    self.layers = nn.Sequential(
      nn.Linear(input_dim, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 2)
    )
  def forward(self, x):
    return self.layers(x)


class Autoencoder(nn.Module):
  def __init__(self, input_dim):
    super(Autoencoder, self).__init__()
    self.encoder = nn.Sequential(
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Sequential(32, 16)
    )
    self.decoder = nn.Sequential(
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, input_dim),
      nn.Sigmoid()
    )
  def forward(self, x):
    z = self.encoder(x)
    return self.decoder(z)


class VAE(nn.Module):
  def __init__(self, input_dim, latent_dim = 16):
    super(VAE, self).__init__()
    self.fc1 = nn.Linear(input_dim, 32)
    self.fc_mu = nn.Linear(32, latent_dim)
    self.fc_logvar = nn.Linear(32, latent_dim)
    self.fc_decode = nn.Linear(latent_dim, input_dim)
  
  def encode(self, x):
    h = torch.relu(self.fc1(x))
    return self.fc_mu(h), self.fc_logvar(h)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    es = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z):
    return torch.sigmoid(self.fc_decode(z))

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar

def train_model(model, dataloader, epochs=5, lr=1e-3, is_autoencoder=False, is_vae=False):
  criterion = nn.CrossEntropyLoss() if not is_autoencoder else nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)

  for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X, y in dataloader:
      optimizer.zero_grad()

      if is_autoencoder:
        outputs = model(X)
        loss = criterion(outputs, X)
      elif is_vae():
        outputs, mu, logvar = model(X)
        recon_loss = nn.MSELoss()(outputs, X)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
      else:
        outputs = model(X)
        loss = criterion(outputs, y)

      loss.backward()
      optimizer.step()
      total_loss += loss.item()

  print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

def evaluate_model(model, dataloader):
  model.eval()
  y_true, y_pred = [], []
  with torch.no_grad():
    for X, y in dataloader:
      outputs = model(X)
      preds = torch.argmax(outputs, dim=1)
      y_true.extend(y.numpy())
      y_pred.extend(preds.numpy())
  print("Accuracy", accuracy_score(y_true, y_pred))
  print("F1 Score:", f1_score(y_true, y_pred))
  

