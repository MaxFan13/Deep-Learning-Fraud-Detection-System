import torch
import torch.nn as nn

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
