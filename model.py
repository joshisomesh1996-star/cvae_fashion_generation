import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, condition_dim=30, latent_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,32,4,2,1),
            nn.ReLU(),

            nn.Conv2d(32,64,4,2,1),
            nn.ReLU(),

            nn.Conv2d(64,128,4,2,1),
            nn.ReLU(),

            nn.Conv2d(128,256,4,2,1),
            nn.ReLU()
        )

        self.fc = nn.Linear(256*8*8 + condition_dim, 512)

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self,x,c):

        x = self.conv(x)
        x = x.view(x.size(0),-1)

        x = torch.cat([x,c],dim=1)

        x = F.relu(self.fc(x))

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu,logvar


class Decoder(nn.Module):

    def __init__(self, condition_dim=30, latent_dim=128):
        super().__init__()

        self.fc = nn.Linear(latent_dim + condition_dim,256*8*8)

        self.deconv = nn.Sequential(

            nn.ConvTranspose2d(256,128,4,2,1),
            nn.ReLU(),

            nn.ConvTranspose2d(128,64,4,2,1),
            nn.ReLU(),

            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(),

            nn.ConvTranspose2d(32,3,4,2,1),
            nn.Sigmoid()
        )

    def forward(self,z,c):

        z = torch.cat([z,c],dim=1)

        x = self.fc(z)
        x = x.view(-1,256,8,8)

        x = self.deconv(x)

        return x


class CVAE(nn.Module):

    def __init__(self,condition_dim=30,latent_dim=128):
        super().__init__()

        self.encoder = Encoder(condition_dim,latent_dim)
        self.decoder = Decoder(condition_dim,latent_dim)