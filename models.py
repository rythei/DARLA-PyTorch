'''

Implementation of DARLA preprocessing, as found in DARLA: Improving Zero-Shot Transfer in Reinforcement Learning
by Higgins and Pal et al (https://arxiv.org/pdf/1707.08475.pdf):


DAE:

X_noisy --J--> Z ----> X_hat

minimizing (X_noisy-X_hat)^2


Beta VAE:

X ----> Z ----> X_hat

minimizing (J(X) - J(X_hat))^2 + beta*KL(Q(Z|X) || P(Z))


Right now this just trains the model using MNIST dataset

rythei

'''

import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import *
from torch.nn import functional as F


class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()

        self.image_dim = 28 # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
                            # can calculate this using output_after_conv() in utils.py
        self.latent_dim = 100
        self.noise_scale = 0.001
        self.batch_size = 50

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU())
        self.fc1 = nn.Linear(32*4*4, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, 32*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=1),
	    nn.Sigmoid())

    def forward(self, x):
        x = torch.add(x, Variable(self.noise_scale*torch.randn(self.batch_size, 1, self.image_dim, self.image_dim)))
        z = self.encoder(x)
        z = z.view(-1, 32*4*4)
        z = self.fc1(z)
        x_hat = self.fc2(z)
        x_hat = x_hat.view(-1, 32, 4, 4)
        x_hat = self.decoder(x_hat)

        return z, x_hat

    def encode(self, x):
        #x = x.unsqueeze(0)
        z, x_hat = self.forward(x)

        return z


def train_dae(num_epochs = 100, batch_size = 128, learning_rate = 1e-3):
    train_dataset = dsets.MNIST(root='./data/',         #### testing that it works with MNIST data
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

    dae = DAE()
    dae.batch_size = batch_size

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dae.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            x = Variable(images)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            z, x_hat = dae(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))
                torch.save(dae.state_dict(), 'dae-test-model.pkl')


class BetaVAE(nn.Module):
    def __init__(self):
        super(BetaVAE, self).__init__()

        self.image_dim = 28 # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
                            # can calculate this using output_after_conv() in utils.py
        self.latent_dim = 100
        self.batch_size = 50

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU())
        self.fc_mu = nn.Linear(32*4*4, self.latent_dim)
        self.fc_sigma = nn.Linear(32 * 4 * 4, self.latent_dim)
        self.fc_up = nn.Linear(self.latent_dim, 32*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=1),
            nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(-1, 32*4*4)
        mu_z = self.fc_mu(z)
        log_sigma_z = self.fc_sigma(z)
        sample_z = mu_z + log_sigma_z.exp()*Variable(torch.randn(self.batch_size, self.latent_dim))
        x_hat = self.fc_up(sample_z)
        x_hat = x_hat.view(-1, 32, 4, 4)
        x_hat = self.decoder(x_hat)

        return mu_z, log_sigma_z, x_hat


def bvae_loss_function(z_hat, z, mu, logvar, beta=1, batch_size=128):
    RCL = F.mse_loss(z, z_hat) #reconstruction loss

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #KL divergence
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size

    return RCL + beta*KLD

def train_bvae(num_epochs = 100, batch_size = 128, learning_rate = 1e-4):
    train_dataset = dsets.MNIST(root='./data/',  #### testing that it works with MNIST data
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

    bvae = BetaVAE()
    bvae.batch_size = batch_size

    dae = DAE()
    dae.load_state_dict(torch.load('dae-test-model.pkl'))
    dae.batch_size = batch_size
    dae.eval()

    optimizer = torch.optim.Adam(bvae.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            x = Variable(images)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            mu_z, log_sigma_z, x_hat = bvae(x)

            loss = bvae_loss_function(dae.encode(x_hat), dae.encode(x), mu_z, 2*log_sigma_z, batch_size=batch_size)
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

        torch.save(bvae.state_dict(), 'bvae-test-model.pkl')

if __name__ == '__main__':
    train_dae() #first need to dave a DAE model trained
    train_bvae()
