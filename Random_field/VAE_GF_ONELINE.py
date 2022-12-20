# Import library

import numpy as np
import torch
import torchvision
import math
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cmath
import warnings
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import wandb
from scipy.linalg import circulant

warnings.filterwarnings("ignore")
matplotlib.style.use('ggplot')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

wandb.init()

#################################
import tensorflow as tf
batch_size = 100
latent_dims = 800
T_dimes = latent_dims

m = 100 # number of block
l = 8 # size of each block
eigen_values = m * l
num_epochs = 100
lr = 1e-5
image_size = 49  # image size should be 20x20

latent_dime_sqr = int(np.sqrt(latent_dims))
latent_dims_T = int(latent_dims / 2)


dataset = torchvision.datasets.ImageFolder('C:/Users/farideh/Desktop/Rnadom field/VAE and RF/DATA',
                                           transforms.Compose([
                                               transforms.Grayscale(num_output_channels=1),
                                               transforms.Resize((image_size, image_size)),
                                               transforms.ToTensor(),


                                           ]))
print(len(dataset), " images loaded")

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          )

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        chanel = 64
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=2,
                               padding=1)  # out: 64 x 50 x 50
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2,
                               padding=1)  # out: 64x2 x 25 x 25
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=2,
                               padding=1)  # out: 64x4 x 12 x 12
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2,
                               padding=1)  # out: 512 x 6 x 6
        self.fc_eigen = nn.Linear(in_features=1024 * 4 * 4, out_features=eigen_values)

    def forward(self, x):
        c = 64
        x = nn.LeakyReLU()(self.conv1(x))
        x = torch.nn.BatchNorm2d(128)(x)
        # print("Encoder1:", x.shape)

        x = nn.LeakyReLU()(self.conv2(x))
        x = torch.nn.BatchNorm2d(256)(x)
        # print("Encoder2:", x.shape)

        x = nn.LeakyReLU()(self.conv3(x))
        x = torch.nn.BatchNorm2d(1024)(x)
        # print("Encoder3:", x.shape)

        x4 = nn.LeakyReLU()(self.conv4(x))
        x4 = torch.nn.BatchNorm2d(1024)(x4)
        # print("Encoder4:", x4.shape)

        x = x4.view(x.size(0), -1)  # flatten batch of multichannel feature maps to a batch of feature vectors

        x_eigen = self.fc_eigen(x)
        x_eigen = nn.ReLU()(x_eigen)

        return x_eigen


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.linear_de = nn.Linear(in_features=l*m, out_features=1024 * 3 * 3)
        self.convT1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.convT2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2)
        self.convT3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.convT4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2)
        self.convT5 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1)

    def forward(self, x_in):

        x = nn.Flatten()(x_in)
        x = self.linear_de(x)
        x0 = x.view((batch_size, 1024, 3, 3))

        x1 = (self.convT1(x0))
        # print("decoder1:", x1.shape)

        x2 = (self.convT2(x1))
        # print("decoder2:", x2.shape)

        x3 = (self.convT3(x2))
        # print("decoder3:", x3.shape)

        x4 = (self.convT4(x3))

        x5 = (self.convT5(x4))
        print(x5.shape)
        return x5


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_eigen = self.encoder(x)
        covariance, eigenvalues, Y = self.latent_sample(latent_eigen)
        x_recon = self.decoder(Y)
        return x_recon, covariance, eigenvalues

    ########### New sampling method ##################

    def latent_sample(self, eigenvalues):
        if self.training:

            Sigma_matrix = eigenvalues.reshape((batch_size, m, l))
            row_circulant = math.sqrt(m*l)*torch.fft.ifft2(Sigma_matrix)
            C = torch.zeros((batch_size, m*l, m*l))

            for i in range(batch_size):
                one_image = torch.reshape(row_circulant[i, :, :], (m*l, 1))
                cirulant = circulant(one_image.detach().numpy())
                C[i, :, :] = torch.tensor(cirulant, requires_grad=True)
            T = C[:, 0:400, 0:400] # Toeplitz matrix
            Sigma_matrix = eigenvalues.reshape((batch_size, m*l))
            Sigma_batch_eps = torch.zeros((batch_size, m*l))

            for i in range(eigen_values):
                batch_one_eigen = Sigma_matrix[:, i].detach().numpy()  # [batch od the ith eigenvalues]
                zeta = torch.zeros(batch_size, dtype=torch.cfloat)
                sample_a = torch.tensor(np.random.normal(0, batch_one_eigen))
                sample_b = torch.tensor(np.random.normal(0, batch_one_eigen))
                zeta.real = sample_b
                zeta.imag = sample_a

                Sigma_batch_eps[:, i] = torch.tensor(batch_one_eigen)

            Sigma_batch_eps = Sigma_batch_eps.reshape((batch_size, m, l))
            w = torch.fft.fft2(Sigma_batch_eps).real

            return C, eigenvalues, w
        else:
            return eigenvalues



def vae_loss(recon_x, x, cov):
    recon_loss = F.l1_loss(recon_x, x, reduction='sum')
    x = (torch.eye(T_dimes).reshape((1, T_dimes, T_dimes))).repeat(batch_size, 1, 1)
    sigma_sqr_sigma_sigma_sqr = 2 * torch.tensor(np.sqrt(torch.tensor(np.sqrt(x)) * cov.detach().numpy() * torch.tensor(np.sqrt(x))))

    W_distance = torch.diagonal((cov + x - sigma_sqr_sigma_sigma_sqr), dim1=1, dim2=2).sum(1)
    W_distance = torch.sum(W_distance, dim=0)
    print(W_distance)
    del sigma_sqr_sigma_sigma_sqr
    del x
    return recon_loss, W_distance


# reload a model
def load_ckp_VAE(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['auto_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizerd2_state_dict'])

    return model, optimizer, checkpoint['epoch']


vae = VariationalAutoencoder()

#num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)

optimizer = torch.optim.Adam(params=vae.parameters(), lr=lr)

# set to training mode
vae.train()
wandb.watch(vae, log='all', log_freq=10)
train_loss_avg = []
#
ckp_path_gen = "New.pth"
#vae, optimizer, start_epoch = load_ckp_VAE(ckp_path_gen, vae, optimizer)

a = 0



figloss, axloss = plt.subplots(ncols=1)
figx1, x1ss = plt.subplots(ncols=1)
figs, axs = plt.subplots(ncols=5, nrows=2)
figeig, axeigen = plt.subplots()
figcov, axcov = plt.subplots(ncols=5)


loss_ep = []
max = []
min = []
median = []

for epoch in range(num_epochs):
    train_loss_avg.append(0)

    a = a + 0.001

    for batch_idx, (image_batch, _) in enumerate(data_loader):
        if batch_idx + 1 == len(data_loader):
            break


        # print(image_batch.shape)

        # vae reconstruction

        image_batch_recon, cov, eigen = vae(image_batch)

        max.append(eigen.detach().numpy().max())
        min.append(eigen.detach().numpy().min())
        median.append(eigen.detach().numpy().mean())
        plt.close()
        axeigen.plot(max)
        axeigen.plot(min)
        axeigen.plot(median)
        figeig.savefig('eigen.png')

        inp_img_one = (image_batch[:, 0, :, :]).detach().cpu().numpy()
        reconstructed_img_one = (image_batch_recon[0:5, 0, :, :]).detach().cpu().numpy()
        plt.close()

        wandb.log({"Eigenvalues": wandb.Image(figx1)})

        plt.close()
        axs[0, 0].imshow(inp_img_one[0, :, :])
        axs[0, 0].grid(False)
        axs[0, 1].imshow(inp_img_one[1, :, :])
        axs[0, 1].grid(False)
        axs[0, 2].imshow(inp_img_one[2, :, :])
        axs[0, 2].grid(False)
        axs[0, 3].imshow(inp_img_one[3, :, :])
        axs[0, 3].grid(False)
        axs[0, 4].imshow(inp_img_one[4, :, :])
        axs[0, 4].grid(False)

        axs[1, 0].imshow(reconstructed_img_one[0, :, :])
        axs[1, 0].grid(False)
        axs[1, 1].imshow(reconstructed_img_one[1, :, :])
        axs[1, 1].grid(False)
        axs[1, 2].imshow(reconstructed_img_one[2, :, :])
        axs[1, 2].grid(False)
        axs[1, 3].imshow(reconstructed_img_one[3, :, :])
        axs[1, 3].grid(False)
        axs[1, 4].imshow(reconstructed_img_one[4, :, :])
        axs[1, 4].grid(False)
        plt.grid(b=None)
        figs.savefig('sample.png')
        wandb.log({"Image_recons": wandb.Image(figs)})
        plt.close()

        c = (cov[:, :, :]).detach().numpy()
        axcov[0].matshow(c[0, :, :].reshape(T_dimes,T_dimes))
        axcov[0].axes.get_xaxis().set_visible(False)
        axcov[0].axes.get_yaxis().set_visible(False)
        axcov[0].grid(False)
        axcov[1].matshow(c[1, :, :].reshape(T_dimes, T_dimes))
        axcov[1].axes.get_xaxis().set_visible(False)
        axcov[1].axes.get_yaxis().set_visible(False)
        axcov[1].grid(False)
        axcov[2].matshow(c[2, :, :].reshape(T_dimes, T_dimes))
        axcov[2].axes.get_xaxis().set_visible(False)
        axcov[2].axes.get_yaxis().set_visible(False)
        axcov[2].grid(False)
        axcov[3].matshow(c[3, :, :].reshape(T_dimes, T_dimes))
        axcov[3].axes.get_xaxis().set_visible(False)
        axcov[3].axes.get_yaxis().set_visible(False)
        axcov[3].grid(False)
        axcov[4].matshow(c[4, :, :].reshape(T_dimes, T_dimes))
        axcov[4].axes.get_xaxis().set_visible(False)
        axcov[4].axes.get_yaxis().set_visible(False)
        axcov[4].grid(False)

        plt.grid(b=None)
        plt.close()

        figcov.savefig("COVARIANCE.png")
        wandb.log({"Image_recons": wandb.Image(figcov)})
        plt.close()

        # reconstruction error
        loss_re,W_distance = vae_loss(image_batch_recon, image_batch, cov)
        print('Level 3')
        # print("KL loss:", loss_KL)

        loss = loss_re + a*W_distance

        print(f'Epoch [{epoch}/{num_epochs}]  Batch {batch_idx + 1}/{len(data_loader)} \
                    Train Loss : {loss:.4f}')
        wandb.log({"loss": loss})

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # print('Level 4')
        optimizer.step()
        #plot_grad_flow_v2(vae.named_parameters())


        # one step of the optimizer (using the gradients from backpropagation)





        # save the model
        # print('Level 5')
        torch.save({
            'epoch': epoch,
            'auto_state_dict': vae.state_dict(),
            'optimizerd2_state_dict': optimizer.state_dict(),
            'loss_auto': loss}, "New.pth")
        # print('Level 6')



