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
import gstools as gs

warnings.filterwarnings("ignore")
matplotlib.style.use('ggplot')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

wandb.init()

#################################
import tensorflow as tf
batch_size = 50
latent_dims = 800
T_dimes = latent_dims
eigen_values = 10000
m = 100 # number of block
l = 8 # size of each block
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
        self.convT1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2,padding=1)
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

            Sigma_batch = torch.zeros((batch_size, l * m, m * l))  # [batch_size,l*m, m*l]
            for i in range(batch_size):
                for j in range(m):
                    Sigma_batch[i, (j * l):(j * l)+l, (j * l):((j * l) + l)] = torch.ones((l,l)) * eigenvalues[i, j]

            #print(Sigma_batch)
            #  F_B (batch_size, lxm, lxm) =
            # F_m {one dimensional discreet fourier transform of order m} O
            # I (lxl)
            #
            F_B = torch.zeros((m*l, m*l), dtype=torch.cfloat)
            for j in range(m):
                for k in range(m):
                    value = (cmath.exp(((-2* math.pi) * (complex(0, 1)))/ m )** j ** k) * torch.eye(l)

                    F_B[(j * l): ((j * l) + l),  (k * l): ((k * l) + l)] = value

            F_B =torch.nn.functional.normalize(F_B)
            F_B_H = torch.nn.functional.normalize(torch.conj(torch.t(F_B)))

            np.save('DFT_mat.npy', F_B.resolve_conj().numpy())
            np.save('DFT_Mat_conj.npy', F_B_H.resolve_conj().numpy() )

            F_B = torch.tensor(np.load('DFT_mat.npy'))
            F_B_H = torch.tensor(np.load('DFT_Mat_conj.npy'))

            C_0 = (1/m) * torch.matmul(F_B, Sigma_batch.cfloat())
            C = torch.matmul(C_0, F_B_H).real  # Circulant matrix
            T = C[:, 0:400, 0:400] # Toeplitz matrix

            # sampling
            zeta = torch.zeros((batch_size, l*m,1), dtype=torch.cfloat)
            sample_a = np.random.multivariate_normal(torch.zeros(1),
                                                     torch.eye(1),
                                                     (batch_size, l*m, 1))
            sample_b = np.random.multivariate_normal(torch.zeros(1),
                                                     torch.eye(1),
                                                     (batch_size, l*m, 1))
            zeta.real = torch.tensor(sample_a.reshape(batch_size, l*m, 1))
            zeta.imag = torch.tensor(sample_b.reshape(batch_size, l*m, 1))

            Y = torch.matmul(F_B_H, torch.square(Sigma_batch).cfloat()) # [batch_size, l*m, l*m]

            z = torch.matmul(Y, zeta).real  # [batch_size, l*m, 1]

            return C, eigenvalues, z
        else:
            return eigenvalues



def vae_loss(recon_x, x, cov):
    recon_loss = F.l1_loss(recon_x, x, reduction='sum')
    x = (torch.eye(T_dimes).reshape((1, T_dimes, T_dimes))).repeat(batch_size, 1, 1)
    sigma_sqr_sigma_sigma_sqr = 2 * torch.tensor(np.sqrt(torch.tensor(np.sqrt(x)) * cov.detach().numpy() * torch.tensor(np.sqrt(x))))

    W_distance = torch.diagonal((cov +x - sigma_sqr_sigma_sigma_sqr), dim1=1, dim2=2).sum(1)
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



