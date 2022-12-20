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

warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader

matplotlib.style.use('ggplot')
import os, psutil
import torch.nn.functional as F

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
###################################
import wandb

wandb.init()


def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:

        if p.grad != None:

            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())

            # fig, ax = plt.subplots()
            plt.plot(ave_grads, alpha=0.3, color="b")

            plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
            plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
            plt.xlim(xmin=0, xmax=len(ave_grads))
            plt.xlabel("Layers")
            plt.ylabel("average gradient")
            plt.title("Gradient flow")
            plt.grid(True)
            plt.savefig("Gradient.png", bbox_inches='tight')
            plt.close()


#################################

batch_size = 5
latent_dims = 1024
num_epochs = 100
lr = 1e-3
# Data
# transforms

dataset = torchvision.datasets.ImageFolder('C:/Users/farideh/Desktop/Rnadom field/VAE and RF/DATA',
                                           transforms.Compose([
                                               transforms.Grayscale(num_output_channels=1),
                                               # transforms.CenterCrop(200),
                                               transforms.Resize((97, 97)),

                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0], std=[0.98]),

                                           ]))

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = 64
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)  # out: 64 x 50 x 50
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 4, kernel_size=4, stride=2,
                               padding=1)  # out: 64x2 x 25 x 25
        self.conv3 = nn.Conv2d(in_channels=c * 4, out_channels=c * 8, kernel_size=4, stride=2,
                               padding=1)  # out: 64x4 x 12 x 12
        self.conv4 = nn.Conv2d(in_channels=c * 8, out_channels=c * 16, kernel_size=4, stride=2,
                               padding=1)  # out: 512 x 6 x 6
        # self.fc_mu = nn.Linear(in_features=c * 16 * 5 * 5, out_features=latent_dims)
        self.fc_eigen = nn.Linear(in_features=c * 16 * 6 * 6, out_features=latent_dims)

    def forward(self, x):
        c = 64
        x = nn.LeakyReLU()(self.conv1(x))
        x = torch.nn.BatchNorm2d(c)(x)
        # print("ENcoder1:", x.shape)

        x = nn.LeakyReLU()(self.conv2(x))
        x = torch.nn.BatchNorm2d(c * 4)(x)
        # print("ENcoder2:", x.shape)

        x = nn.LeakyReLU()(self.conv3(x))
        x = torch.nn.BatchNorm2d(c * 8)(x)
        # print("ENcoder3:", x.shape)

        x4 = nn.LeakyReLU()(self.conv4(x))
        x4 = torch.nn.BatchNorm2d(c * 16)(x4)
        # print("ENcoder4:", x4.shape)

        x = x4.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors

        x_eigen = self.fc_eigen(x)
        x_eigen = nn.ReLU()(x_eigen)

        return x_eigen, x4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = 64
        feature_map = 8
        self.fc1 = nn.Linear(in_features=latent_dims * batch_size,
                             out_features=batch_size * 68 * feature_map * feature_map)
        self.fc2 = nn.Linear(in_features=batch_size * 68 * feature_map * feature_map,
                             out_features=batch_size * 128 * feature_map * feature_map)

        self.conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2)
        self.conv4 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=1)
        self.conv5 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=5, stride=1)
        self.conv6 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=5, stride=1)
        # self.conv7 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=5, stride=1)
        # self.conv8 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=5, stride=1)

        # self.fc = nn.Linear(in_features=latent_dims, out_features=c * 4 * 11 * 11)   # out: C*8 X 5 X 5
        # self.conv1 = nn.ConvTranspose2d(in_channels=c * 4, out_channels=c *3, kernel_size=5, stride=2, padding=1)# out: C* 8 X 23 X 23
        # self.conv2 = nn.ConvTranspose2d(in_channels=c * 3, out_channels=c *2, kernel_size=5, stride=2, padding=1) # out: C* 8 X 47 X 47
        # self.conv3 = nn.ConvTranspose2d(in_channels=c * 2, out_channels=1, kernel_size=5, stride=2,padding=1)  # out: C* 8 X 95 X 95
        # self.conv4 = nn.ConvTranspose2d(in_channels=c * 1, out_channels=int(c/2), kernel_size=5, stride=1,padding=1)  # out: C* 8 X 97 X 97
        # self.conv5 = nn.ConvTranspose2d(in_channels=int(c/2) , out_channels=1, kernel_size=5, stride=1)  # out: C* 8 X 100 X 100

    def forward(self, x, x4):
        c = 64
        feature_map = 8
        x = torch.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)  # out: C*8 X 5 X 5
        x = x.view(batch_size, 128, feature_map, feature_map)

        x = (self.conv1(x))
        x = nn.LeakyReLU()(x)

        # x4 = x4.reshape((batch_size, 256, 12, 12))
        # x4 = x4[:, 0:32, :, :]
        # pad1 = (3, 4)
        # x4_p = F.pad(x4, pad1, "constant", 0)
        # x4_p = x4_p.reshape(batch_size, 32, 19, 12)

        # x4_p = F.pad(x4_p, (3,4), "constant", 0)
        # print(x4_p.shape)
        # print("decoder1:", x.shape)
        # x = x4_p+x

        x = (self.conv2(x))
        x = nn.LeakyReLU()(x)
        # print("decoder2:", x.shape)

        x = (self.conv3(x))
        x = nn.LeakyReLU()(x)
        # print("decoder3:", x.shape)

        x = (self.conv4(x))
        x = nn.LeakyReLU()(x)
        # print("decoder4:", x.shape)

        x = (self.conv5(x))
        x = nn.LeakyReLU()(x)
        # print("decoder5:", x.shape)

        x = (self.conv6(x))
        # x = nn.LeakyReLU()(x)
        # print("decoder6:", x.shape)

        x = torch.sigmoid(x)  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_eigen, x4 = self.encoder(x)
        cov, latent = self.latent_sample(latent_eigen)
        x_recon = self.decoder(latent, x4)
        return x_recon, cov

    ########### New sampling method ##################

    def latent_sample(self, eigen):
        def check_symmetric(a, rtol=1e-05, atol=1e-08):
            return np.allclose(a, a.T, rtol=rtol, atol=atol)

        if self.training:

            latent_dims = eigen.shape[1]
            latent_dims_T = int(latent_dims / 2)
            eigenvalues = eigen
            sigma_matrix = torch.diag_embed(eigenvalues)
            sigma_matrix_sqr = torch.linalg.matrix_power(sigma_matrix, 2)

            ##### DFT (latent_dim, latent_dim) #####
            DFT_mat = torch.zeros((latent_dims, latent_dims), dtype=torch.cfloat)
            #two_phi_i = cmath.exp((-2 * math.pi * (complex(0, 1)) / complex(latent_dims, 0)))
            for j in range(0, latent_dims):
                for k in range(0, latent_dims):
                    value = cmath.exp(-2 * math.pi * (complex(0, 1))*j*k/(latent_dims))/(np.sqrt(latent_dims))
                    DFT_mat[j][k] = value
            ##### DFT* (latent_dim, latent_dim) #####
            DFT_mat_transpose = torch.t(DFT_mat)
            DFT_Mat_conj = torch.conj(DFT_mat_transpose)
            # DFT_mat -= DFT_mat.mean((0, 1), keepdim=True)
            # DFT_mat /= DFT_mat.std((0, 1), keepdim=True)

            # DFT_mat_transpose -= DFT_mat_transpose.mean((0, 1), keepdim=True)
            # DFT_mat_transpose /= DFT_mat_transpose.std((0, 1), keepdim=True)

            ############################################
            #  compute the covariance matrix
            # B = DFT[latent_dim, latent_dim] x
            # sigma_matrix [batch_size, latent_dime,latent_dime]  x
            # DFT*[latent_dim, latent_dim]

            B = torch.matmul(DFT_mat, sigma_matrix.cfloat())
            B = torch.matmul(B, DFT_Mat_conj).real  # B shape (batch_size,latent_dims,latent_dims)
            # print(check_symmetric(B.detach().numpy()))

            ##############################################
            # compute the sample Y [batch_size, latent_dims] = DFT [latent_dims, latent_dims] x
            # sigma_matrix_sqr [batch_size, latent_dime,latent_dime] x
            # zeta [latent_dime, 1]

            eps_a = np.random.multivariate_normal(torch.zeros(1), torch.eye(1), (batch_size, latent_dims, 1))
            eps_b = np.random.multivariate_normal(torch.zeros(1), torch.eye(1), (batch_size, latent_dims, 1))
            # N_G = torch.distributions.Normal(0, 1)
            # esp = N_G.sample((batch_size, batch_size))

            zeta = torch.zeros((batch_size, latent_dims, 1), dtype=torch.cfloat)

            zeta.real = torch.tensor(eps_a.reshape(batch_size, latent_dims, 1))
            zeta.imag = torch.tensor(eps_b.reshape(batch_size, latent_dims, 1))

            Y = torch.matmul(DFT_mat, sigma_matrix_sqr.cfloat())

            # print(Y.shape)
            Y = torch.matmul(Y, DFT_Mat_conj)

            Y = torch.matmul(Y.cfloat(), zeta.cfloat())
            Y = Y.real + Y.imag
            #########################################
            return B, Y
        else:
            return eigen


###################################################

def vae_loss(recon_x, x, cov):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')

    KLD = torch.mean(0.5 * torch.sum(torch.sum(1 + cov, dim=2), dim=1), dim=0)

    # KLD = -0.5 * torch.sum(torch.sum(1 + cov- cov.exp()))
    # print(torch.log(torch.det(cov) + 0.1 * torch.ones(batch_size)))
    # print(torch.det(cov))
    # KLD = torch.mean((cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)) + torch.log(torch.det(cov)), dim=0)
    return recon_loss, KLD


vae = VariationalAutoencoder()

num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)

optimizer = torch.optim.SGD(params=vae.parameters(), lr=lr, weight_decay=1e-5)

# set to training mode
vae.train()
wandb.watch(vae, log='all', log_freq=10)
train_loss_avg = []

print('Training ...')
a = 0
fig, axs = plt.subplots(ncols=5, nrows=2)
figloss, axloss = plt.subplots(ncols=1)
figcov, axcov = plt.subplots(ncols=5)

# figloss, axloss = plt.subplots(ncols=1)
loss_ep = []
for epoch in range(num_epochs):
    train_loss_avg.append(0)
    num_batches = 0
    a = a + 0.001

    for batch_idx, (image_batch, _) in enumerate(data_loader):
        if batch_idx + 1 == len(data_loader):
            break

        image_batch = image_batch

        # vae reconstruction
        image_batch_recon, cov = vae(image_batch)
        inp_img_one = (image_batch[0:5, 0, :, :]).detach().cpu().numpy()
        reconstructed_img_one = (image_batch_recon[0:5, 0, :, :]).detach().cpu().numpy()

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

        fig.savefig('sample.png')
        wandb.log({"Image_recons": wandb.Image(fig)})
        plt.close()
        c = (cov[:, :, :]).detach().numpy()
        axcov[0].matshow(c[0, :, :].reshape(latent_dims, latent_dims))
        axcov[0].axes.get_xaxis().set_visible(False)
        axcov[0].axes.get_yaxis().set_visible(False)
        axcov[0].grid(False)
        axcov[1].matshow(c[1, :, :].reshape(latent_dims, latent_dims))
        axcov[1].axes.get_xaxis().set_visible(False)
        axcov[1].axes.get_yaxis().set_visible(False)
        axcov[1].grid(False)
        axcov[2].matshow(c[2, :, :].reshape(latent_dims, latent_dims))
        axcov[2].axes.get_xaxis().set_visible(False)
        axcov[2].axes.get_yaxis().set_visible(False)
        axcov[2].grid(False)
        axcov[3].matshow(c[3, :, :].reshape(latent_dims, latent_dims))
        axcov[3].axes.get_xaxis().set_visible(False)
        axcov[3].axes.get_yaxis().set_visible(False)
        axcov[3].grid(False)
        axcov[4].matshow(c[4, :, :].reshape(latent_dims, latent_dims))
        axcov[4].axes.get_xaxis().set_visible(False)
        axcov[4].axes.get_yaxis().set_visible(False)
        axcov[4].grid(False)
        # axcov.grid(False)
        plt.grid(b=None)

        figcov.savefig("COVARIANCE.png")
        wandb.log({"Image_recons": wandb.Image(figcov)})
        plt.close()

        # reconstruction error
        loss_re, loss_KL = vae_loss(image_batch_recon, image_batch, cov)
        print("KL loss:", loss_KL)

        loss = loss_re  # + a*loss_KL

        print(f'Epoch [{epoch}/{num_epochs}]  Batch {batch_idx + 1}/{len(data_loader)} \
                    Train Loss : {loss:.4f}')
        wandb.log({"loss": loss})

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        plot_grad_flow_v2(vae.named_parameters())

        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()

        train_loss_avg[-1] += loss.item()
        num_batches += 1

    train_loss_avg[-1] /= num_batches
