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
warnings.filterwarnings("ignore")
matplotlib.style.use('ggplot')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
###################################

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

batch_size = 100
latent_dims = 2025
latent_dime_sqr = int(np.sqrt(latent_dims))
num_epochs = 200
latent_dims_T = int(latent_dims/2)
lr = 1e-5

dataset = torchvision.datasets.ImageFolder('C:/Users/farideh/Desktop/Rnadom field/VAE and RF/DATA',
                                           transforms.Compose([
                                               transforms.Grayscale(num_output_channels=1),
                                               #transforms.CenterCrop(200),
                                               transforms.Resize((91, 91)),

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
        self.fc_eigen = nn.Linear(in_features=c * 16 * 5 * 5, out_features=latent_dims)

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
        #print("Encoder4:", x4.shape)

        x = x4.view(x.size(0), -1)  # flatten batch of multichannel feature maps to a batch of feature vectors

        x_eigen = self.fc_eigen(x)
        x_eigen = nn.ReLU()(x_eigen)

        return x_eigen


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        #self.conv = nn.Conv2d(in_channels=1, out_channels=32,kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=3, stride=2,padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, stride=1)
        #self.conv3 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=1)


    def forward(self, x):
        #x = self.conv(x)
        #x = nn.LeakyReLU()(x)
        #print(x.shape)


        x = (self.conv1(x))
        x = nn.LeakyReLU()(x)
        #print("decoder1:", x.shape)

        x = (self.conv2(x))
        #x = nn.LeakyReLU()(x)

        # print("decoder2:", x.shape)

        # x = (self.conv3(x))
        x = torch.sigmoid(x)
        #print("decoder out",x.shape)

        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_eigen = self.encoder(x)
        cov, latent = self.latent_sample(latent_eigen)
        x_recon = self.decoder(latent)
        return x_recon, cov,latent

    ########### New sampling method ##################

    def latent_sample(self, eigen):
        def check_symmetric(a, rtol=1e-05, atol=1e-08):
            return np.allclose(a, a.T, rtol=rtol, atol=atol)

        if self.training:


            latent_dims = eigen.shape[1]
            latent_dime_sqr = int(np.sqrt(latent_dims))
            latent_dims_T = int(latent_dims / 2)
            eigenvalues = eigen
            sigma_matrix = torch.diag_embed(eigenvalues)
            sigma_matrix_sqr = torch.square(sigma_matrix)


            ##### DFT (latent_dim, latent_dim) #####
            #DFT_mat = torch.zeros((latent_dims, latent_dims), dtype=torch.cfloat)
            F = torch.zeros((latent_dime_sqr, latent_dime_sqr), dtype=torch.cfloat)

            for j in range(0, latent_dime_sqr):
                for k in range(latent_dime_sqr):
                    value = cmath.exp(-2 * math.pi * (complex(0, 1))*j*k/(latent_dime_sqr))/(np.sqrt(latent_dime_sqr))
                    F[j][k] = value

            DFT_mat = torch.kron(F,F)
            ##### DFT* (latent_dim, latent_dim) #####
            DFT_mat_transpose = torch.t(DFT_mat)
            DFT_Mat_conj = torch.conj(DFT_mat_transpose)




            #################### Matrix B : complex  square root of Σ #############
            B = torch.matmul(DFT_mat, sigma_matrix.cfloat())
            B = torch.matmul(B, DFT_Mat_conj).real
            B = B[:, 0:latent_dims_T, 0:latent_dims_T]
            print(check_symmetric(B[0,:,:].detach().numpy()))

            eps_a = torch.normal(0, 1, size=(batch_size, latent_dims, 1))
            eps_b = torch.normal(0, 1, size=(batch_size, latent_dims, 1))
            zeta = torch.zeros((batch_size, latent_dims, 1), dtype=torch.cfloat)

            zeta.real = torch.tensor(eps_a.reshape(batch_size, latent_dims, 1))
            zeta.imag = torch.tensor(eps_b.reshape(batch_size, latent_dims, 1))

            Y = torch.matmul(DFT_mat, sigma_matrix_sqr.cfloat())

            l = torch.matmul(Y.cfloat(), zeta.cfloat())
            Y = l.real
            z = l.imag

            Y = torch.reshape(Y, (batch_size,  latent_dime_sqr, latent_dime_sqr)) #batch_size x 16 x16
            z = torch.reshape(z, (batch_size,  latent_dime_sqr, latent_dime_sqr)) #batch_size x 16 x16
            YY = torch.zeros((batch_size, 2, latent_dime_sqr, latent_dime_sqr))
            print(YY.shape)
            print(z.shape)
            YY[:, 0, :, :] = Y
            YY[:, 1, :, :] = z


            #########################################
            return B, YY
        else:
            return eigen


###################################################

def vae_loss(recon_x, x, cov):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')

    ones = torch.eye(latent_dims_T)
    x = ones.reshape((1, latent_dims_T, latent_dims_T))
    y = x.repeat(batch_size, 1, 1)

    I_sqr = torch.tensor(np.sqrt(y))
    sigma_sqr_sigma_sigma_sqr = 2*torch.tensor(np.sqrt(I_sqr * cov.detach().numpy() * I_sqr))
    sigma_tr =  y - sigma_sqr_sigma_sigma_sqr
    W_distance = torch.diagonal(sigma_tr, dim1=1, dim2=2).sum(0)
    W_distance = torch.mean(W_distance, dim=0)
    # KLD = torch.mean(-0.5 * torch.sum(torch.sum(1 + cov -cov.exp(), dim=2), dim=1), dim=0)
    #print(W_distance.shape)

    # KLD = -0.5 * torch.sum(torch.sum(1 + cov- cov.exp()))
    # print(torch.log(torch.det(cov) + 0.1 * torch.ones(batch_size)))
    # print(torch.det(cov))
    #KLD = torch.mean((cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)) + torch.log(torch.det(cov)), dim=0)
    return recon_loss, W_distance


vae = VariationalAutoencoder()

num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)

optimizer = torch.optim.Adam(params=vae.parameters(), lr=lr, weight_decay=1e-5)

# set to training mode
vae.train()
wandb.watch(vae, log='all', log_freq=10)
train_loss_avg = []

print('Training ...')
a = 0
figs, axs = plt.subplots(ncols=5, nrows=2)
figloss, axloss = plt.subplots(ncols=1)
figcov, axcov = plt.subplots(ncols=5)
figlatenet, axlatent = plt.subplots(ncols=5)
# figloss, axloss = plt.subplots(ncols=1)
loss_ep = []
#latent_dims_T =int(latent_dims/2)

for epoch in range(num_epochs):
    train_loss_avg.append(0)
    num_batches = 0
    a = a + 0.001

    for batch_idx, (image_batch, _) in enumerate(data_loader):
        if batch_idx + 1 == len(data_loader):
            break

        image_batch = image_batch

        # vae reconstruction
        image_batch_recon, cov,latent = vae(image_batch)
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
        figs.savefig('sample.png')
        wandb.log({"Image_recons": wandb.Image(figs)})
        plt.close()
        print(latent.shape)
        axlatent[0].imshow(latent[0,0, :, :].detach().cpu().numpy())
        axlatent[0].grid(False)
        axlatent[1].imshow(latent[1,0, :, :].detach().cpu().numpy())
        axlatent[1].grid(False)
        axlatent[2].imshow(latent[2,0, :, :].detach().cpu().numpy())
        axlatent[2].grid(False)
        axlatent[3].imshow(latent[3,0, :, :].detach().cpu().numpy())
        axlatent[3].grid(False)
        axlatent[4].imshow(latent[4,0, :, :].detach().cpu().numpy())
        axlatent[4].grid(False)
        plt.grid(b=None)

        figlatenet.savefig('latent.png')
        wandb.log({"latent": wandb.Image(figlatenet)})
        plt.close()




        c = (cov[:, :, :]).detach().numpy()
        axcov[0].matshow(c[0, :, :].reshape(latent_dims_T, latent_dims_T))
        axcov[0].axes.get_xaxis().set_visible(False)
        axcov[0].axes.get_yaxis().set_visible(False)
        axcov[0].grid(False)
        axcov[1].matshow(c[1, :, :].reshape(latent_dims_T, latent_dims_T))
        axcov[1].axes.get_xaxis().set_visible(False)
        axcov[1].axes.get_yaxis().set_visible(False)
        axcov[1].grid(False)
        axcov[2].matshow(c[2, :, :].reshape(latent_dims_T, latent_dims_T))
        axcov[2].axes.get_xaxis().set_visible(False)
        axcov[2].axes.get_yaxis().set_visible(False)
        axcov[2].grid(False)
        axcov[3].matshow(c[3, :, :].reshape(latent_dims_T, latent_dims_T))
        axcov[3].axes.get_xaxis().set_visible(False)
        axcov[3].axes.get_yaxis().set_visible(False)
        axcov[3].grid(False)
        axcov[4].matshow(c[4, :, :].reshape(latent_dims_T, latent_dims_T))
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

        loss = a*loss_KL + loss_re

        print(f'Epoch [{epoch}/{num_epochs}]  Batch {batch_idx + 1}/{len(data_loader)} \
                    Train Loss : {loss:.4f}')
        wandb.log({"loss": loss})

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        plot_grad_flow_v2(vae.named_parameters())

        # one step of the optimizer (using the gradients from backpropagation)
        optimizer.step()

        train_loss_avg[-1] += loss.item()
        num_batches += 1

    #train_loss_avg[-1] /= num_batches
