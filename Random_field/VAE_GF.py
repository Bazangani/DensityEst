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


###################################
# def plot_grad_flow_v2(named_parameters):
#     '''Plots the gradients flowing through different layers in the net during training.
#     Can be used for checking for possible gradient vanishing / exploding problems.
#
#     Usage: Plug this function in Trainer class after loss.backwards() as
#     "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
#
#     ave_grads = []
#     max_grads = []
#     layers = []
#     for n, p in named_parameters:
#
#         if p.grad != None:
#
#             if (p.requires_grad) and ("bias" not in n):
#                 layers.append(n)
#                 ave_grads.append(p.grad.abs().mean())
#
#             # fig, ax = plt.subplots()
#             plt.plot(ave_grads, alpha=0.3, color="b")
#
#             plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
#             plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
#             plt.xlim(xmin=0, xmax=len(ave_grads))
#             plt.xlabel("Layers")
#             plt.ylabel("average gradient")
#             plt.title("Gradient flow")
#             plt.grid(True)
#             plt.savefig("Gradient.png", bbox_inches='tight')
#             plt.close()


wandb.init()

#################################

batch_size = 100
latent_dims = 100
latent_dime_sqr = int(np.sqrt(latent_dims))
num_epochs = 100
latent_dims_T = int(latent_dims / 2)
lr = 1e-5
image_size = 49

dataset = torchvision.datasets.ImageFolder('C:/Users/farideh/Desktop/Rnadom field/VAE and RF/DATA',
                                           transforms.Compose([
                                               transforms.Grayscale(num_output_channels=1),
                                               # transforms.CenterCrop(320),
                                               # transforms.Resize((100, 100)),
                                               # transforms.RandomCrop(400),
                                               transforms.Resize((image_size, image_size)),
                                               transforms.ToTensor(),
                                               # transforms.Normalize(mean=[0], std=[0.98]),

                                           ]))
print(len(dataset)," images loaded")

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          )
def Conv_block(in_channel):
    conv_block = torch.nn.Sequential(
        nn.Conv2d(in_channel,in_channel, 3,stride=1,padding=1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(in_channel),
        nn.Conv2d(in_channel,in_channel, 3,stride=1,padding=1),
        nn.LeakyReLU()
    )
    return conv_block

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        chanel = 64
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2,
                               padding=1)  # out: 64 x 50 x 50
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2,
                               padding=1)  # out: 64x2 x 25 x 25
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2,
                               padding=1)  # out: 64x4 x 12 x 12
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2,
                               padding=1)  # out: 512 x 6 x 6
        # self.fc_mu = nn.Linear(in_features=c * 16 * 5 * 5, out_features=latent_dims)
        self.fc_eigen = nn.Linear(in_features=1024 * 4 * 4, out_features=latent_dims)

    def forward(self, x_in):
        c = 64
        # [100,1,49,49]

        x = nn.LeakyReLU()(self.conv1(x_in))
        x = torch.nn.BatchNorm2d(64)(x)


        xx = Conv_block(64)(x)
        y1 = torch.concat((x, xx), 1)





        # print("Encoder1:", x.shape)
        # [100, 128, 25, 25]



        x = nn.LeakyReLU()(self.conv2(y1))
        x = torch.nn.BatchNorm2d(256)(x)
        # print("Encoder2:", x.shape)
        # [100, 256, 13, 13]

        xx = Conv_block(256)(x)
        y1 = torch.concat((x, xx), 1)
        # print("Encoder2:", y1.shape)



        x = nn.LeakyReLU()(self.conv3(y1))
        x = torch.nn.BatchNorm2d(512)(x)
        xx = Conv_block(512)(x)
        y1 = torch.concat((x, xx), 1)
        # print("Encoder3:", y1.shape)
        # [100, 1024, 7, 7]

        x4 = nn.LeakyReLU()(self.conv4(y1))
        x4 = torch.nn.BatchNorm2d(1024)(x4)
        # print("Encoder4:", x4.shape)
        # [100, 1024, 4, 4]

        x = x4.view(x.size(0), -1)  # flatten batch of multichannel feature maps to a batch of feature vectors

        x_eigen = self.fc_eigen(x)
        x_eigen = nn.ReLU()(x_eigen)

        return x_eigen


class Decoder(nn.Module):



    def __init__(self):
        super(Decoder, self).__init__()

        self.linear_de = nn.Linear(in_features=latent_dims*latent_dims, out_features=1024 * 3 * 3)
        self.convT1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2,padding=1)
        self.convT2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2)
        self.convT3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.convT4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2)
        self.convT5 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1)

    def forward(self, x_in):
        x = nn.Flatten()(x_in)
        print(x.shape)
        x = self.linear_de(x)
        x0 = x.view((batch_size, 1024, 3, 3))
        # x0 = torch.nn.BatchNorm2d(512)(x0)

        # x = self.conv(x)
        # print(x.shape)

        x1 = (self.convT1(x0))
        # print("decoder1:", x1.shape)
        # x1 = torch.nn.BatchNorm2d(256)(x1)
        #
        x2 = (self.convT2(x1))
        # x2 = torch.nn.BatchNorm2d(128)(x2)

        # print("decoder2:", x2.shape)

        x3 = (self.convT3(x2))
        # x3 = torch.nn.BatchNorm2d(64)(x3)
        # print("decoder3:", x3.shape)

        x4 = (self.convT4(x3))
        # print(x4.shape)

        x5 = (self.convT5(x4))

        # x =(self.conv4(x))

        # print("decoder out", x4.shape)

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
        # def check_symmetric(a, rtol=1e-05, atol=1e-08):
        #     return np.allclose(a, a.T, rtol=rtol, atol=atol)

        if self.training:

            #### DFT (latent_dim, latent_dim) #####
            # latent_dimtion = eigenvalues.shape[1]
            # latent_dimemtion_sqr = int(np.sqrt(latent_dimtion))
            # latent_dims_T = int(latent_dimtion / 2)
            #
            # F = torch.zeros((latent_dimemtion_sqr, latent_dimemtion_sqr), dtype=torch.cfloat)
            #
            # for j in range(0, latent_dimemtion_sqr):
            #     for k in range(latent_dimemtion_sqr):
            #         value = cmath.exp(-2 * math.pi * (complex(0, 1)) * j * k / (latent_dimemtion_sqr)) / (
            #             np.sqrt(latent_dimemtion_sqr))
            #         F[j][k] = value
            #
            # DFT_mat = torch.kron(F, F)
            # del F
            # del value
            #
            # ##### DFT* (latent_dim, latent_dim) #####
            # DFT_mat_transpose = torch.t(DFT_mat)
            # DFT_Mat_conj = torch.conj(DFT_mat_transpose)
            # np.save('DFT_mat.npy', DFT_mat.resolve_conj().numpy())
            # np.save('DFT_Mat_conj.npy', DFT_Mat_conj.resolve_conj().numpy())

            DFT_mat = torch.tensor(np.load('DFT_mat.npy'))
            DFT_Mat_conj = torch.tensor(np.load('DFT_Mat_conj.npy'))




            #################### Matrix B : complex  square root of Σ #############
            B = torch.matmul(DFT_mat, torch.diag_embed(eigenvalues).cfloat())
            B = torch.matmul(B, DFT_Mat_conj).real


            # eps = torch.distributions.MultivariateNormal(torch.zeros(latent_dims), scale_tril=torch.diag(
            # torch.ones(latent_dims))) sample_a = eps.sample([batch_size]) sample_b = eps.sample([batch_size]) zeta

            # N_G = torch.distributions.Normal(0, 1)
            # esp = N_G.sample((batch_size, latent_dims,1))

            # x = y = range(latent_dimtion)
            # model = gs.Gaussian(dim=2, var=1, len_scale=1)
            # srf = gs.SRF(model, seed=20170519)
            # field = srf.structured([x, y])
            # # print(field.shape)
            # field = torch.tensor(field)
            zeta = torch.zeros((batch_size, latent_dims, latent_dims), dtype=torch.cfloat)
            sample_a = np.random.multivariate_normal(torch.zeros(1), torch.eye(1), (batch_size, latent_dims, latent_dims))
            sample_b = np.random.multivariate_normal(torch.zeros(1), torch.eye(1), (batch_size, latent_dims, latent_dims))

            zeta.real = torch.tensor(sample_a.reshape(batch_size, latent_dims, latent_dims))
            zeta.imag = torch.tensor(sample_b.reshape(batch_size, latent_dims, latent_dims))

            L = torch.matmul(DFT_Mat_conj, torch.square(torch.diag_embed(eigenvalues)).cfloat()).real
            #z = torch.matmul(L, zeta).real

            #Y = torch.torch.matmul(L, field.cfloat())


            del DFT_mat
            del DFT_Mat_conj

            return B, eigenvalues, L
        else:
            return eigenvalues


###################################################

def vae_loss(recon_x, x, cov):
    recon_loss = F.l1_loss(recon_x, x, reduction='sum')


    x = (torch.eye(latent_dims).reshape((1, latent_dims, latent_dims))).repeat(batch_size, 1, 1)
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



