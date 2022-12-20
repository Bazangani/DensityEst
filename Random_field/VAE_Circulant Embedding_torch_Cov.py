import torch
import torchvision
import torch.optim as optim
import argparse
import os
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchsummary
import numpy as np
import gc
import itertools
from torch.utils.data import DataLoader
import scipy.linalg
matplotlib.style.use('ggplot')
import math
import cmath

# Data

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((100, 100)),
])

dataset = torchvision.datasets.ImageFolder('C:/Users/farideh/Desktop/Rnadom field/VAE and RF/DATA',
                                                transforms.Compose([
                                                transforms.Grayscale(num_output_channels=1),
                                                transforms.Resize((100, 100)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.1], std=[0.9]),

                                                ]))

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=50,
                                          shuffle=True,
                                          )

print("Number of samples:",len(data_loader))

#dataiter = iter(data_loader)
#images, labels = dataiter.next()
#print(type(images))
#print(images.shape)
#print(labels.shape)

# Block of conv
def Conv_block(in_ch,out_ch):
    conv_block = nn.Sequential(

            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch,out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(in_ch,out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU()

    )
    return conv_block


# Circulant function defined here
def circulant(arr, n):
    # Initializing an empty
    # 2D matrix of order n
    c = [[0 for i in range(49)] for j in range(49)]
    for k in range(24): # n is only for 25 values
        #print(k)
        c[k][0] = arr[k]
    for i in range(0, 25):
        #print(48-i)
        c[48-i][0] = c[0][i]


    # Forming the circulant matrix
    for i in range(1, n):
        for j in range(n):
            if (j - 1 >= 0):
                c[j][i] = c[j - 1][i - 1]
            else:
                c[j][i] = c[n - 1][i - 1]
    return c


# define a simple linear VAE
class VAE(nn.Module):
    def __init__(self, inp, out):
        super(VAE, self).__init__()


        # encoder
        self.x1 = Conv_block(1,1)
        self.x1_res = nn.AvgPool2d(2, stride=2, padding=1) # 50 x 50
        self.x1_res_chanel = nn.Conv2d(1, 16, 3,  stride=1, padding=1) # increase the channel to 16

        self.x2 = Conv_block(16, 16)
        self.x2_res = nn.AvgPool2d(2, stride=2, padding=1)   # 25 x 25
        self.x2_res_chanel = nn.Conv2d(16, 32, 3, stride=1, padding=1) # increase the channel to 32

        self.x3 = Conv_block(32, 32)
        self.x3_res = nn.AvgPool2d(2, stride=2, padding=1)   # 25 x 25
        self.x3_res_chanel = nn.Conv2d(32, 64, 3, stride=1, padding=1) # increase the channel to 64

        self.x4 = Conv_block(64,64)
        self.x4_res = nn.AvgPool2d(2, stride=2, padding=1)  # 7 x 7
        self.x4_res_chanel = nn.Conv2d(64, 128, 3,  stride=1, padding=1)  # increase the channel to 128


        self.Dense = nn.Linear(32768, 7 * 7 * 128)
        self.sigma_entries = nn.Linear(7*7*128, 24)



        # decoder
        self.y = nn.Linear(7*7, 7 * 7 * 128)

        self.activ = nn.ReLU()
        self.activ2 = nn.Sigmoid()
        #self.Encoder_active = 0.5 + nn.ReLU()
        self.y1 = nn.ConvTranspose2d(128, 64, 3,  stride=2, padding=1)
        self.y2 = nn.ConvTranspose2d(64, 32, 3,  stride=2, padding=1)
        self.y3 = nn.ConvTranspose2d(32, 16, 3,  stride=2, padding=1)
        self.y4 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0)
        self.y5 = nn.ConvTranspose2d(8,1, 2, stride=1)




    def Sampling(self, sigma_entries):
        """
        :param Cov_matrix: covariance matrix of the random field
        """
        batch_size = 50
        sigma_entries = sigma_entries

        # DFT and DFT^-1
        DFT_mat = torch.zeros((49, 49), dtype=torch.cfloat)
        two_phi_i = 2 * math.pi * (complex(0, 1) / complex(49, 0))
        for j in range(0, 49):
            for k in range(0, 49):
                aa = cmath.exp(((two_phi_i) * j) * k)
                DFT_mat[j][k] = complex(round(aa.real), round(aa.imag))
        DFT_Mat_inv = torch.inverse(DFT_mat)

        sigma_entries = sigma_entries.detach().numpy()
        #print(sigma_entries)

        #sigma_mat = scipy.linalg.circulant(sigma_entries)
        sigma_mat = circulant(sigma_entries,49)
        sigma_mat = torch.tensor(sigma_mat)

        # eigenvalues matrix
        lambda_matrix = np.fft.ifft2(sigma_mat)


        eps_a = torch.rand((49, batch_size))
        eps_b = torch.rand((49, batch_size))

        zeta = torch.zeros((49, batch_size), dtype=torch.cfloat)
        zeta.real = eps_a
        zeta.imag = eps_b

        sqr_lambda_matrix = torch.tensor(scipy.linalg.sqrtm(lambda_matrix),dtype=torch.cfloat)


        q = torch.matmul(DFT_mat, sqr_lambda_matrix )

        qq = torch.matmul(q, zeta)
        Y = torch.real(qq)

        Y = torch.transpose(Y, 0, 1)


        return Y, sigma_mat


    def forward(self, x):
        # encoding
        x1 = self.x1(x)
        x1_res = x1 + x
        x1_res = self.x1_res(x1_res)
        x1_res_chanel = self.x1_res_chanel(x1_res)
        x1_res_chanel = self.activ(x1_res_chanel)

        x2 = self.x2(x1_res_chanel)
        x2_res = x2 + x1_res_chanel
        x2_res = self.x2_res(x2_res)
        x2_res_chanel = self.x2_res_chanel(x2_res)
        x2_res_chanel = self.activ(x2_res_chanel)

        x3 = self.x3(x2_res_chanel)
        x3_res = x3 + x2_res_chanel
        x3_res = self.x3_res(x3_res)
        x3_res_chanel = self.x3_res_chanel(x3_res)
        x3_res_chanel = self.activ(x3_res_chanel)

        x4 = self.x4(x3_res_chanel)
        x4_res = x4 + x3_res_chanel
        x4_res = self.x4_res(x4_res)
        x4_res_chanel = self.x4_res_chanel(x4_res)
        x4_res_chanel = self.activ(x4_res_chanel)

        flat = torch.flatten(x4_res_chanel)
        active_flat = self.activ(flat)

        Dense = nn.Linear( len(active_flat), 7 * 7 * 128)(active_flat)
        active_Dense = self.activ(Dense)
        Sigma_entries = self.sigma_entries(active_Dense)
        sigma_entries_activ = self.activ(Sigma_entries)


        z, embedded_Cov = self.Sampling(sigma_entries_activ)

        # decoder
        y = self.y(z)
        y = self.activ(y)
        reshape = torch.reshape(y, (batch_size, 128, 7, 7))
        y1 = self.y1(reshape)
        y1 = self.activ(y1)
        y2 = self.y2(y1)
        y2 = self.activ(y2)
        y3 = self.y3(y2)
        y3 = self.activ(y3)

        y4 = self.y4(y3)
        y4 = self.activ(y4)
        y5 = self.y5(y4)
        reconstruction = self.activ2(y5)
       #embedded_Cov = torch.view_as_real(embedded_Cov)
        return reconstruction, embedded_Cov

epochs = 100
model = VAE(1, 1)
fig, axs = plt.subplots(ncols=2)
batch_size = 50
lr = 0.001
torchsummary.summary(model, (1, 100, 100))
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCELoss(reduction="sum")
running_loss = 0.0


def final_loss(bce_loss, Cov_matrix):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    Recons = bce_loss
    #print(torch.sum(np.log(Cov_matrix)))
    KLD = -0.5 * torch.sum(1 + Cov_matrix- Cov_matrix.exp())
    return  KLD , Recons
yBCD_loss = []  # loss history
yKL_loss = []  # loss history
loss_ep =[]
figloss = plt.figure()
ax0 = figloss.add_subplot(121, title="lossRecons & LossKL")
#ax1 = figloss.add_subplot(122, title="Loss")




a = 0.01
for epoch in range(epochs):
    torch.cuda.empty_cache()
    gc.collect()
    batch_idx = 0

    torch.backends.cudnn.benchmark = True
    print("Training starts!")

    for batch_idx,(sample_batched, labels) in enumerate(data_loader):

        if batch_idx + 1 == len(data_loader):
            break

        inp_image = sample_batched.detach()

        #print("Batch shape is : ",inp_image.shape)
        optimizer.zero_grad()
        print("Inp_Min",inp_image.min())
        print("Inp_Max", inp_image.max())
        reconstructed_img, embedded_Cov = model(inp_image)
        print("generated Images MIN", reconstructed_img.min())
        print("generated Images MAX", reconstructed_img.max())

        reconstructed_img_one = (reconstructed_img[0, 0, :, :]).detach().cpu().numpy()  # red patch in upper left
        inp_img_one = (inp_image[0, 0, :, :]).detach().cpu().numpy()  # red patch in upper left
        axs[0].imshow(reconstructed_img_one, cmap='hot')
        axs[1].imshow(inp_img_one, cmap='hot')
        fig.savefig('sample.png')




        #print("Reconstructed Batch shape is",reconstructed_img.shape)
        bce_loss = criterion(reconstructed_img, inp_image)
        KLD,Recons = final_loss(bce_loss, embedded_Cov)


        loss = a*KLD + Recons
        print("KL divergence loss: ", KLD)
        a = 0.01 + a

        loss_ep.append(loss.detach().numpy())
        yBCD_loss.append(Recons.detach().numpy())
        yKL_loss.append(KLD.detach().numpy())
        #ax0.plot(yBCD_loss, 'bo-', label='RECONS')
        #ax0.plot( yKL_loss, 'ro-', label='KL')
        ax0.plot(loss_ep,'ro-',label='Loss')
        figloss.savefig('loss.jpg')


        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        #train_loss = running_loss / len(data_loader.dataset)
        print("=====================================================================")

        print(f'Epoch [{epoch}/{epochs}]  Batch {batch_idx + 1}/{len(data_loader)} \
                    Train Loss : {loss:.4f}')
        print("=====================================================================")













































