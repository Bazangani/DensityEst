import torch
import torchvision
import torch.optim as optim
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchsummary
import gc
from torch.utils.data import DataLoader
matplotlib.style.use('ggplot')
import math
import cmath
from sklearn import decomposition
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


batch_size = 65
# Data
# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((50, 50)),
])

dataset = torchvision.datasets.ImageFolder('C:/Users/farideh/Desktop/Rnadom field/VAE and RF/DATA',
                                                transforms.Compose([
                                                transforms.Grayscale(num_output_channels=1),
                                                transforms.Resize((50, 50)),
                                                #transforms.RandomRotation(180),
                                               #transforms.CenterCrop(200),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.1], std=[0.99]),

                                                ]))

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          )

#print("Number of samples:", len(data_loader))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.2)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# define a simple linear VAE
class VAE(nn.Module):
    def __init__(self, inp, out):
        super(VAE, self).__init__()

        self.y2 =  nn.ConvTranspose2d(8, 16, 3,  stride=2)
        self.y3 = nn.ConvTranspose2d(16, 32, 3,  stride=2)
        self.y4 = nn.ConvTranspose2d(32, 64, 3, stride=1)
        self.y5 = nn.ConvTranspose2d(64,1,2,stride=2)




        self.activ = nn.LeakyReLU()
        self.activ2 = nn.Sigmoid()


######################################### sampling ####################################

    def Sampling(self, eighenvalues):
        """
        :param Cov_matrix: covariance matrix of the random field
        """
        batch_size = 65

        DFT_mat = torch.zeros((64, 64), dtype=torch.cfloat)
        two_phi_i = 2 * math.pi * (complex(0, 1) / complex(64, 0))
        for j in range(0, 64):
            for k in range(64):
                aa = cmath.exp(((two_phi_i) * j) * k)
                DFT_mat[j][k] = complex(round(aa.real), round(aa.imag))
        DFT_mat = torch.nn.functional.normalize(DFT_mat)
        DFT_mat_transpose = torch.t(DFT_mat)

        DFT_Mat_inv = torch.conj(DFT_mat_transpose)
        DFT_Mat_inv = torch.nn.functional.normalize(DFT_Mat_inv)
        DFT_Mat_inv = DFT_Mat_inv.type(torch.cfloat)
        DFT_mat = DFT_mat.type(torch.cfloat)
        torch.save(DFT_mat, 'DFT_mat.pt')
        torch.save(DFT_Mat_inv,'DFT_Mat_inv.pt')
        DFT_mat=torch.load('DFT_mat.pt')
        DFT_Mat_inv = torch.load('DFT_Mat_inv.pt')

        # eighenvalues [2,64]
        COVAR_batch= torch.zeros((batch_size, 64, 64))
        Sample_batch = torch.zeros((batch_size, 64, 64))
        for i in range(0,batch_size):

            eig = torch.tensor(eighenvalues[i, :])
           # print(eig)

            #print(eig.mean())
            #print(eig.std())
            sigma_matrix= torch.diag(eig).type(torch.cfloat)
            sigma_matrix_2 = torch.sqrt(sigma_matrix)
            sigma_matrix_2 = torch.nn.functional.normalize(sigma_matrix_2).type(torch.cfloat)

            # # compute the covariance atrix B = F-1*D*F
            a = torch.matmul(DFT_mat, sigma_matrix)
            cov = torch.real(torch.matmul(a, DFT_Mat_inv))
            #cov = torch.real(embedded_cov) # the multiplication should be a real tensor

            eps_a = torch.rand((64, 64))
            eps_b = torch.rand((64, 64))


            zeta = torch.zeros((64, 64), dtype = torch.cfloat)
            zeta.real = eps_a
            zeta.imag = eps_b
            q = torch.matmul(DFT_mat, sigma_matrix_2)
            qq = torch.matmul(q, zeta)
            Y = torch.imag(qq)
            sigma_matrix = sigma_matrix.real

            COVAR_batch[i,:,:] = cov
            Sample_batch [i,:,:] = Y




        return Sample_batch, COVAR_batch, sigma_matrix,DFT_Mat_inv
        ########################################################################################

    def forward(self, x):
        # encoder

        eighenvalues_activ =x

        z, embedded_Cov, sigma_matrix,DFT_mat = self.Sampling(eighenvalues_activ)

        # decoder
        z_flatten = torch.flatten(z)
        z1_linear = nn.Linear(batch_size*64*64, batch_size*8*5*5)(z_flatten)
        z1_reshape = z1_linear.view((batch_size, 8, 5, 5))

        y2 = self.y2(z1_reshape)

        y2 = self.activ(y2)

        y3 =self.y3(y2)

        y3 = self.activ(y3)

        y3 = self.y4(y3)
        y3 =self.activ(y3)

        y4 =self.y5(y3)
        print(y4.shape)
        reconstruction = self.activ2(y4)

        return reconstruction, embedded_Cov, sigma_matrix, z , DFT_mat

epochs = 100
model = VAE(1, 1).to(device)
model.apply(weights_init)
fig, axs = plt.subplots(ncols=2)

lr = 0.01
#torchsummary.summary(model, (1, 50, 50))

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion =torch.nn.BCELoss(size_average=True)
running_loss = 0.0


def final_loss(bce_loss, Cov_matrix):

    BCE = bce_loss

    KLD = torch.mean(0.5*(Cov_matrix.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 2 -
                          torch.log(torch.det(Cov_matrix)+0.05*torch.ones(65))), dim=0)

    #KLD = torch.mean(-0.5 * torch.sum(torch.sum(1 + Cov_matrix - Cov_matrix.exp(), dim=2), dim=1), dim=0)

    return  KLD , BCE


loss_ep =[]

figcov, axscov = plt.subplots(1, 3)
figloss, ax0 = plt.subplots(ncols=2)
a= 0.001

pca = decomposition.PCA(n_components=64)
eig_std=[]
eig_mean=[]
for epoch in range(epochs):
    torch.cuda.empty_cache()
    gc.collect()
    batch_idx = 0

    torch.backends.cudnn.benchmark = True
    for batch_idx,(sample_batched, labels) in enumerate(data_loader):
        if a != 1:
           a = 0.001 + a

        if batch_idx + 1 == len(data_loader):
            break

        inp_image_one = sample_batched.detach().to(device)
        print(inp_image_one.shape)

        inp_image = torch.flatten(inp_image_one, start_dim=1,end_dim=3)
        pca.fit(inp_image)
        eighenvalues_activ = pca.transform(inp_image)

        eig_mean.append(eighenvalues_activ.mean().item())
        eig_std.append(eighenvalues_activ.std().item())
        #print(eighenvalues_activ.shape)



        optimizer.zero_grad()

        reconstructed_img, embedded_Cov, sigma_matrix, z,DFT_mat = model(eighenvalues_activ)

        axscov[0].imshow(embedded_Cov[0, :, :].detach().cpu().numpy().reshape(64, 64),cmap = 'autumn' , interpolation = 'nearest')
        axscov[0].set_title("Covariance")

        axscov[1].imshow((z[0,:,:].detach().cpu().numpy().reshape(64, 64)), cmap='hot')
        axscov[1].set_title("sample")
        axscov[2].imshow((DFT_mat.real.detach().cpu().numpy()), cmap='twilight')
        axscov[2].set_title("DFT_conj_Tran")
        #axscov[3].imshow(x4_res_chanel[0, 30, :, :].detach().cpu().numpy().reshape(5, 5), cmap='twilight')
        #axscov[3].set_title("sample")
        figcov.savefig('Covariance.png')


        reconstructed_img_one = (reconstructed_img[0, 0, :, :]).detach().cpu().numpy()  # red patch in upper left
        inp_im = (inp_image_one[0, 0, :, :]).detach().cpu().numpy()  # red patch in upper left
        axs[0].imshow(reconstructed_img_one, cmap='gray')
        axs[1].imshow(inp_im, cmap='gray')
        fig.savefig('sample.png')
        print(reconstructed_img.shape)
        print(inp_image_one.shape)
        bce_loss = criterion(reconstructed_img, inp_image_one)
        KLD, BCE = final_loss(bce_loss, embedded_Cov)

        print("The KL loss value:", KLD)

        loss =  a*KLD + BCE

        loss_ep.append(loss.detach().numpy())

        ax0[0].plot(loss_ep, 'ro-', label='Loss')
        ax0[1].plot(np.array(eig_mean), color='green', label='Mean_eig')

        figloss.savefig('loss.jpg')
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        print("=====================================================================")

        print(f'Epoch [{epoch}/{epochs}]  Batch {batch_idx + 1}/{len(data_loader)} \
                    Train Loss : {loss:.4f}')
        print("=====================================================================")












































