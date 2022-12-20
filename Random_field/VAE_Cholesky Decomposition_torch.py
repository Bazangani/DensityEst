import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchsummary
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from subpixel_conv2d import SubpixelConv2D
matplotlib.style.use('ggplot')


# Data

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((100, 100)),
])

dataset = torchvision.datasets.ImageFolder('C:/Users/farideh/Desktop/Rnadom field/VAE and RF/DATA/',
                                                transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((100, 100)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=4,
                                          shuffle=True,
                                          )

print(len(data_loader))


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


# define a simple linear VAE
class VAE(nn.Module):
    def __init__(self):
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

        ### SPD METHOD FOR THE COV MATRIX
        self.Conv_one_one = nn.Conv2d(128, 128, 1)



        self.Cov_matrix = nn.Conv2d(128, 1, 3, stride=1, padding=1)  # 7 x 7 @1 channel


        # decoder
        self.y = nn.Linear(7*7,7 * 7 * 128)

        self.activ = nn.ReLU()
        self.y1 = nn.ConvTranspose2d(128, 64, 3,  stride=2, padding=1)
        self.y2 = nn.ConvTranspose2d(64, 32, 3,  stride=2, padding=1)
        self.y3 = nn.ConvTranspose2d(32, 16, 3,  stride=2, padding=1)
        self.y4 = nn.Conv2d(32, 16, 7, stride=1, padding=0)
        self.y5 = nn.Conv2d(16, 1, 7, stride=1, padding=0)

    def Sampling(self, Cov_matrix):
        """
        :param Cov_matrix: covariance matrix of the random field
        """
        Cov_matrix = Cov_matrix
        eps = torch.randn_like(Cov_matrix)  # `randn_like` as we need the same size
        L = torch.linalg.cholesky(Cov_matrix)
        sample = Cov_matrix * eps

        return sample, Cov_matrix

    def forward(self, x):
        # encoding
        x1 = self.x1(x)
        x1_res = x1 + x
        x1_res = self.x1_res(x1_res)
        x1_res_chanel = self.x1_res_chanel(x1_res)
        x1_res_chanel = self.activ(x1_res_chanel)

        x2 =  self.x2(x1_res_chanel)
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

        # SPD
        x4_one_one = self.Conv_one_one(x4_res_chanel)




        Cov_matrix = self.Cov_matrix(x4_res_chanel)
        Cov_matrix = self.activ(Cov_matrix)
        z, Cov_matrix = self.Sampling(Cov_matrix)

        # decoder
        y = self.y(z)
        y = self.activ(y)
        reshape = torch.reshape(y,(128,7,7))
        y1 = self.y1(reshape)
        y1 =self.activ(y1)
        y2 = self.y2(y1)
        y2 = self.activ(y2)
        y3 = self.y3(y2)
        y3 = self.activ(y3)

        y4 = self.y4(y3)
        y4 = self.activ(y4)
        y5 = self.y5(y4)
        reconstruction = torch.sigmoid(y5)
        return reconstruction, Cov_matrix

# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int,
                    help='number of epochs to train the VAE for')
args = vars(parser.parse_args())

# leanring parameters
epochs = args['epochs']
batch_size = 64
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = VAE()
torchsummary.summary(model, (1, 100, 100))
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

def final_loss(bce_loss, Cov_matrix):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + Cov_matrix - Cov_matrix.exp())
    return BCE + KLD


def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(data_loader)/dataloader.batch_size)):
        data, _ = data
        data = data
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss