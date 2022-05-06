#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.utils import save_image
import torchvision.transforms as transforms

import os
import io
import numpy as np
from zipp import zipfile
from tqdm import tqdm
from time import sleep
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

#%% Default params
PATH_ZIP = "dataset/archive.zip"
DIR_OUT = "results/anime"

EPOCHS = 40
LR = 5e-4
BATCH_SIZE = 200
SPLIT_PERCENT = 0.9
LOG_INT = 10    # log interval

LAMBDA = 1
LAT_DIM = 10

if not os.path.exists(DIR_OUT):
    os.makedirs(DIR_OUT)

#%% Dataset
class AnimeDataset(Dataset):
    def __init__(self, path_file):
        self.images = []
        self.trans = transforms.ToTensor()

        z = zipfile.ZipFile(path_file)
        for file in z.namelist():
            if (".png" in file or ".PNG" in file):
                data = z.read(file)
                dataEnc = io.BytesIO(data)
                img = Image.open(dataEnc).convert("RGB")
                self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.trans(self.images[i])
        # data = data.unsqueeze(0)
        return img.to(device)

#%% model
class AnimeVAE(nn.Module):
    def __init__(self, shape, nLatent=2):
        super(AnimeVAE, self).__init__()
        self.shape = shape
        c, h, w = self.shape
        self.hh = (h-1)//2**3 + 1
        self.ww = (w-1)//2**3 + 1

        self.conv_layer = nn.Sequential(
            # (in_channels, out_channels, kernel_size, stride, padding)
            nn.Conv2d(c, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten()
            )   # (batch, 64*8*8)

        self.fc1 = nn.Sequential(
            nn.Linear(64*self.hh*self.ww, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            )   # (batch, 64)

        self.fc2_mean = nn.Linear(64, nLatent)
        self.fc2_logv = nn.Linear(64, nLatent)

        self.fc3 = nn.Sequential(
            nn.Linear(nLatent, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64*self.hh*self.ww),
            nn.ReLU()
            )

        self.deconv_layer = nn.Sequential(
            # (in_c, out_c, kernel_size, stride, padding, output_padding)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, c, 3, 2, 1, 1),
            nn.Tanh()
            )

    def encode(self, x):
        x = self.conv_layer(x)
        x = self.fc1(x)
        mean = self.fc2_mean(x)
        logv = self.fc2_logv(x)
        return mean, logv

    def decode(self, x):
        c, h, w = self.shape
        x = self.fc3(x)
        x = x.view(-1, 64, self.hh, self.ww)
        x = self.deconv_layer(x)
        return x

    def reparam(self, mu, logvar):
        sigma = torch.exp(0.5*logvar)
        z = torch.randn(size = (mu.size(0),mu.size(1)))
        z= z.type_as(mu)
        return mu + sigma*z

    def kl_calc(self, mu, logvar):
        return (-0.5*(1 + logvar - mu**2- torch.exp(logvar)).sum(dim = 1)).mean(dim =0)

    def forward(self, x):
        mu, logv = self.encode(x)               # (batch, nLatent)
        kl_loss = self.kl_calc(mu, logv)
        x = self.reparam(mu, logv)
        x = self.decode(x)
        return x, kl_loss


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

#%%
def train(zip_path):

    data_train = AnimeDataset(zip_path)
    num_train = int(len(data_train) * SPLIT_PERCENT)
    data_train, data_valid = random_split(data_train, [num_train, len(data_train) - num_train])
    # kl_weight_train, kl_weight_valid = BATCH_SIZE/len(data_train), BATCH_SIZE/len(data_valid)

    print("Train data: %d, Validation data: %d, Train batches: %.2f\n" %  \
          (len(data_train), len(data_valid), len(data_train)/BATCH_SIZE))

    trainloader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=False)

    net = AnimeVAE(shape=(3,64,64), nLatent=LAT_DIM)
    net.to(device)
    net.apply(init_weights)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(net.parameters(), lr=LR)

    sleep(0.3)
    train_loss_hist, valid_loss_hist = [], []
    # t = tqdm(range(EPOCHS), ncols=200, bar_format='{l_bar}{bar:15}{r_bar}{bar:-10b}', unit='epoch')
    for epoch in range(EPOCHS):

        t = tqdm(trainloader, ncols=200, bar_format='{l_bar}{bar:15}{r_bar}{bar:-10b}')
        t_desc = "Epoch %3d/%d" % (epoch + 1, EPOCHS)
        t.set_description(t_desc)

        # train
        net.train()
        train_loss = 0
        for idx, image in enumerate(t):
            optimizer.zero_grad()

            out, kl_loss = net(image)
            recon_loss = criterion(out, image)
            loss = kl_loss * LAMBDA + recon_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            rl_post = "%3.4f" % (recon_loss.item()/image.size(0))
            kl_post = "%3.4f" % (kl_loss.item()/image.size(0))
            tl_post = "%3.4f" % (loss.item()/image.size(0))
            t.set_postfix_str(f"Recon Loss: {rl_post}, KLD: {kl_post}, Total Loss: {tl_post}")
            t.update(0)

        # validation
        net.eval()
        valid_loss= 0
        with torch.no_grad():
            for batch_id, image in enumerate(validloader):
                out, kl_loss = net(image)
                recon_loss = criterion(out, image)
                loss2 = kl_loss * LAMBDA + recon_loss

                valid_loss += loss2.item()

                # validation images
                if (epoch+1) % LOG_INT == 0 and batch_id == 0:
                    img_list = out[:64]
                    out_filename = f"{DIR_OUT}/lambda_{LAMBDA}_epoch_{epoch+1}.png"
                    save_image(img_list, out_filename)

        train_loss = train_loss/len(data_train)
        valid_loss = valid_loss/len(data_valid)

        # tl_post = "%3.4f" % (train_loss)
        # vl_post = "%3.4f" % (valid_loss)
        # t.set_postfix({"T_Loss": tl_post, "V_Loss": vl_post})
        # t.update(0)

        train_loss_hist.append(train_loss)
        valid_loss_hist.append(valid_loss)

    # plot loss
    plt.figure()
    plt.plot(train_loss_hist, label="Train")
    plt.plot(valid_loss_hist, label="Valid")
    plt.title("Average Loss History")
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()

    return net

#%%
def interp_vectors(start, stop, ncols):
    steps = (1.0/(ncols-1)) * (stop - start)
    return np.vstack([(start + steps*x) for x in range(ncols)])

def generateImg(net, randn_mult=5, nrows=8):
    # fake images
    fake_in = np.random.randn(nrows**2, LAT_DIM)*randn_mult
    fake_in = torch.from_numpy(fake_in).to(device).float()
    fake_imgs = net.decode(fake_in)
    fake_filename = f"{DIR_OUT}/lambda_{LAMBDA}_fake.png"
    save_image(fake_imgs, fake_filename, nrow=nrows)

    # fake images interpolation
    if LAT_DIM == 2:
        a = np.array([-randn_mult,-randn_mult])
        b = np.array([randn_mult,-randn_mult])
        c = np.array([-randn_mult,randn_mult])
        d = np.array([randn_mult,randn_mult])
    else:
        a = np.random.randn(1, LAT_DIM)*randn_mult
        b = np.random.randn(1, LAT_DIM)*randn_mult
        c = np.random.randn(1, LAT_DIM)*randn_mult
        d = np.random.randn(1, LAT_DIM)*randn_mult

    r1, r2 = interp_vectors(a, b, nrows), interp_vectors(c, d, nrows)
    interp_in = torch.from_numpy(interp_vectors(r1, r2, nrows)).to(device).float()
    interp_out = net.decode(interp_in)
    interp_filename = f"{DIR_OUT}/lambda_{LAMBDA}_interp.png"
    save_image(interp_out, interp_filename, nrow=nrows)


#%% main
if __name__ == "__main__":
    net = train(PATH_ZIP)

    if str(device) == 'cuda':
        torch.cuda.empty_cache()

    generateImg(net, nrows=8, randn_mult=2)