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
import numpy as np
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

#%% Default params
PATH_TRAIN = "dataset/mnist.npz"
DIR_OUT = "results/mnist"

BATCH_SIZE = 600
EPOCHS = 200
LR = 1e-3
SPLIT_PERCENT = 0.9
LOG_INT = 50    # log interval

LAMBDA = 1
LAT_DIM = 2

if not os.path.exists(DIR_OUT):
    os.makedirs(DIR_OUT)

#%% Dataset
class trainDataset(Dataset):
    def __init__(self, images):
        self.images = [(img > 128)*1.0 for img in images]
        self.trans = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.trans(self.images[i]).float()
        img = torch.flatten(img)
        return img.to(device)

#%% model
class MnistVAE(nn.Module):
    def __init__(self, nLatent=2):
        super(MnistVAE, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(784, 196),
            nn.BatchNorm1d(196),
            nn.ReLU(),
            nn.Linear(196, 49),
            nn.BatchNorm1d(49),
            nn.ReLU(),
            )

        self.fc2_mu = nn.Linear(49, nLatent)
        self.fc2_lo = nn.Linear(49, nLatent)

        self.fc3 = nn.Sequential(
            nn.Linear(nLatent, 49),
            nn.ReLU(),
            nn.Linear(49, 196),
            nn.ReLU(),
            nn.Linear(196, 784),
            nn.Sigmoid()  # use with BCELoss
            )

    def encoder(self, x):
        x = self.fc1(x)
        x1 = self.fc2_mu(x)
        x2 = self.fc2_lo(x)
        return x1, x2

    def reparam(self, mu, logvar):
        sigma = torch.exp(0.5*logvar)
        z = torch.randn_like(sigma)
        return mu + sigma*z

    def kl_calc(self, mu, logvar):
        return (-0.5*(1 + logvar - mu**2 - torch.exp(logvar)).sum(dim=1)).mean(dim=0)

    def decoder(self, x):
        return self.fc3(x)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        kl_loss = self.kl_calc(mu, logvar)
        x = self.reparam(mu, logvar)
        x = self.decoder(x)
        return x, kl_loss

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

#%%
def train(npz_data):

    data_train = trainDataset(npz_data['x_train'])
    num_train = int(len(data_train) * SPLIT_PERCENT)
    data_train, data_valid = random_split(data_train, [num_train, len(data_train) - num_train])
    # kl_weight_train, kl_weight_valid = BATCH_SIZE/len(data_train), BATCH_SIZE/len(data_valid)

    print("Train data: %d, Validation data: %d, Train batches: %.2f\n" %  \
          (len(data_train), len(data_valid), len(data_train)/BATCH_SIZE))

    trainloader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=False)

    net = MnistVAE(nLatent=LAT_DIM)
    net.apply(init_weights)
    net.to(device)
    # criterion = nn.BCEWithLogitsLoss(reduction='sum')
    criterion = nn.BCELoss(reduction='sum')     # last layer needs sigmoid
    optimizer = optim.Adam(net.parameters(), lr=LR)

    sleep(0.3)
    train_loss_hist, valid_loss_hist = [], []
    t = tqdm(range(EPOCHS), ncols=200, bar_format='{l_bar}{bar:15}{r_bar}{bar:-10b}', unit='epoch')
    for epoch in t:

        # train
        net.train()
        train_loss = 0
        for batch_id, image in enumerate(trainloader):
            optimizer.zero_grad()

            out, kl_loss = net(image)
            recon_loss = criterion(out, image)
            loss = kl_loss * LAMBDA + recon_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

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
                    img_list = img_list.reshape(-1, 1, 28, 28)
                    # grid_img = make_grid(img_list, nrow=8).permute(1,2,0)
                    out_filename = f"{DIR_OUT}/lambda_{LAMBDA}_epoch_{epoch+1}.png"
                    save_image(img_list, out_filename)

        train_loss = train_loss/len(data_train)
        valid_loss = valid_loss/len(data_valid)

        tl_post = "%3.4f" % (train_loss)
        vl_post = "%3.4f" % (valid_loss)
        t.set_postfix({"T_Loss": tl_post, "V_Loss": vl_post})
        t.update(0)

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
    fake_imgs = net.decoder(fake_in)
    fake_imgs = fake_imgs.reshape(-1, 1, 28, 28)
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
    interp_out = net.decoder(interp_in)
    interp_out = interp_out.reshape(-1, 1, 28, 28)
    interp_filename = f"{DIR_OUT}/lambda_{LAMBDA}_interp.png"
    save_image(interp_out, interp_filename, nrow=nrows)


#%% main
if __name__ == "__main__":
    mnist_npz = np.load(PATH_TRAIN)

    net = train(mnist_npz)

    if str(device) == 'cuda':
        torch.cuda.empty_cache()

    generateImg(net, nrows=10, randn_mult=12)