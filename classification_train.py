from sknn_triplet_dataset.research.pytorch_dataset import TripletNetTexture
from torchvision import transforms
from torchvision.datasets import FashionMNIST

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from sknn_triplet_dataset.tools.remote_dataset import load_test_dataset, load_train_dataset
from sknn_triplet_dataset.research.general_dataset import GeneralDataset

from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit

from metrics import AccumulatedAccuracyMetric
import torch.nn as nn
import torch.nn.functional as F

ConvOUT = 8, 8


cuda = torch.cuda.is_available()

# Set up data loaders
batch_size = 256


# Set up the network and training parameters

#mean, std = 0.28604059698879553, 0.35302424451492237
mean = (0.5,0.5,0.5)
std = (0.5,0.5,0.5)

batch_size = 256

print (torch.cuda.get_device_name(0))

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


class ClassificationNetTexture(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.train = train
        self.curr_dataset = None
        self.load_func = load_train_dataset if self.train else load_test_dataset
        self.reinitialize_train_dataset()
        self.__len = len(self.curr_dataset)

    def reinitialize_train_dataset(self):
        self.curr_dataset = GeneralDataset(self.load_func())
        print("Min Label: ", np.min(self.curr_dataset.labels))
        print("Max Label: ", np.max(self.curr_dataset.labels))
        self.__len = len(self.curr_dataset)

    def __getitem__(self, index):
        img1, label1 = self.curr_dataset.get_data(index)

        # print(img1.shape)
        img1 = Image.fromarray(img1, mode='RGB')
        # img1 = img1.convert('L')
        if self.transform is not None:
            img1 = self.transform(img1)
        return img1, label1

    def __len__(self):
        return self.__len



train_ds = ClassificationNetTexture(True,  transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                            ]))

test_ds = ClassificationNetTexture(False,  transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                            ]))

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)

print("Len: ", len(train_ds))
print(train_ds[0])
print(np.min(train_ds[1][0].numpy()), np.max(train_ds[0][0].numpy()))


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * ConvOUT[0] * ConvOUT[1], 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        # print ("OUTPUT: ", output.shape)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        # print (output.shape)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        # print (scores.shape)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))



if __name__ == '__main__':
    embedding_net = EmbeddingNet()

    n_classes = np.max(train_ds.curr_dataset.labels) + 1 #453
    model = ClassificationNet(embedding_net, n_classes=n_classes)
    if cuda:
        model.cuda()
    loss_fn = torch.nn.NLLLoss()
    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 50

    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()], project_root="./projects/net-tex-weights")