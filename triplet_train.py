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


# Set up the network and training parameters

# mean, std = 0.28604059698879553, 0.35302424451492237
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

print(torch.cuda.get_device_name(0))

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


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


class TripletNetTexture(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def reinitialize_train_dataset(self):
        self.general_dataset = GeneralDataset(self.load_func())
        self.ds_labels = self.general_dataset.labels
        self.labels_set = set(self.ds_labels)
        self.label_to_indices = {label: np.where(self.ds_labels == label)[0]
                                 for label in self.labels_set}
        self.__len = len(self.general_dataset)

    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.train = train

        self.load_func = load_train_dataset if self.train else load_test_dataset
        self.reinitialize_train_dataset()
        self.__len = len(self.general_dataset)
        # self.transform = self.mnist_dataset.transform


        if not self.train:
            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.ds_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - {self.ds_labels[i]})
                                                 )
                                             ])
                         ]
                        for i in range(len(self.ds_labels))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.general_dataset.get_data(index)
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - {label1}))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2, _ = self.general_dataset.get_data(positive_index)  # self.train_data[positive_index]
            img3, _ = self.general_dataset.get_data(negative_index)  # [negative_index]
        else:
            img1, _ = self.general_dataset.get_data(self.test_triplets[index][0])
            # self.test_data[self.test_triplets[index][0]]
            img2, _ = self.general_dataset.get_data(self.test_triplets[index][1])
            # self.test_data[self.test_triplets[index][1]]
            img3, _ = self.general_dataset.get_data(self.test_triplets[index][2])
            # self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1, mode='RGB')
        img2 = Image.fromarray(img2, mode='RGB')
        img3 = Image.fromarray(img3, mode='RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return self.__len


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):

        return self.embedding_net(x)

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def imscatter(x, y, data, embs, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for image, coord in zip(data, embs):
        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, coord, xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

if __name__ == '__main__':
    triplet_train_dataset = TripletNetTexture(True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))  # Returns triplets of images

    triplet_test_dataset = TripletNetTexture(False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))

    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True,
                                                       **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False,
                                                      **kwargs)

    # Set up the network and training parameters
    margin = 1.
    embedding_net = EmbeddingNet()
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 500

    print(triplet_test_dataset[0])

    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
        project_root="./projects//net-tex-weights")