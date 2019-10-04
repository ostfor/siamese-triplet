import numpy as np
import random
import torch
from PIL import Image
from sknn_triplet_dataset.research.general_dataset import GeneralDataset
from sknn_triplet_dataset.tools.remote_dataset import load_test_dataset, load_train_dataset
from torch.utils.data import Dataset

cuda = torch.cuda.is_available()

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

batch_size = 256


def set_seed(seed_num=99999999999):  # 2147483647):
    return random.randint(0, seed_num)


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

    def __init__(self, train=True, transform=None, aug=None):

        self.transform = transform
        self.aug = aug
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

    def set_aug(self, aug):
        self.aug = aug

    def __getitem__(self, index):
        if self.train:

            augment = [self.aug, set_seed()]
            img1, label1 = self.general_dataset.get_data(index, augment)
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - {label1}))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2, _ = self.general_dataset.get_data(positive_index, augment)  # self.train_data[positive_index]
            img3, _ = self.general_dataset.get_data(negative_index, augment)  # [negative_index]
            # if self.aug != 0:
            #    img1, img2, img3 = aug_triple(img1, img2, img3, alpha)
        else:
            img1, _ = self.general_dataset.get_data(self.test_triplets[index][0])
            img2, _ = self.general_dataset.get_data(self.test_triplets[index][1])
            img3, _ = self.general_dataset.get_data(self.test_triplets[index][2])

        ims = [img1, img2, img3]
        if self.transform is not None:
            ims = [Image.fromarray(im, mode='RGB') for im in ims]
            for i in range(len(ims)):
                ims[i] = self.transform(ims[i])
        return tuple(ims), []
