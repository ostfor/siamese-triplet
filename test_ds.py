import numpy as np
import torch
from PIL import Image
from sknn_triplet_dataset.research.general_dataset import GeneralDataset
from sknn_triplet_dataset.tools.remote_dataset import load_test_dataset
from torch.utils.data import Dataset


from torchvision import transforms


def get_test_data( mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = TestDs(False, transform=transforms.Compose(transform))
    test_data = [test_dataset[i] for i in range(0, 20)]
    test_data = torch.stack(test_data).cuda()
    test_data_cpu = test_data.cpu().numpy().transpose([0, 2, 3, 1])
    return test_data, test_data_cpu

class TestDs(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.train = train

        self.general_dataset = GeneralDataset(load_test_dataset())
        self.ds_labels = self.general_dataset.labels
        self.labels_set = set(self.ds_labels)
        self.label_to_indices = {label: np.where(self.ds_labels == label)[0]
                                 for label in self.labels_set}
        self.__len = len(self.general_dataset)

    def __getitem__(self, index):
        img1, label1 = self.general_dataset.get_data(index)
        img1 = Image.fromarray(img1, mode='RGB')

        if self.transform is not None:
            img1 = self.transform(img1)

        return img1

    def __len__(self):
        return self.__len
