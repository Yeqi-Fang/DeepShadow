import torch
from torch.utils.data import Dataset


class MyData(Dataset):

    def __init__(self, dataset, labels, transform=None):
        assert len(dataset) == len(labels)
        self.transform = transform
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, idx):
        image = self.dataset[idx]
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)