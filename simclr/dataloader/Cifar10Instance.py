from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from augment import TransformsSimCLR

class Cifar10Instance(Dataset):
    def __init__(self):

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor()])
        
        self.cifar10 = datasets.CIFAR10(root='./train-data',
                                        download=True,
                                        train=True,
                                        transform=self.transform)
        
    def __getitem__(self, index):
        data, target = self.cifar10[index]        
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

"""
dataset = MyDataset()
loader = DataLoader(dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=1)
"""