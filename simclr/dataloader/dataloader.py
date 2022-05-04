import torchvision
import torch
from augment import TransformsSimCLR

def train_loader_simclr(dataset,batch_size):

        if dataset == "CIFAR10":

                train_dataset = torchvision.datasets.CIFAR10(
                            root='./train-data',
                            train=True,
                            download=True,
                            transform=TransformsSimCLR(size=32)
                        )
                train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=2
                    )
                return train_loader
        else:
                pass
        
def test_loader(dataset,batch_size):

        if dataset == "CIFAR10":

                test_dataset = torchvision.datasets.CIFAR10(
                    root='./test-data',
                    train=False,
                    download=True,
                    transform=TransformsSimCLR(size=32).test_transform)
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True,
                    num_workers=2
                )

                return test_loader
        else:
                pass

def get_testimgs_list(dataset):
        if dataset == "CIFAR10":
                test_dataset = torchvision.datasets.CIFAR10(
            root='./test-data',
            train=False,
            download=True,
            transform=TransformsSimCLR(size=32).test_transform)

                test_loader_list = torch.utils.data.DataLoader(
                    test_dataset,
                            batch_size=len(test_dataset),
                            shuffle=False,
                            drop_last=True,
                            num_workers=2
                        )
                test_images, test_labels = next(iter(test_loader_list))
                return test_images, test_labels
        else:
                pass

def train_loader_dstream(dataset,batch_size):
        if dataset == "CIFAR10":

                train_dataset = torchvision.datasets.CIFAR10(
                            root='./train-data',
                            train=True,
                            download=True,
                            transform=TransformsSimCLR(size=32).test_transform
                        )
                train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=2
                    )
                return train_loader
        else:
                pass