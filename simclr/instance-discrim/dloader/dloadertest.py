import torch
import torchvision
import torchvision.transforms as transforms


def dloaders_test(batch_size):
	tr = torchvision.transforms.Compose(
	            [
	                torchvision.transforms.ToTensor()])

	training_dateset = torchvision.datasets.CIFAR10(
	            root='./train-data',
	            train=True,
	            download=True,
	            transform=tr)
	training_loader = torch.utils.data.DataLoader(training_dateset,
	                    batch_size=len(training_dateset),
	                    shuffle=False,
	                    drop_last=True,
	                    num_workers=2)
	test_dataset = torchvision.datasets.CIFAR10(
	            root='./test-data',
	            train=False,
	            download=True,
	            transform=tr)

	test_loader = torch.utils.data.DataLoader(test_dataset,
	                    batch_size=batch_size,
	                    shuffle=False,
	                    drop_last=True,
	                    num_workers=2)
	_, training_labels  = next(iter(training_loader))

	return test_loader, training_labels