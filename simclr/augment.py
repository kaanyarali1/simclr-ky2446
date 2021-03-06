import torchvision


class TransformsSimCLR:
    """
    A stochastic data augmentation module to generate two correlated views of the same example. Called as positive pair
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter( #define color jitter hyperpatameter
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size), #randomly crop from the image and make it 32x32 for CIFAR-10
                torchvision.transforms.RandomHorizontalFlip(),  # make random horizontal flip with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size), # at test time you will not do any augmentation, only resize for 32x32
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x) #return two correlated views for SimCLR
