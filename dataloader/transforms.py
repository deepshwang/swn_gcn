import torch
import torchvision
from params import argument_parser
import torchvision.transforms.functional as TF
import PIL


class CircularMask(object):
    """
    Apply circular mask to the input
    """
    def __call__(self, x):
        mask = self.make_mask(16)

        x = x * mask
        return x

    def __repr__(self):
        return self.__class__.__name__+'()'

    def make_mask(self, R):
        s = int(2 * R)
        mask = torch.zeros(1, s, s, dtype=torch.float32)
        c = (s-1) / 2
        for x in range (s):
            for y in range(s):
                r = (x - c) ** 2 + (y - c) ** 2
                if r > ((R-2) ** 2):
                    mask[..., x, y] = 0
                else:
                    mask[..., x, y] = 1
        active = torch.count_nonzero(mask)
        return mask


class Rotation(object):
    """Rotate by one of the given angles."""

    def __init__(self, angle, resample=PIL.Image.BILINEAR):
        self.angle = angle
        self.resample = resample

    def __call__(self, x):
        return TF.rotate(x, self.angle, expand=True, resample=self.resample)

    def __repr__(self):
        return self.__class__.__name__+'()'

class expandChannel(object):
    """
    Expand grayscale image to 3 channel format
    """

    def __call__(self, x):
        x = x.repeat(3, 1, 1)
        return x

    def __repr__(self):
        return self.__class__.__name__+'()'

args = argument_parser()

# INTER_METHOD = PIL.Image.NEAREST
INTER_METHOD = PIL.Image.BILINEAR


T_MNIST = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Resize((32, 32)),
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                          CircularMask(),
                                          expandChannel()])

T_MNIST_ROT = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Resize((32, 32), interpolation=INTER_METHOD),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                              torchvision.transforms.RandomRotation(degrees=(-180, 180), expand=True, resample=INTER_METHOD),
                                              # Rotation(args.single_rotation_angle, resample=INTER_METHOD),
                                              torchvision.transforms.CenterCrop(32),
                                              CircularMask(),
                                              expandChannel()])




T_CIFAR10 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    CircularMask(),
])

T_CIFAR10_ROT = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # Rotation(args.single_rotation_angle, resample=PIL.Image.NEAREST),
    torchvision.transforms.RandomRotation(degrees=(-180, 180), expand=True, resample=PIL.Image.NEAREST),
    torchvision.transforms.CenterCrop(32),
    CircularMask(),
])