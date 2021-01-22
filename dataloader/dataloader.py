from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler



def MNISTDataloader(args, mode, T):

	# Set batch-size
	if mode == 'train':
		batch_size = args.train_batch
	elif mode == 'test' or mode == 'val':
		batch_size = args.test_batch

	# Load dataset
	dataset = datasets.MNIST(root='./data', train=(mode == 'train' or mode =='val'),
							 download=True, transform=T)

	indices = list(range(len(dataset)))
	split = int(args.train_val_ratio*len(dataset))

	if mode == 'train':
		sampler = SubsetRandomSampler(indices[:split])
		shuffle = False
	elif mode == 'val':
		sampler = SubsetRandomSampler(indices[split:])
		shuffle = False
	else:
		sampler = None
		shuffle = False

	dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
							sampler=sampler,
							num_workers=args.num_workers,
							shuffle=shuffle)

	return dataloader

def CIFAR10Dataloader(args, mode, T):

	# Set batch-size
	if mode == 'train':
		batch_size = args.train_batch
	elif mode == 'test' or mode == 'val':
		batch_size = args.test_batch

	# Load dataset
	dataset = datasets.CIFAR10(root='./data', train=(mode == 'train' or mode =='val'),
							 download=True, transform=T)

	indices = list(range(len(dataset)))
	split = int(args.train_val_ratio*len(dataset))

	if mode == 'train':
		sampler = SubsetRandomSampler(indices[:split])
		shuffle = False
	elif mode == 'val':
		sampler = SubsetRandomSampler(indices[split:])
		shuffle = False
	else:
		sampler = None
		shuffle = True

	dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
							sampler=sampler,
							num_workers=args.num_workers,
							shuffle=shuffle)

	return dataloader