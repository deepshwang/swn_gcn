import argparse
import torch



###### MODEL CONFIGURATIONS #####

MODEL_CFGS={   'F': [[3, 64, 64], [64, 64, 64], [64, 128, 128], [128, 128, 128],
						 [128, 128, 128], [128, 256, 256], [256, 256, 256], [256, 256, 256], [256, 256, 256],
						 [256, 512, 512], [512, 512, 512], [512, 512, 512], [512, 512, 512]],

			   'G': [[3, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 128],
						 [128, 128, 128], [128, 128, 128], [128, 128, 128], [128, 256, 256], [256, 256, 256],
						 [256, 256, 256], [256, 256, 256], [256, 256, 256], [256, 256, 256], [256, 512, 512],
						 [512, 512, 512], [512, 512, 512]]
			   }

CLASSIFIER_CFGS = {'A': [512, 256, 100],
				   'B': [512, 256, 10]}

##### ARGUMENT PARSER #####

def argument_parser():
	parser = argparse.ArgumentParser(description='Argument parser for DnQNet training/testing')



	parser.add_argument('--use_cosface', type=bool,
						default=True,
						help='Whether to use cosface classifer for models')

	parser.add_argument('--train_dataset', type=str,
						default='MNIST',
						choices=['MNIST', 'RotNIST', 'CIFAR10', 'RotCIFAR10'],
						help='Dataset to train')

	parser.add_argument('--test_dataset', type=str,
						default='RotNIST',
						choices=['MNIST', 'RotNIST', 'CIFAR10', 'RotCIFAR10'],
						help='Dataset to test')

	parser.add_argument('--single_rotation_angle', type=int,
						default=0,
						help='Single rotation angel to test')

	parser.add_argument('--test_model_name', type=str,
					default='./data/saved_models/checkpoint.pth.tar',
					help='testing model state dict')


	parser.add_argument('--edge_neighbor', type=int,
						default=3,
						help='Set diameter for fixed radius near-neighbors search (kernel size)')

	parser.add_argument('--graph_agg_type', type=str,
						default='A',
						choices=['M', 'A'],
						help='Edge information aggregation type')

	parser.add_argument('--point_agg_type', type=str,
						default='A',
						choices=['M', 'A'],
						help='Point-net like vertex aggregation type')

	parser.add_argument('--m', type=float,
						default=0.4,
						help='Margin for Large Margin Cosine Loss')

	parser.add_argument('--optimizer', type=str,
						default='adam',
						choices=['adam', 'sgd'],
						help='optimizer type')


	parser.add_argument('--test_only', type=bool,
						default=False,
						help='Conduct testing only')

	parser.add_argument('--train_batch', type=int, 
						default=64,
						help='Train batch size')

	parser.add_argument('--test_batch', type=int, 
						default=64,
						help='Test & validation batch size')

	parser.add_argument('--epochs', type=int, 
						default=1000,
						help='Number of epochs for training')
	parser.add_argument('--num_workers', type=int,
						default=8,
						help='Number of workers')

	parser.add_argument('--batch_iter', type=int, 
						default=100,
						help='batch iteration size for logging')

	parser.add_argument('--device', type=str, 
						default='cuda' if torch.cuda.is_available() else 'cpu',
						help='Test batch size')


	parser.add_argument('--train_val_ratio', type=float,
					default=0.8,
					help='percentage of train data over the whole train dataset')

	parser.add_argument('--save_bestmodel_name', type=str,
					default='./data/saved_models/checkpoint.pth.tar',
					help='percentage of train data over the whole train dataset')

	parser.add_argument('--seed', type=int,
					default=77,
					help='Random seed to control stochasticity')


	parser.add_argument('--resume_training', type=bool,
						default=False,
						help='Whether to resume training from checkpoint')

	## Visualization arguments
	parser.add_argument('--viz_network', type=str,
					default='DnQ',
					choices=['DnQ', 'CNN', 'E2CNN'],
					help='percentage of train data over the whole train dataset')



	args = parser.parse_args()
	return args
