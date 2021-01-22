from dataloader.dataloader import *
from params import argument_parser,  MODEL_CFGS, CLASSIFIER_CFGS
from models.models import *
import torch.nn as nn  #
import torch.optim as optim  # various optimization functions for model
from dataloader.transforms import *
from utils.loss_functions import apply_margin
import math

from torch.autograd import Variable
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import pdb
import warnings

warnings.filterwarnings("ignore")

class Trainer():
    def __init__(self, args, train_dataloader, test_dataloader, val_dataloader, model, criterion, optimizer):
        self.device = args.device
        print("Training with device: ", self.device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.best_loss = 0.0
        self.best_model = None
        self.stagger_count = 0

    def train(self):
        # Train
        for e in range(args.epochs):
            print("=== Epoch: ", e + 1, "/ ", args.epochs, " ===")
            self.model.train()
            loss_tracker = 0.0
            for i, datum in enumerate(self.train_dataloader, 0):
                inputs, labels = datum
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs, equi_feat, inv_feat = self.model(inputs)
                if args.use_cosface:
                    outputs = apply_margin(outputs, labels, args.m)

                loss = self.criterion(outputs, labels)


                loss.backward()


                for name, param in self.model.named_parameters():
                    if 'aggregate' in name:
                        b, c, w, h = param.shape
                        for x in range(w):
                            for y in range(h):
                                if x != int((w-1)/2) or y != int((h-1)/2):
                                    param.grad[:, :, x, y] = 0
                self.optimizer.step()


                loss_tracker += loss.item()

                if (i + 1) % args.batch_iter == 0:

                    # The loss calcluated here is sum of loss PER mini-batch (128)
                    print('[%d, %5d/%d] loss: %.3f' % (
                        e + 1, i + 1, int(len(self.train_dataloader.dataset) / args.train_batch),
                        loss_tracker / args.batch_iter))
                    loss_tracker = 0.0

            # Validation
            self.model.eval()

            per_instance_loss_tracker = 0.0
            for inputs, labels in self.val_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)


                outputs, equi_feat, inv_feat = self.model(inputs)


                loss = self.criterion(outputs, labels)


                per_instance_loss_tracker += loss.item()
            per_instance_loss_tracker /= (len(self.val_dataloader))
            print("This iteration's per-train-minibatch loss on validation: ",
                  per_instance_loss_tracker)

            if e == 0:
                self.best_loss = per_instance_loss_tracker
                torch.save(self.model.state_dict(), args.save_bestmodel_name)

            elif e > 0:
                if per_instance_loss_tracker < self.best_loss:
                    print("Val. loss improved: ", self.best_loss * args.train_batch, '-> ',
                          per_instance_loss_tracker * args.train_batch)
                    torch.save(self.model.state_dict(), args.save_bestmodel_name)
                    print("=== Model SAVED in ", args.save_bestmodel_name)
                    self.best_loss = per_instance_loss_tracker
                    self.stagger_count = 0
                    print("New validation achieved!")
                else:
                    self.stagger_count += 1
                    print("Not satisfied validation improvement. STAGGER COUNT: ", self.stagger_count)

            else:
                return



    def test(self, model_path=None):

        # model.eval() not implemented. See Section A. in supplementary material for detail.
        test_loss = 0
        correct = 0
        if args.single_rotation_angle is not None and not args.single_rotation_angle == 0:
            print("Testing over ", args.single_rotation_angle, " degree rotation augmented dataset")

        for i, (inputs, labels) in enumerate(self.test_dataloader, 0):


            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            output, equi_feat, inv_feat = self.model(inputs)


            pred = output.max(1, keepdim=True)[1]  # get the index of the max

            # Consider 6 and 9 as the same class
            if self.args.test_dataset == 'MNIST' or self.args.test_dataset == 'RotNIST':
                pred[pred==6] = 10
                pred[pred==9] = 10
                labels[labels==6] = 10
                labels[labels==9] = 10

            correct += pred.eq(labels.view_as(pred)).sum().item()

            test_loss /= len(self.test_dataloader.dataset)
        print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.
              format(correct, len(self.test_dataloader.dataset),
                     100. * correct / len(self.test_dataloader.dataset)))
        return correct


if __name__ == '__main__':
    args = argument_parser()
    torch.manual_seed(args.seed)

    ##############################
    ########  LOAD MODEL  ########
    ##############################
    reverse = False
    if args.test_dataset == 'RotCIFAR10' or args.test_dataset == 'CIFAR10':
        reverse = True
    model = WN_GCN(args, MODEL_CFGS['F'], CLASSIFIER_CFGS['B'], cosface=args.use_cosface, reverse=reverse)

    if args.resume_training and not args.test_only:
        print("Resuming Training!")
        print("Loading: ", args.resume_training)
        model.load_state_dict(torch.load(args.save_bestmodel_name))

    elif args.test_only:
        print("Testing only!")
        print("Loading for test: ", args.test_model_name)
        model.load_state_dict(torch.load(args.test_model_name))

    if args.train_dataset == 'MNIST':
        train_dataloader = MNISTDataloader(args, 'train', T_MNIST)
        val_dataloader = MNISTDataloader(args, 'val', T_MNIST)

    elif args.train_dataset == 'RotNIST':
        train_dataloader = MNISTDataloader(args, 'train', T_MNIST_ROT)
        val_dataloader = MNISTDataloader(args, 'val', T_MNIST_ROT)

    elif args.train_dataset == 'CIFAR10':
        train_dataloader = CIFAR10Dataloader(args, 'train', T_CIFAR10)
        val_dataloader = CIFAR10Dataloader(args, 'val', T_CIFAR10)

    if args.test_dataset == 'MNIST':
        test_dataloader = MNISTDataloader(args, 'test', T_MNIST)

    elif args.test_dataset == 'RotNIST':
        test_dataloader = MNISTDataloader(args, 'test', T_MNIST_ROT)

    elif args.test_dataset == 'CIFAR10':
        test_dataloader = CIFAR10Dataloader(args, 'test', T_CIFAR10)

    elif args.test_dataset == 'RotCIFAR10':
        test_dataloader = CIFAR10Dataloader(args, 'test', T_CIFAR10_ROT)

    ####################################
    #### Loss Function & Optimizers ####
    ####################################

    #
    criterion = nn.CrossEntropyLoss()

    #### Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam((filter(lambda p: p.requires_grad, model.parameters())), lr=1e-4) # 1e-4
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)




    print("<<<<<<<<<<<<<< SPECIFICATIONS >>>>>>>>>>>>>>>")
    print("Test only: ", args.test_only)
    print("Use cosface: ", args.use_cosface)
    if not args.test_only and args.use_cosface:
        print("Margin: ", args.m)
    if not args.test_only:
        print("Train Dataset: ", args.train_dataset)
    print("Test Dataset: ", args.test_dataset)



    trainer = Trainer(args=args,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      val_dataloader=val_dataloader,
                      model=model,
                      criterion=criterion,
                      optimizer=optimizer) # 0.0005

    if not args.test_only:
        trainer.train()
    trainer.test()

# # Check total number of parameters

