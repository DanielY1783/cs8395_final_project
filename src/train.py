# train.py
# Matthew C. Sedam
# CS 8395 Deep Learning in Medical Image Processing
# Base code from: https://github.com/pytorch/examples/blob/master/mnist/main.py
# Unet from: https://github.com/milesial/Pytorch-UNet

import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
# from torchvision.models import densenet201
from .unet import UNet
from .util import CTImageDataset, get_args, TR_DATA, TR_DATA_LABELS, VAL_DATA, VAL_DATA_LABELS

LOSS_LOG_FILE_NAME = 'loss.txt'
MODEL_LOWEST_VAL_SAVE_FILE_NAME = 'best_model.pt'
MODEL_LOWEST_VAL_LOSS_SAVE_FILE_NAME = 'best_model_val_loss.txt'
MODEL_SAVE_FILE_NAME = 'model.pt'
VALIDATION_DATA_PROPORTION = 0.25


def train_helper(args, model, device, optimizer, loss_func, epoch, num_total, num_batches,
                 batch_idx,
                 batch_data, batch_target):
    """
    Helper function for training
    :param args: the arguments
    :param model: the model
    :param device: the device
    :param optimizer: the optimizer
    :param loss_func: the loss
    :param epoch: the current epoch
    :param num_total: the total number of instances
    :param num_batches: the total number of batches
    :param batch_idx: the batch index
    :param batch_data: the batch data
    :param batch_target: the batch target
    :return: None
    """

    batch_data, batch_target = batch_data.to(device), batch_target.to(device)
    # batch_target = torch.max(batch_target, 1)[1]
    optimizer.zero_grad()
    output = model(batch_data)
    loss = loss_func(output, batch_target)
    loss.backward()
    optimizer.step()
    print('Train Epoch: {} [{:4d}/{:4d} ({:3.1f}%)]\tLoss: {:.6f}'
          .format(epoch, batch_idx * args.batch_size, num_total,
                  batch_idx * 100 / num_batches, loss.item()))
    return loss.item() * batch_data.shape[0]


def train(args, model, device, data_loader, optimizer, loss_func, epoch):
    """
    Trains the model
    :param args: the arguments
    :param model: the model
    :param device: the device
    :param data_loader: the data loader
    :param optimizer: the optimizer
    :param epoch: the epoch number
    :param loss_func: the loss
    :return: the mean training loss
    """

    model.train()
    total_loss = 0
    total_instances = 0
    for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
        total_loss += train_helper(args, model, device, optimizer, loss_func, epoch,
                                   len(data_loader.dataset),
                                   len(data_loader.dataset) // args.batch_size,
                                   batch_idx, batch_data, batch_target)
        total_instances += batch_data.shape[0]

    return total_loss / total_instances


def validation(args, model, device, validation_loader, loss_func=torch.nn.CrossEntropyLoss()):
    """
    Performs validation calculations.
    :param args: the arguments
    :param model: the model
    :param device: the device
    :param validation_loader: the validation loader
    :param loss_func: the loss function
    :return: the mean validation loss, number correct, total validation instances
    """

    # log_softmax = torch.nn.LogSoftmax(dim=1)
    model.eval()
    val_loss = 0
    num_correct = 0
    total_dice = 0
    total_val = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            # target = torch.max(target, 1)[1]
            output = model(data)
            val_loss += loss_func(output, target).item() * data.shape[0]
            total_val += data.shape[0]

            # compute accuracy
            # output = torch.exp(log_softmax(output))
            # output = torch.max(output, 1)[1]
            # num_correct += torch.sum((output == target) * 1).item()

            # compute dice
            output = output.detach().cpu().numpy()
            output = (output >= 0.5) * 1  # make binary
            target = target.detach().cpu().numpy()
            target = (target >= 0.5) * 1  # make binary

            for out_img, target_img in zip(output, target):
                dice = (2.0 * np.sum(np.multiply(out_img, target_img)) + 1) / (np.sum(out_img) + np.sum(target_img) + 1)
                total_dice += dice

        val_loss /= total_val
        print('\nValidation set: Average loss: {:.4f}: Average dice: {:.4f}\n'.format(val_loss, total_dice / total_val))

        return val_loss, num_correct, total_val


def main():
    # setup
    args = get_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')

    # get data
    tr_data = CTImageDataset(image_dir=TR_DATA, seg_image_dir=TR_DATA_LABELS)
    val_data = CTImageDataset(image_dir=VAL_DATA, seg_image_dir=VAL_DATA_LABELS)
    data_loader = DataLoader(tr_data,
                             batch_size=args.batch_size,
                             num_workers=0,
                             shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            num_workers=0,
                            shuffle=False)

    # setup pipeline
    # Unet
    model = UNet(1, 1, bilinear=False)
    model = model.to(device)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_FILE_NAME, map_location=device))
        print('Successfully loaded model from file: ' + MODEL_SAVE_FILE_NAME)
    except:
        print('Could not load model from file: ' + MODEL_SAVE_FILE_NAME)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss_func = torch.nn.BCELoss()

    # train
    best_val_loss = None
    best_model_state_dict = None
    try:
        for epoch in range(1, args.epochs + 1):
            tr_loss = train(args, model, device, data_loader, optimizer, loss_func, epoch)
            val_loss, val_num_correct, val_total = validation(args, model, device, val_loader,
                                                              loss_func)
            with open(LOSS_LOG_FILE_NAME, 'a') as file:
                file.write(str(epoch) + ',' + str(tr_loss) + ',' + str(val_loss) + ',' +
                           str(val_num_correct) + ',' + str(val_total) + '\n')

            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = deepcopy(model.state_dict())

            scheduler.step()
    except KeyboardInterrupt:
        pass
    finally:
        torch.save(model.state_dict(), MODEL_SAVE_FILE_NAME)
        torch.save(best_model_state_dict, MODEL_LOWEST_VAL_SAVE_FILE_NAME)
        with open(MODEL_LOWEST_VAL_LOSS_SAVE_FILE_NAME, 'w') as file:
            file.write(str(best_val_loss) + '\n')
