import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import itertools

def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id<0 or cnn_id>2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model

class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """
    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model 
    (a number in [0, 1]) on the labeled data returned by 
    data_loader.
    """
    batches_num = len(data_loader)
    batches_acc = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        y_hat = torch.argmax(y_hat, dim=1)
        batches_acc += torch.sum(y_hat == y).float()/len(y)
    return batches_acc/batches_num    

def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    perturbed_xs = []
    labels = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        if targeted:
            t = (y+torch.randint(1, n_classes, size=(len(y),)))%n_classes
        perturbed_x = attack.execute(x, t if targeted else y, targeted)
        perturbed_xs.append(perturbed_x)
        labels.append(t if targeted else y)
    return torch.cat(perturbed_xs), torch.cat(labels)

def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    perturbed_xs = []
    labels = []
    n_queries = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        if targeted:
            t = (y+torch.randint(1, n_classes, size=(len(y),)))%n_classes
        perturbed_x, batch_n_queries = attack.execute(x, t if targeted else y, targeted)
        perturbed_xs.append(perturbed_x)
        labels.append(t if targeted else y)
        n_queries.append(batch_n_queries)
    return torch.cat(perturbed_xs), torch.cat(labels), torch.cat(n_queries)

def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    outputs = model(x_adv.to(device))
    outputs = torch.argmax(outputs, dim=1)
    if targeted:
        return torch.sum(outputs == y)/y.shape[0]
    elif (not targeted):
        return torch.sum(outputs != y)/y.shape[0]

def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    pass # FILL ME

def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    pass # FILL ME

def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    pass # FILL ME
