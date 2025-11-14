from pathlib import Path
import json
import random
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision

if __name__ == '__main__':
    #a = [97.4, 98.5, 88.3, 89.8, 97.1, 98.4, 84.2, 85.7, 93.8, 95.8, 78.9, 80.3]
    a = [97.3, 98.3, 88.1, 89.6, 97, 98.3, 83.8, 85.6, 93.1, 95.2, 77.5, 79.1]

    for i in range(len(a)):
        z = a[i] / 100
        e = 1 - z / (2 - z)
        print(100 * e)