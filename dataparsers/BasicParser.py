
import torch
from torch.utils import data

class BasicParser(data.Dataset):
    def __init__(self):
        super(BasicParser, self).__init__()

        self.regression = False


