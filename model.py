import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
class  CNN_Text(nn.Module):
    def __init__(self, hyperparameter_1):
        super(CNN_Text, self).__init__()
        self.hyperparameter = hyperparameter_1
        V = self.hyperparameter.vocab_num
        D = self.hyperparameter.embed_dim
        C = self.hyperparameter.class_num

        self.embed = nn.Embedding(V, D)
        if self.hyperparameter.word_embedding:
            pretrained_weight = np.array(self.hyperparameter.pretrained_weight)
            # print(type(pretrained_weight))
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.fc1 = nn.Linear(D, C)

    def forward(self, x):
        # print("未过embed的x = {}".format(x))

        x = self.embed(x)
        # print("过完embed的x = {}---------------------------------------------------".format(x))
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        x = self.fc1(x)
        return x



