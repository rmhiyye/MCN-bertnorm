####################
# addition of layers for fine-tuning bert;
# activate function
####################

import torch
from torch import nn

# linear layer
class NeuralNetwork(nn.Module):
    def __init__(self, embbed_size):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(embbed_size, embbed_size) # Single linear layer
        torch.nn.init.eye_(self.linear.weight) # Linear layer weights initialization

    def forward(self, x):
        x = torch.nn.functional.normalize(x)
        x = self.linear(x)
        return x

# calculates a cosine similarity*(-1) between mention and label vectors.
def cos_dist(t1, t2):
    cos = nn.CosineSimilarity()
    cos_sim = cos(t1, t2)*(-1)
    return cos_sim
