import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from skorch import NeuralNetClassifier, NeuralNet
import skorch
from torch import nn
import torch
import math

def mlp_model(data_shape):
    class MyModule(nn.Module):
        def __init__(self, data_shape, nonlin=nn.ReLU()):
            super().__init__()
            # -1 for sensitive attribute removal (group-blind training)
            num_units = math.ceil((2*data_shape[0])/(data_shape[1] - 1))
            self.dense0 = nn.Linear(data_shape[1] - 1, num_units)
            self.nonlin = nonlin
            #self.dropout = nn.Dropout(0.5)
            self.output = nn.Linear(num_units, 2)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, data, sample_weight, **kwargs):
            X = data
            X = self.nonlin(self.dense0(X))
            X = self.softmax(self.output(X))
            return X

    class MyNet(NeuralNet):
        def __init__(self, *args, criterion__reduce=False, **kwargs):
            # make sure to set reduce=False in your criterion, since we need the loss
            # for each sample so that it can be weighted
            super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)

        def get_loss(self, y_pred, y_true, X, *args, **kwargs):
            # override get_loss to use the sample_weight from X
            loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
            sample_weight = skorch.utils.to_tensor(X['sample_weight'], device=self.device)
            loss_reduced = (sample_weight * loss_unreduced).mean()
            return loss_reduced
    
    net = MyNet(
        MyModule(data_shape),
        criterion=nn.NLLLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        max_epochs=100,
        lr=0.01,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        device=0,
        verbose=False)
    return net