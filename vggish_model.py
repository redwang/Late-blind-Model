import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import vggish_params as params

import pdb

class Vggish(nn.Module):
    def __init__(self):
        super(Vggish, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
            )

        # self.fc = nn.Sequential(
        #     nn.Linear(512*6*4, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(1024, 100),
        #     nn.BatchNorm1d(100, affine=False)
        #     )

        self.fc = nn.Sequential(
            nn.Linear(512*6*4, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 100),
            nn.BatchNorm1d(100, affine=False),
            nn.Dropout()
            )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        # pdb.set_trace()
        x = x.view(batch_size, -1)
        # pdb.set_trace()
        x = self.fc(x)
        return x

class MyModel(nn.Module):
    def __init__(self, num_class=10, num_unit=1024, weights_path=None):
        super(MyModel, self).__init__()

        self.vggish = Vggish()
        self.classifer = nn.Sequential(
            nn.Linear(100, num_unit),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(num_unit, num_class)
            )

        if weights_path is not None:
            self.load_weights(weights_path)

    def forward(self, x):
        x = self.vggish(x)
        x = self.classifer(x)
        # pdb.set_trace()
        return F.softmax(x, dim=1)

    def load_weights(self, weights_path):
        data = np.load(weights_path)
        weights = data['dict'][()]

        weights_name = weights.keys()

        for name, param in self.named_parameters():
            if name in weights_name and 'vggish' in name:
                # print name
                param.data = torch.from_numpy(weights[name])



if __name__ == '__main__':
    model = MyModel(weights_path='./save_models/vggish_weights.npz')

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print name

    # own_state = model.state_dict()
    # var_names = own_state.keys()
    # for i in range(len(var_names) - 2):
    #     print(var_names[i], own_state[var_names[i]].size())
