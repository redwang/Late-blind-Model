from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vggish_datasets import MNIST
from torchvision import datasets, transforms

import vggish_params as params
from vggish_model import MyModel

import pdb

mode_save_path = './save_models'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
    # for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = torch.squeeze(target).to(device)

        # data = data.view(-1, )
        # pdb.set_trace()

        output = model(data)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader), loss.item()))

def test(model, test_loader):
    model.eval()
    with torch.no_grad(): 
        test_loss = 0
        correct = 0

        for data, target in test_loader:
            data = data.to(device)
            target = torch.squeeze(target).to(device)
            # data = data.view(-1, )
            # pdb.set_trace()

            output = model(data)
            test_loss += F.cross_entropy(output, target).item()

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        MNIST(root='./data/mnist', train=True),
        batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        MNIST(root='./data/mnist', train=False),
        batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model = MyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    prec = test(model, test_loader)
    best_prec = 0
    for epoch in range(0, 5):
        if epoch == 3:
            optimizer = optim.Adam(model.parameters(), lr=1e-5)
        train(model, train_loader, optimizer, epoch)
        prec = test(model, test_loader)

        if best_prec < prec:
            best_prec = prec

            save_state = {'shared_layers': model.state_dict(),
                          'best_prec': best_prec}
            # pdb.set_trace()
            torch.save(save_state, './save_models/vggish_mnist_best.pth')

