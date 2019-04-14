from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import vggish_params as params
from network import ClassifyNet, Vggish, generator, discriminator
from vggish_datasets import MNIST

import pdb

mode_save_path = './save_models'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_weight(model, model_path):
    own_state = model.state_dict()

    save_state_dict = torch.load(model_path)
    save_state = save_state_dict['shared_layers']

    load_state = {}
    for name in save_state.keys():
        if 'vggish' in name:
            new_name = name[7:]
            load_state[new_name] = save_state.pop(name)

    model.load_state_dict(load_state, strict=False)

def train(vggish_model, class_model, G_model, D_model, train_loader, optimizer, epoch):
    vggish_model.train()
    class_model.train()
    G_model.train()

    criterion = nn.BCELoss().cuda()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = torch.squeeze(target).to(device)

        batch_size = data.size(0)
        label_real = torch.ones(batch_size).to(device)

        output = vggish_model(data)
        G_result = G_model(output.view(-1, 100, 1, 1))
        output = class_model(G_result)
        cls_loss = F.cross_entropy(output, target)

        loss = cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader), loss.item()))

def test(vggish_model, class_model, G_model, test_loader, epoch):
    vggish_model.eval()
    class_model.eval()
    G_model.eval()

    with torch.no_grad(): 
        test_loss = 0
        correct = 0

        for data, target in test_loader:
            data = data.to(device)
            target = torch.squeeze(target).to(device)

            output = vggish_model(data)
            G_result = G_model(output.view(-1, 100, 1, 1))
            img_result = G_result
            output = class_model(img_result)
            test_loss += F.cross_entropy(output, target).item()

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    n_rows = n_cols = 8
    is_gray=True
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8))
    
    for ax, img in zip(axes.flatten(), G_result):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        if is_gray:
            img = img.cpu().data.view(64, 64).numpy()
            ax.imshow(img, cmap='gray', aspect='equal')
        else:
            img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap='None', aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(epoch)
    fig.text(0.5, 0.04, title, ha='center')
    
    plt.savefig(os.path.join('./results', 'train_all_emnist_mnist_epoch_{}.png'.format(epoch)))
    plt.close()


    return 100. * correct / len(test_loader.dataset)

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        MNIST(root='./data/mnist', train=True),
        batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        MNIST(root='./data/mnist', train=False),
        batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    vggish_model = Vggish().to(device)
    load_weight(vggish_model, './save_models/vggish_mnist_best.pth')

    class_model = ClassifyNet().to(device)
    class_model.load_state_dict(torch.load('./save_models/classify_mnist_best.pth')['shared_layers'])

    G_model = generator(out_size=3).to(device)
    G_model.load_state_dict(torch.load('./save_models/emnist_G_best.pth.tar'))

    D_model = discriminator(in_size=3, ndf=128).to(device)
    D_model.load_state_dict(torch.load('./save_models/emnist_D_best.pth.tar'))


    optimizer = optim.Adam(vggish_model.parameters(), lr=1e-3)

    prec = test(vggish_model, class_model, G_model, test_loader, 0)
    best_prec = prec
    for epoch in range(1, 5):
        if epoch > 3:
            optimizer = optim.Adam(list(vggish_model.parameters())+list(G_model.parameters()), lr=1e-4)
        train(vggish_model, class_model, G_model, D_model, train_loader, optimizer, epoch)
        prec = test(vggish_model, class_model, G_model, test_loader, epoch)

        if best_prec < prec:
            best_prec = prec

            save_state = {'vgg_layers': vggish_model.state_dict(),
                          'class_layers': class_model.state_dict(),
                          'gen_layers': G_model.state_dict(),
                          'best_prec': best_prec}
            torch.save(save_state, './save_models/train_all_emnist_mnist_best.pth')

