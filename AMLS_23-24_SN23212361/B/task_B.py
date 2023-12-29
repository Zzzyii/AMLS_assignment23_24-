
import os
import argparse
import json
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from B.models import ResNet18
from B.dataset import INFO, taskB
from B.evaluator import getAUC, getACC, save_results




def mainB(flag, input_root, output_root, end_epoch):

    dataclass = {
        "PathMNIST": taskB
    }

    with open(INFO, 'r') as f:
        info = json.load(f)
        n_channels = info[flag]['n_channels']
        n_classes = len(info[flag]['label'])

    start_epoch = 0
    lr = 0.001
    batch_size = 128
    val_auc_list = []
    dir_path = os.path.join(output_root, '%s_checkpoints' % (flag))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('==> Preparing data...')
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = dataclass[flag](root=input_root, split='train', transform=train_transform)
    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = dataclass[flag](root=input_root, split='val', transform=val_transform)
    val_loader = data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = dataclass[flag](root=input_root, split='test', transform=test_transform)
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True)

    print('==> Building and training model...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in trange(start_epoch, end_epoch):
        train(model, optimizer, criterion, train_loader, device)  ##我
        val(model, val_loader, device, val_auc_list, dir_path, epoch)#我
        if epoch == end_epoch - 1:
            print("Training completed!")    

    plt.figure()
    plt.plot(range(start_epoch, end_epoch), val_auc_list, label='AUC of Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC Convergence Graph')
    plt.legend()
    plt.show(block=True)



    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('epoch %s is the best model' % (index))

    print('==> Testing model...')
    restore_model_path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    model.load_state_dict(torch.load(restore_model_path)['net'])
    test(model, 'train', train_loader, device, flag, output_root=output_root)
    test(model, 'val', val_loader, device, flag, output_root=output_root)
    test(model, 'test', test_loader, device, flag, output_root=output_root)


def train(model, optimizer, criterion, train_loader, device):

    step=0
    print("\n")
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        step += 1
        print("batch",step)

        targets = targets.squeeze().long().to(device)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
    


def val(model, val_loader, device, val_auc_list, dir_path, epoch):

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))
            targets = targets.squeeze().long().to(device)
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }

    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)


def test(model, split, test_loader, device, flag, output_root=None):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            outputs = model(inputs.to(device))

            targets = targets.squeeze().long().to(device)
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score)
        acc = getACC(y_true, y_score)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        if output_root is not None:
            output_dir = os.path.join(output_root, flag)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path)


def run_B():
    parser = argparse.ArgumentParser(description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name', default='PathMNIST', help='subset of MedMNIST', type=str)
    parser.add_argument('--input_root', default='./Datasets', help='input root, the source of dataset files', type=str)
    parser.add_argument('--output_root', default='./B/output_B', help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epoch', default=100, help='num of epochs of training', type=int)
    args = parser.parse_args()
    data_name = args.data_name
    input_root = args.input_root
    output_root = args.output_root
    end_epoch = args.num_epoch
    mainB(data_name, input_root, output_root, end_epoch=end_epoch)
