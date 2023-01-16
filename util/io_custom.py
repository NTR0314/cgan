import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset


def unpickle(file):
    with open(file, 'rb') as fo:
        out = pickle.load(fo, encoding='bytes')
    return out


class CIFARDataset(Dataset):
    def __init__(self, batches_data, batches_label):  # Constructor
        self.data = list(zip(batches_data, batches_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        feat, label = self.data[index]

        return {'label': label, 'feat': feat}


def get_cifar_datasets():
    data_batches_data = torch.empty(0)
    data_batches_label = torch.empty(0)

    for i in range(1, 6):
        file_path = "cifar-10-batches-py/data_batch_" + str(i)
        data_batch = unpickle(file_path)
        data_batch_feats = torch.tensor(data_batch[b'data'])
        data_batch_len = data_batch_feats.shape[0]
        data_batch_feats = (torch.reshape(data_batch_feats, (data_batch_len, 3, 32, 32)) - 127.5) / 127.5

        data_batches_data = torch.cat((data_batches_data, data_batch_feats), 0)
        data_batches_label = torch.cat((data_batches_label, torch.tensor(data_batch[b'labels'])), 0)

    # squeeze batchtes
    train_batches_data = torch.squeeze(data_batches_data)
    train_batches_label = torch.squeeze(data_batches_label)

    # [1,2, ...] -> [dog ...] ?
    label_names = unpickle("cifar-10-batches-py/batches.meta")[b'label_names']

    test_batch = unpickle("cifar-10-batches-py/test_batch")
    test_batch_data = torch.tensor(test_batch[b'data'])
    test_batch_len = test_batch_data.shape[0]
    test_batch_data = (torch.reshape(test_batch_data, (test_batch_len, 3, 32, 32)) - 127.5) / 127.5
    test_batch_label = torch.tensor(test_batch[b'labels'])

    train_dataset = CIFARDataset(train_batches_data, train_batches_label)

    class_ctr = [0] * 10
    dev_set_imgs = [torch.empty(0) for _ in range(10)]
    dev_set_labels = [torch.empty(0) for _ in range(10)]
    all_done = False
    for image, label in zip(test_batch_data, test_batch_label):
        if all_done:
            break
        all_done = True
        for i in range(10):
            if class_ctr[i] < 99:
                all_done = False
            if label == i:
                if class_ctr[i] < 99:
                    if dev_set_imgs[i].numel() == 0 and dev_set_labels[i].numel() == 0:
                        dev_set_imgs[i] = image.unsqueeze(0)
                        dev_set_labels[i] = label.unsqueeze(0)
                        break
                    else:
                        print(f"image shape: {image.shape}")
                        dev_set_imgs[i] = torch.cat((dev_set_imgs[i], image.unsqueeze(0)), 0)
                        dev_set_labels[i] = torch.cat((dev_set_labels[i], label.unsqueeze(0)), 0)
                        class_ctr[i] += 1
                        break

    print(dev_set_imgs[i].shape)
    for i in range(10):
        dev_set_imgs[i] = dev_set_imgs[i].squeeze()
    test_set_images = torch.cat(dev_set_imgs)
    test_set_labels = torch.cat(dev_set_labels)
    print(test_set_images.shape)

    # This is sorted 100 labels per class 1000 in total
    test_dataset = CIFARDataset(test_set_images, test_set_labels)

    return train_dataset, test_dataset, label_names


def load_best_cp_data(model_path, netG, netD, optimizerG, optimizerD, last=False):
    # Load stuff if existing
    if not last:
        best_cp_path = model_path / 'model_best.pth'
    else:
        # Find maximum
        files = os.listdir(model_path)
        max_cp = max([int(x[len('model_'):-len('.pth')]) for x in files if 'model' in x and 'best' not in x])
        best_cp_path = model_path / f'model_{max_cp}.pth'

    # Remove dumb prefix if existing
    if os.path.exists(best_cp_path):
        model = torch.load(best_cp_path)
        net_d_dict = model['netD_state_dict']
        net_g_dict = model['netG_state_dict']
        # DEBUG
        print(f"\n\n{net_d_dict.keys()=}\n\n")
        net_d_dict_fixed = OrderedDict([(k[len('module.'):], v) if 'module.' in k else (k,v) for k, v in net_d_dict.items()])
        net_g_dict_fixed = OrderedDict([(k[len('module.'):], v) if 'module.' in k else (k,v) for k, v in net_g_dict.items()])

        netD.load_state_dict(net_d_dict_fixed)
        netG.load_state_dict(net_g_dict_fixed)
        optimizerD.load_state_dict(model['optimizer_d_state_dict'])
        optimizerG.load_state_dict(model['optimizer_g_state_dict'])

        img_list = model['img_list']
        G_losses = model['G_losses']
        D_losses = model['D_losses']
        inc_scores = model['inc_scores']
        best_epoch = model['best_epoch']
        start_epoch = model['start_epoch']
        print(f"loaded start epoch is {start_epoch}")
        no_improve_count = model['no_improve_count']
    else:
        img_list = []
        G_losses = []
        D_losses = []
        inc_scores = []
        best_epoch = 0
        start_epoch = 0
        no_improve_count = 0


    return netG, netD, optimizerG, optimizerD, img_list, G_losses, D_losses, inc_scores, best_epoch, start_epoch, no_improve_count
