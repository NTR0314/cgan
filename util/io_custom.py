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

    class_imgs = [[] for i in range(10)]
    class_ctr = [0] * 10
    dev_set_imgs = torch.empty(0)
    dev_set_labels = torch.empty(0)
    all_done = False
    for image, label in zip(test_batch_data, test_batch_label):
        if all_done:
            break
        all_done = True
        for i in range(10):
            if class_ctr[i] < 200:
                all_done = False
            if label == i:
                if class_ctr[i] < 100:
                    class_imgs[i].append((image, label))
                    class_ctr[i] += 1
                    break
                elif class_ctr[i] < 200:
                    if dev_set_imgs.numel() == 0 and dev_set_labels.numel() == 0:
                        dev_set_imgs = image.unsqueeze(0)
                        dev_set_labels = label.unsqueeze(0)
                        break
                    else:
                        dev_set_imgs = torch.cat((dev_set_imgs, image.unsqueeze(0)), 0)
                        dev_set_labels = torch.cat((dev_set_labels, label.unsqueeze(0)), 0)
                        class_ctr[i] += 1
                        break

    dev_dataset = CIFARDataset(dev_set_imgs[:-1], dev_set_labels[:-1])
    test_set_images = torch.stack([x[0] for y in class_imgs for x in y])
    test_set_labels = torch.stack([x[1] for y in class_imgs for x in y])
    # This is sorted 100 labels per class 1000 in total
    test_dataset = CIFARDataset(test_set_images, test_set_labels)

    return train_dataset, test_dataset, dev_dataset, label_names


def load_best_cp_data(model_path, netG, netD):
    # Load stuff if existing
    best_cp_d_path = model_path / 'model_weights_netD_best.pth'
    best_cp_g_path = model_path / 'model_weights_netG_best.pth'
    tr_d_path = model_path / 'training_data.npz'

    # If existing load: best weights & training data

    # Remove dumb prefix if existing
    if os.path.exists(best_cp_d_path) and os.path.exists(best_cp_g_path):
        net_d_dict = torch.load(best_cp_d_path)
        net_g_dict = torch.load(best_cp_g_path)
        # DEBUG
        print(f"\n\n{net_d_dict.keys()=}\n\n")
        net_d_dict_fixed = OrderedDict([(k[len('module.'):], v) if 'module.' in k else (k,v) for k, v in net_d_dict.items()])
        net_g_dict_fixed = OrderedDict([(k[len('module.'):], v) if 'module.' in k else (k,v) for k, v in net_g_dict.items()])

        netD.load_state_dict(net_d_dict_fixed)
        netG.load_state_dict(net_g_dict_fixed)
    if os.path.exists(tr_d_path):
        tr_d = np.load(tr_d_path, allow_pickle=True)
        img_list = list(tr_d['img_list']) if 'img_list' in tr_d else []
        G_losses = list(tr_d['G_losses'])  if 'G_losses' in tr_d else []
        D_losses = list(tr_d['D_losses']) if 'D_losses' in tr_d else []
        inc_scores = list(tr_d['inc_scores']) if 'inc_scores' in tr_d else []
        best_epoch = int(tr_d['best_epoch']) if 'best_epoch' in tr_d else 0
        start_epoch = int(tr_d['start_epoch']) if 'start_epoch' in tr_d else 0
        fid_scores = list(tr_d['fid_scores']) if 'fid_scores' in tr_d else []
        fid_scores_classes = tr_d['fid_scores_classes'].item() if 'fid_scores_classes' in tr_d else {}
        no_improve_count = int(tr_d['no_improve_count']) if 'no_improve_count' in tr_d else 0

    return netG, netD, img_list, G_losses, D_losses, inc_scores, best_epoch, start_epoch, fid_scores, fid_scores_classes, no_improve_count
