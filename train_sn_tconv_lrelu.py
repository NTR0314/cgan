import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import util.visualization as vis
import util.training
from util.io_custom import get_cifar_datasets
from torch.nn import functional as F
from util import metrics as me


class UpsampleConv(nn.Module):
    def __init__(self, in_feat, out_feat, scale_factor=2, sn=False):
        super().__init__()
        self.us = nn.Upsample(scale_factor=scale_factor)
        self.c2d = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1)
        if sn:
            self.c2d = nn.utils.spectral_norm(self.c2d)

    def forward(self, x):
        return self.c2d(self.us(x))


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.label_upscale = nn.Sequential(
            nn.ConvTranspose2d(10, 10, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),  # Maybe this can also be reported as an improvement
            nn.ReLU(True)
        )
        self.img_upscale = nn.Sequential(
            # input is Z, going into a convolution
            # This is supposed to be used as an improvement over Convolution + Upsamling
            #  convTranspose2d args: c_in, c_out, kernel_size, stride, padding
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),  # Maybe this can also be reported as an improvement
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8 + 10, ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input_image, input_label):
        one_hot_label = F.one_hot(input_label.long(), num_classes=10).float()
        # b x 10 -> b x 10 x 1 x 1
        one_hot_label = one_hot_label.unsqueeze(-1).unsqueeze(-1)

        lbl_4 = self.label_upscale(one_hot_label)
        inp_img_4 = self.img_upscale(input_image)
        # Cat | dim: N x C x 4 x 4
        catvec = torch.cat((inp_img_4, lbl_4), dim=1)

        return self.main(catvec)


class Discriminator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Discriminator, self).__init__()
        # 10 x 1 x 1 -> 1 x 16 x 16
        # self.label_conv = nn.ConvTranspose2d(10, 1, 16, 1)
        self.label_conv = UpsampleConv(10, 1, scale_factor=16, sn=True)  # TODO try 10 instead of ngf * 8
        # nc x 32 x 32 -> 2ndf x 16 x 16
        self.ds_img = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(nc, ngf * 2, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main = nn.Sequential(
            # state size. (ndf) x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(ngf * 2 + 1, ngf * 4, 4, 2, 1, bias=False)),
            #                nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)),
            #                nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # conv2d args: c_in, c_out, kernel_size, stride, padding
            nn.utils.spectral_norm(nn.Conv2d(ngf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, input_image, label: torch.Tensor):
        # print(f"input image shape {input_image.shape}")
        # One hot of labels \in [0, 9]
        one_hot_label = F.one_hot(label.long(), num_classes=10).float()
        # reshape b x 10 -> b x 10 x 1 x 1
        one_hot_label = one_hot_label.unsqueeze(-1).unsqueeze(-1)
        # Scale labels to same as image after first conv layer like: https://www.researchgate.net/publication/331915834_Designing_nanophotonic_structures_using_conditional-deep_convolutional_generative_adversarial_networks
        label_upscaled = self.label_conv(one_hot_label)
        # Concatenate upscaled labels and downscaled img.
        downscaled_image = self.ds_img(input_image)
        # Cat, dim: batch x C x 16 x 16
        # print(downscaled_image.shape, label_upscaled.shape)
        concated = torch.cat((downscaled_image, label_upscaled), dim=1)  # TODO error lies here.

        return self.main(concated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cGAN for NN PR",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--gen_images", action="store_false", help="Don't generate images")
    parser.add_argument("-t", "--training", action="store_true", help="(Continue to) train the model")
    parser.add_argument("-m", "--model_name",
                        help="Model name. If specified the model will be saved to that directory and if"
                             "already existing the model will be loaded.")
    parser.add_argument("--no_last_inception", action="store_true",
                        help="If this arg is set then the last inception scores will not be calculated. This is mainly used for local computation.")
    parser.add_argument("--ngf", help="ngf dim", type=int, default=64)
    args = parser.parse_args()
    model_path = Path(f"models/{args.model_name}/")
    # Root directory for dataset
    workers = 2
    batch_size = 128
    lr = 0.0002
    beta1 = 0.5
    # Create the generator
    nz = 100 + 10  # 100 latent noise space dz + 10 dims of one-hot encoded label
    nc = 3  # 3 channels rgb
    ngf = args.ngf
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = Generator(nz, ngf, nc).to(device)
    netD = Discriminator(nz, ngf, nc).to(device)
    with open(model_path / 'architecture.txt', 'w+') as f:
        f.write(str(netG))
        f.write('\n\n ----- \n\n')
        f.write(str(netD))

    # Load stuff if existing
    best_cp_d_path = model_path / 'model_weights_netD_best.pth'
    best_cp_g_path = model_path / 'model_weights_netG_best.pth'
    tr_d_path = model_path / 'training_data.npz'

    # If existing load: best weights & training data
    if os.path.exists(best_cp_d_path) and os.path.exists(best_cp_g_path):
        netD.load_state_dict(torch.load(best_cp_d_path))
        netG.load_state_dict(torch.load(best_cp_g_path))
    if os.path.exists(tr_d_path):
        tr_d = np.load(tr_d_path, allow_pickle=True)
        img_list = list(tr_d['img_list'])
        G_losses = list(tr_d['G_losses'])
        D_losses = list(tr_d['D_losses'])
        inc_scores = list(tr_d['inc_scores'])
        best_epoch = int(tr_d['best_epoch'])
        start_epoch = int(tr_d['start_epoch'])
        fid_scores = list(tr_d['fid_scores'])
        fid_scores_classes = tr_d['fid_scores_classes']
        no_improve_count = int(tr_d['no_improve_count'])
    else:
        img_list = []
        G_losses = []
        D_losses = []
        inc_scores = []
        best_epoch = 0
        start_epoch = 0
        fid_scores = []
        fid_scores_classes = {}
        no_improve_count = 0

    # Load data
    dataset_train, dataset_test, dataset_dev, label_names = get_cifar_datasets()

    if args.training:
        util.training.train_model(model_path, 100, batch_size, workers, netD, netG, nz, lr, beta1, dataset_train,
                                  dataset_dev, device, img_list, G_losses, D_losses, inc_scores, fid_scores,
                                  fid_scores_classes,
                                  best_epoch, start_epoch, no_improve_count)

    # Generate images if flag is set.
    if args.gen_images:
        vis.gen_plots(img_list, G_losses, D_losses, model_path, model_name=args.model_name)

    if not args.no_last_inception:
        # Save Inception Sore and FID (Frechet Inception Distance) of best checkpoint
        # Load best CP for last FID score
        best_cp_d_path = model_path / 'model_weights_netD_best.pth'
        best_cp_g_path = model_path / 'model_weights_netG_best.pth'

        if os.path.exists(best_cp_d_path) and os.path.exists(best_cp_g_path):
            netD.load_state_dict(torch.load(best_cp_d_path))
            netG.load_state_dict(torch.load(best_cp_g_path))

        reals = [x for x in dataset_test]
        gen_imgs = me.gen_images(netG, device, nz)
        test_fid = me.FID_torchmetrics(gen_imgs, reals)
        test_is_mean, test_is_std = me.inception_score_torchmetrics(gen_imgs)

        # Save best scores
        with open(model_path / 'final_inception_score.txt', 'w+') as f:
            f.write(f"Inception scores torchmetrics. Mean: {test_is_mean}, std: {test_is_std}\n")
            f.write(f"FID-torchmetrics: {test_fid}\n")
            # Calc FID score for each class
            for class_i in range(10):
                gen_imgs_class = me.gen_images_class(netG, device, nz, 100, class_i)
                fid_i = me.FID_torchmetrics(gen_imgs_class, reals[class_i * 100:(class_i + 1) * 100])
                f.write(f"FID-Class {label_names[class_i].decode()}: {fid_i}\n")
