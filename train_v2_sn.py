import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import util.visualization as vis
import util.training
from util.io_custom import get_cifar_datasets, load_best_cp_data
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


class GeneratorBlock(nn.Module):
    def __init__(self, ngf, bn=True, tconv=True, residual=False):
        super().__init__()
        self.bn = bn
        self.tconv = tconv
        self.residual = residual

        self.tconv1 = nn.ConvTranspose2d(ngf, ngf, 4, 2, 1)

        self.bn1 = nn.BatchNorm2d(ngf)
        self.bn2 = nn.BatchNorm2d(ngf)

        self.c1 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1)

    def forward(self, x):
        orig = x
        if self.bn:
            x = self.bn1(x)
        x = nn.ReLU()(x)
        if self.tconv:
            x = self.tconv1(x)
        else:
            x = self.c1(nn.Upsample(scale_factor=2)(x))
        if self.bn:
            x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.c2(x)
        # TODO fix residual
        if self.residual:
            return x + orig
        return x


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=128, nc=3, bn=True, tconv=True):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.tconv = tconv
        self.bn = bn
        self.linear = nn.Linear(nz + 10, 4 ** 2 * ngf)

        self.gb1 = GeneratorBlock(ngf, tconv=True)
        self.gb2 = GeneratorBlock(ngf, tconv=True)
        self.gb3 = GeneratorBlock(ngf, tconv=True)

        if bn:
            self.bn1 = nn.BatchNorm2d(ngf)
        self.c1 = nn.Conv2d(ngf, nc, kernel_size=1, padding=0)

    def forward(self, input_image, input_label):
        input_label = input_label.to(torch.int64)
        # print(input_image.shape, input_label.shape, input_label.dtype)
        input_image = input_image.squeeze(dim=2).squeeze(dim=2)
        input_label = F.one_hot(input_label, num_classes=10)
        # print(input_image.shape, input_label.shape, input_label.dtype)
        x = torch.cat((input_image, input_label), dim=1)
        # print(x.shape)
        # B x nz + 10
        x = self.linear(x)
        # B x ngf * 4 * 4
        x = torch.reshape(x, (-1, self.ngf, 4, 4))
        # B x ngf x 4 x 4
        x = self.gb1(x)
        x = self.gb2(x)
        x = self.gb3(x)
        if self.bn:
            x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.c1(x)
        x = nn.Tanh()(x)

        return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, lrelu=False, suppres_first_relu=False, down_sample=True, sn=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ds = down_sample
        self.lrelu = lrelu
        self.suppres_first_relu = suppres_first_relu
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        if sn:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

    def forward(self, x):
        if not self.suppres_first_relu:
            if self.lrelu:
                x = nn.LeakyReLU()(x)
            else:
                x = nn.ReLU()(x)

        x = self.c1(x)

        if self.lrelu:
            x = nn.LeakyReLU()(x)
        else:
            x = nn.ReLU()(x)
        x = self.c2(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, sn=True, lrelu=True, num_classes=10):
        super(Discriminator, self).__init__()

        self.sn = sn
        self.num_classes = num_classes
        self.d1 = DiscriminatorBlock(nc, ndf, suppres_first_relu=True, down_sample=True, lrelu=lrelu, sn=sn)
        self.d2 = DiscriminatorBlock(ndf, ndf, down_sample=True, lrelu=lrelu, sn=sn)
        self.d3 = DiscriminatorBlock(ndf, ndf, down_sample=False, lrelu=lrelu, sn=sn)
        self.d4 = DiscriminatorBlock(ndf, ndf, down_sample=False, lrelu=lrelu, sn=sn)

        self.emb = nn.Linear(num_classes, ndf)
        if self.sn:
            self.emb = nn.utils.spectral_norm(self.emb)
        self.ll = nn.Linear(ndf, 1)
        if self.sn:
            self.ll = nn.utils.spectral_norm(self.ll)

    def forward(self, x, y: torch.Tensor):
        # One hot of labels \in [0, 9]
        oh = F.one_hot(y.long(), num_classes=self.num_classes).float()

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)  # b x ndf x 8 x 8
        x = torch.sum(x, (2, 3))  # b x ndf

#        print(oh.shape)
#        print(self.emb(oh).shape)
        proj = torch.sum(self.emb(oh) * x, 1, keepdim=True)  # b x 1
        lin = self.ll(x)

        return nn.Sigmoid()(proj + lin)


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
    os.makedirs(model_path, exist_ok=True)
    # Root directory for dataset
    workers = 2
    batch_size = 128
    lr = 0.0002
    beta1 = 0.5
    # Create the generator
    nz = 100
    nc = 3  # 3 channels rgb
    ngf = args.ngf
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = Generator(bn=False, tconv=False).to(device)
    netD = Discriminator(sn=True, lrelu=False).to(device)
    with open(model_path / 'architecture.txt', 'w+') as f:
        f.write(str(netG))
        f.write('\n\n ----- \n\n')
        f.write(str(netD))

    netG, netD, img_list, G_losses, D_losses, inc_scores, best_epoch, start_epoch, fid_scores, fid_scores_classes, no_improve_count = load_best_cp_data(model_path, netG, netD)

    # Load data
    dataset_train, dataset_test, dataset_dev, label_names = get_cifar_datasets()

    if args.training:
        util.training.train_model(model_path, 100, batch_size, workers, netD, netG, nz, lr, beta1, dataset_train,
                                  dataset_dev, device, img_list, G_losses, D_losses, inc_scores, fid_scores,
                                  fid_scores_classes,
                                  best_epoch, start_epoch, no_improve_count, ls_loss=False)

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

        reals = torch.stack([data['feat'] for data in dataset_dev])
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
