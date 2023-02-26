import argparse
import os
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F

import util.training
import util.visualization as vis
from util import metrics as me
from util.architecture import GeneratorBlock, DiscriminatorBlock
from util.io_custom import get_cifar_datasets, load_best_cp_data


class Generator(nn.Module):
    """
    Generator of the GAN architecture.

    Args:
        nz: Dimension of noise that is fed into the Generator
        ngf: hyper-parameter that determines the hidden size of the GeneratorBlock layers
        nc: Number of channels of the image. Default = 3 because of [R, G, B]. Could be 4 for PNG images with Alpha
        channel.
        batchnorm: Flag if batchnorm layers should be used.
        tconv: Flag of transposed convolution layers should be used.
        residual: Flag if residual connections should be used.
        lsc: Flag if the weight of the residual connection should be learnable
        use_emb: Flag if torch.nn.Embedding should be used as embedding instead of a linear layer.

    """
    def __init__(self, nz, ngf=64, nc=3, batchnorm=True, tconv=True, residual=True, lsc=True, use_emb=False):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.tconv = tconv
        self.batchnorm = batchnorm
        self.linear = nn.Linear(nz + 10, 4 ** 2 * ngf)
        self.use_emb = use_emb
        if self.use_emb:
            self.emb_layer = nn.Embedding(10, 50)
            self.linear = nn.Linear(nz + 50, 4 ** 2 * ngf)

        self.gb1 = GeneratorBlock(ngf, tconv=True, residual=residual, learnable_sc=lsc, batchnorm=self.batchnorm)
        self.gb2 = GeneratorBlock(ngf, tconv=True, residual=residual, learnable_sc=lsc, batchnorm=self.batchnorm)
        self.gb3 = GeneratorBlock(ngf, tconv=True, residual=residual, learnable_sc=lsc, batchnorm=self.batchnorm)

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(ngf)
        self.c1 = nn.Conv2d(ngf, nc, kernel_size=1, padding=0)

    def forward(self, input_image, input_label):
        input_label = input_label.to(torch.int64)
        # squeeze noise b x z x 1 x 1 to b x z
        input_image = input_image.squeeze(dim=2).squeeze(dim=2)
        if not self.use_emb:
            input_label = F.one_hot(input_label, num_classes=10)
            x = torch.cat((input_image, input_label), dim=1)
        else:
            label_emb = self.emb_layer(input_label)
            x = torch.cat((input_image, label_emb), dim=1)
        # B x nz + 10
        x = self.linear(x)
        # B x ngf * 4 * 4
        x = torch.reshape(x, (-1, self.ngf, 4, 4))
        # B x ngf x 4 x 4
        x = self.gb1(x)
        x = self.gb2(x)
        x = self.gb3(x)
        if self.batchnorm:
            x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.c1(x)
        x = nn.Tanh()(x)

        return x


class Discriminator(nn.Module):
    """
    Discriminator of the GAN architecture.

    Args:
          ndf: hyperparameter that determines the hidden size of Disciminator Blocks
          nc: Number of image channels
          lrelu: Flag if LReLU should be used isntead of ReLU
          num_classes: Number of classes for the last linear classification layer
          residual: Flag if residual connections should be used.
          lsc: Flag if the weight of the residual connection should be learnable
          use_emb: Flag if torch.nn.Embedding should be used as embedding instead of a linear layer.
    """
    def __init__(self, ndf=64, nc=3, sn=True, lrelu=True, num_classes=10, residual=True, lsc=True, use_emb=False,
                 leastsquare=False):
        super(Discriminator, self).__init__()

        self.sn = sn
        self.leastsquare = leastsquare
        self.num_classes = num_classes
        self.d1 = DiscriminatorBlock(nc, ndf, suppres_first_relu=True, down_sample=True, lrelu=lrelu, sn=sn,
                                     residual=residual, learnable_sc=lsc)
        self.d2 = DiscriminatorBlock(ndf, ndf, down_sample=True, lrelu=lrelu, sn=sn, residual=residual,
                                     learnable_sc=lsc)
        self.d3 = DiscriminatorBlock(ndf, ndf, down_sample=False, lrelu=lrelu, sn=sn, residual=residual,
                                     learnable_sc=lsc)
        self.d4 = DiscriminatorBlock(ndf, ndf, down_sample=False, lrelu=lrelu, sn=sn, residual=residual,
                                     learnable_sc=lsc)
        self.use_emb = use_emb
        if self.use_emb:
            self.emb_layer = torch.nn.Embedding(10, 50)

        if not self.use_emb:
            self.emb = nn.Linear(num_classes, ndf)
        else:
            self.emb = nn.Linear(50, ndf)
        if self.sn:
            self.emb = nn.utils.spectral_norm(self.emb)
        self.ll = nn.Linear(ndf, 1)
        if self.sn:
            self.ll = nn.utils.spectral_norm(self.ll)

    def forward(self, x, y: torch.Tensor):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)  # b x ndf x 8 x 8
        x = torch.sum(x, (2, 3))  # b x ndf

        if self.use_emb:
            emb_label = self.emb_layer(y.long())
            emb_label = self.emb(emb_label)
        else:
            oh = F.one_hot(y.long(), num_classes=self.num_classes).float()
            emb_label = self.emb(oh)

        proj = torch.sum(emb_label * x, 1, keepdim=True)  # b x 1
        lin = self.ll(x)

        # LS GAN does not use Sigmoid uses logits
        if not self.leastsquare:
            return nn.Sigmoid()(proj + lin)
        else:
            return proj + lin


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
    parser.add_argument("-s", "--sloppy", action="store_true",
                        help="If this arg is set then a sloppy IS of 50 images will be calculated instead of FID and IS of 1000 images.")
    parser.add_argument("--ngf", help="ngf dim", type=int, default=64)
    parser.add_argument("--spectral", action="store_true", help="Use Spectral normalization")
    parser.add_argument("--lrelu", action="store_true", help="Use leaky relu")
    parser.add_argument("--tconv", action="store_true", help="Use transposed convulation")
    parser.add_argument("--leastsquare", action="store_true", help="Use least square loss")
    parser.add_argument("--batchnorm", action="store_true", help="Use batch norm")
    parser.add_argument("--embedding", action="store_true", help="Use embedding matrix")
    parser.add_argument("--noresidual", action="store_true", help="don't use residual path")
    parser.add_argument("--thousand_fid", action="store_true", help="use 1k images for each class instead of 100")
    args = parser.parse_args()
    model_path = Path(f"models/{args.model_name}/")
    os.makedirs(model_path, exist_ok=True)
    # Root directory for dataset
    workers = 2
    batch_size = 64
    lr = 0.0002
    beta1 = 0.0
    # Create the generator
    nz = 128
    nc = 3  # 3 channels rgb
    num_epochs = 200
    ngf = args.ngf
    learnable_sc = True
    residual = not args.noresidual
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = Generator(nz=nz, batchnorm=args.batchnorm, tconv=args.tconv, residual=residual, lsc=learnable_sc,
                     use_emb=args.embedding).to(device)
    netD = Discriminator(sn=args.spectral, lrelu=args.lrelu, residual=residual, lsc=learnable_sc,
                         use_emb=args.embedding).to(device)

    # Print number of params
    print("Generator trainable weights: ", sum(p.numel() for p in netG.parameters() if p.requires_grad))
    print("Discriminator trainable weights: ", sum(p.numel() for p in netD.parameters() if p.requires_grad))

    with open(model_path / 'architecture.txt', 'w+') as f:
        f.write(str(netG))
        f.write('\n\n ----- \n\n')
        f.write(str(netD))

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.9))

    netG, netD, optimizerG, optimizerD, img_list, G_losses, D_losses, inc_scores, best_epoch, start_epoch, no_improve_count = load_best_cp_data(
        model_path, netG, netD, optimizerG, optimizerD)

    # Load data
    print("Loading Data")
    dataset_train, dataset_test, label_names = get_cifar_datasets()

    if args.training:
        util.training.train_model(model_path, num_epochs, batch_size, workers, netD, netG, nz, dataset_train,
                                  dataset_test, device, optimizerG, optimizerD, img_list, G_losses, D_losses,
                                  inc_scores,
                                  best_epoch, start_epoch, no_improve_count, ls_loss=False, sloppy=args.sloppy)

    # Generate images if flag is set.

    if not args.no_last_inception:
        print("Calculating last Inception score for best cp")
        reals = torch.stack([data['feat'] for data in dataset_test])
        real_labels = torch.stack([data['label'] for data in dataset_test])
        print(reals.shape)
        gen_imgs = me.gen_images(netG, device, nz)
        test_fid = me.FID_torchmetrics(gen_imgs, reals)
        test_is_mean, test_is_std = me.inception_score_torchmetrics(gen_imgs)

        # Save best scores
        if not args.thousand_fid:
            with open(model_path / 'final_inception_score_best.txt', 'w+') as f:
                f.write(f"Best epoch was: {best_epoch}\n")
                f.write(f"Inception scores torchmetrics. ('best' cp) Mean: {test_is_mean}, std: {test_is_std}\n")
                f.write(f"FID-torchmetrics: (best cp) {test_fid}\n")
                # Calc FID score for each class
                for class_i in range(10):
                    gen_imgs_class = me.gen_images_class(netG, device, nz, 100, class_i)
                    for x in real_labels[class_i * 100:(class_i + 1) * 100]:
                        print(f"real image label = {x}, current eval class = {class_i}")
                    fid_i = me.FID_torchmetrics(gen_imgs_class, reals[class_i * 100:(class_i + 1) * 100])
                    f.write(f"FID-Class (best cp) {label_names[class_i].decode()}: {fid_i}\n")
                dataset_test = util.io_custom.get_cifar_datasets_test_1000()
                reals = torch.stack([data['feat'] for data in dataset_test])
                for class_i in range(10):
                    gen_imgs_class = me.gen_images_class(netG, device, nz, 1000, class_i)
                    for x in real_labels[class_i * 1000:(class_i + 1) * 1000]:
                        print(f"real image label = {x}, current eval class = {class_i}")
                    fid_i = me.FID_torchmetrics(gen_imgs_class, reals[class_i * 1000:(class_i + 1) * 1000])
                    f.write(f"FID-Class-1000 (best cp) {label_names[class_i].decode()}: {fid_i}\n")

    netG, netD, optimizerG, optimizerD, img_list, G_losses, D_losses, inc_scores, best_epoch, start_epoch, no_improve_count = load_best_cp_data(
        model_path, netG, netD, optimizerG, optimizerD)

    if args.gen_images:
        vis.gen_plots(img_list, G_losses, D_losses, model_path, model_name=args.model_name)
