#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from util import visualization as vis
from util import metrics as me

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cGAN for NN PR",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--gen_images", action="store_false", help="Don't generate images")
    parser.add_argument("-t", "--training", action="store_true", help="(Continue to) train the model")
    parser.add_argument("-m", "--model_name",
                        help="Model name. If specified the model will be saved to that directory and if"
                             "already existing the model will be loaded.")
    parser.add_argument("--ngf", help="ngf dim", type=int, default=64)
    args = parser.parse_args()
    config = vars(args)

    # Set random seed for reproducibility
    manualSeed = 1337
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    model_path = Path(f"models/{args.model_name}/")

    # Root directory for dataset
    dataroot = "data/cifar10"
    workers = 2
    batch_size = 128
    lr = 0.0002
    beta1 = 0.5
    num_img_inception = 2500
    ngpu = torch.cuda.device_count()
    print(f"num gpu = {ngpu}")

    # Create dir for saving stuff
    os.makedirs(model_path, exist_ok=True)


    def unpickle(file):
        with open(file, 'rb') as fo:
            out = pickle.load(fo, encoding='bytes')
        return out


    def pickle_save(file, obj):
        with open(file, 'wb') as f:
            pickle.dump(obj, f)
        return


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
    data_batches_data = torch.squeeze(data_batches_data)
    data_batches_label = torch.squeeze(data_batches_label)

    label_names = unpickle("cifar-10-batches-py/batches.meta")[b'label_names']

    test_batch = unpickle("cifar-10-batches-py/test_batch")
    test_batch_data = torch.tensor(test_batch[b'data'])
    test_batch_len = test_batch_data.shape[0]
    test_batch_data = (torch.reshape(test_batch_data, (test_batch_len, 3, 32, 32)) - 127.5) / 127.5
    test_batch_label = torch.tensor(test_batch[b'labels'])


    class CIFARDataset(Dataset):
        def __init__(self, test=False):  # Constructor
            if test:
                self.data = list(zip(test_batch_data, test_batch_label))
            else:
                self.data = list(zip(data_batches_data, data_batches_label))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index: int):
            feat, label = self.data[index]

            return {'label': label, 'feat': feat}


    dataset_train = CIFARDataset()
    dataset_test = CIFARDataset(test=True)
    dataloader = torch.utils.data.DataLoader(dataset_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    # custom weights initialization called on netG and netD
    def weights_init(m):
#        print(f"DEBUG: {m}")
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif type(m) == nn.BatchNorm2d:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    class UpsampleConv(nn.Module):
        def __init__(self, in_feat, out_feat, scale_factor=2):
            super().__init__()
            self.us = nn.Upsample(scale_factor=scale_factor)
            self.c2d = nn.Conv2d(in_feat, out_feat, kernel_size = 3, stride=1, padding=1)

        def forward(self, x):
            return self.c2d(self.us(x))



    class Generator(nn.Module):
        def __init__(self, nz, ngf, nc):
            super(Generator, self).__init__()
            self.label_upscale = nn.Sequential(
                # nn.ConvTranspose2d(10, ngf * 8, 4, 1, 0, bias=False),
                UpsampleConv(10, 10, scale_factor=4), # TODO try 10 instead of ngf * 8
                # nn.BatchNorm2d(ngf * 8),  # Maybe this can also be reported as an improvement
                nn.ReLU(True)
            )
            self.img_upscale = nn.Sequential(
                # input is Z, going into a convolution
                # This is supposed to be used as an improvement over Convolution + Upsamling
                #  convTranspose2d args: c_in, c_out, kernel_size, stride, padding
                # nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                UpsampleConv(nz, ngf * 8, scale_factor=4),
                # nn.BatchNorm2d(ngf * 8),  # Maybe this can also be reported as an improvement
                nn.ReLU(True)
            )

            self.main = nn.Sequential(
                # state size. (ngf*8) x 4 x 4
                # nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=False),
                UpsampleConv(ngf * 8 + 10, ngf * 4),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                # nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                UpsampleConv(ngf * 4, ngf * 2),
                # nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                # nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                UpsampleConv(ngf * 2, nc),
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


    # Create the generator
    nz = 100 + 10  # 100 normal dz + 10 dims of one-hot encoded label
    nc = 3
    ngf = args.ngf

    netG = Generator(nz, ngf, nc).to(device)
    netG.apply(weights_init)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Print the model
    print("Using Generator:\n", str(netG))
    with open(model_path / 'architecture.txt', 'w+') as f:
        f.write(str(netG))


    class Discriminator(nn.Module):
        def __init__(self, nz, ngf, nc):
            super(Discriminator, self).__init__()
            # 10 x 1 x 1 -> 1 x 16 x 16
            #self.label_conv = nn.ConvTranspose2d(10, 1, 16, 1)
            self.label_conv = UpsampleConv(10, 1, scale_factor=16) # TODO try 10 instead of ngf * 8
            # nc x 32 x 32 -> 2ndf x 16 x 16
            self.ds_img = nn.Sequential(
                nn.Conv2d(nc, ngf * 2, 4, 2, 1, bias=False),
#               nn.BatchNorm2d(ngf * 2),
                # nn.LeakyReLU(0.2, inplace=True)
                nn.ReLU()
            )

            self.main = nn.Sequential(
                # state size. (ndf) x 16 x 16
                nn.Conv2d(ngf * 2 + 1, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(),
                # state size. (ndf*2) x 8 x 8
                nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(),
                # state size. (ndf*8) x 4 x 4
                # conv2d args: c_in, c_out, kernel_size, stride, padding
                nn.Conv2d(ngf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input_image, label: torch.Tensor):
            #print(f"input image shape {input_image.shape}")
            # One hot of labels \in [0, 9]
            one_hot_label = F.one_hot(label.long(), num_classes=10).float()
            # reshape b x 10 -> b x 10 x 1 x 1
            one_hot_label = one_hot_label.unsqueeze(-1).unsqueeze(-1)
            # Scale labels to same as image after first conv layer like: https://www.researchgate.net/publication/331915834_Designing_nanophotonic_structures_using_conditional-deep_convolutional_generative_adversarial_networks
            label_upscaled = self.label_conv(one_hot_label)
            # Concatenate upscaled labels and downscaled img.
            downscaled_image = self.ds_img(input_image)
            # Cat, dim: batch x C x 16 x 16
            #print(downscaled_image.shape, label_upscaled.shape)
            concated = torch.cat((downscaled_image, label_upscaled), dim=1) #TODO error lies here.

            return self.main(concated)


    # Create the Discriminator
    netD = Discriminator(nz, ngf, nc).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print("Using Discriminator:\n", str(netG))
    with open(model_path / 'architecture.txt', 'a+') as f:
        f.write('\n\n ----- \n\n')
        f.write(str(netD))

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    # randn args: input dims -> 64 x nz x 1 x 1
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    # shape: 64
    fixed_labels = torch.arange(8).unsqueeze(-1).repeat(1, 8).view(-1).to(device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Load model
    # Check if best cp of model exists
    best_cp_d_path = model_path / 'model_weights_netD_best.pth'
    best_cp_g_path = model_path / 'model_weights_netG_best.pth'
    tr_d_path = model_path / 'training_data.npz'

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
    else:
        img_list = []
        G_losses = []
        D_losses = []
        inc_scores = []
        best_epoch = 0
        start_epoch = 0

    no_improve_count = 0
    num_epochs = 100
    num_img_inception = 5000
    epoch = 0

    if args.training:
        # Lists to keep track of progress
        print("Starting Training Loop...")
        for epoch in range(start_epoch, num_epochs):
            for i, data in enumerate(dataloader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data['feat'].to(device)
                real_img_labels = data['label'].to(device)
                b_size = real_cpu.size(0)
                # torch.full args: size, fill_value -> batch_size großeen tensor mit 1 = echtes Bild erstellen
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                # output of netD is 0/1 depending on whether it guesses it is fake or not
                # view(-1) sorgt dafür dass alle in einer grossen liste sind um mit label zu vergleichen
                output = netD(real_cpu, real_img_labels).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise, real_img_labels)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach(), real_img_labels).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake, real_img_labels).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            # Each Epoch do:
            # Check how the generator is doing by saving G's output on fixed_noise
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_labels).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            # Calc Inception Score for Epoch

            # TODO: Does this work? pen
            gen_imgs = me.gen_images(netG, device, nz)
            is_mean, is_std = me.inception_score_own(gen_imgs, device, splits=5, batch_size=32, upscale=True)
            inc_scores.append((is_mean, is_std))
            # Save best pt, compare mean.
            if is_mean >= max(inc_scores, key=lambda x: x[0])[0]:
                # Save best
                torch.save(netD.state_dict(), model_path / f'model_weights_netD_best.pth')
                torch.save(netG.state_dict(), model_path / f'model_weights_netG_best.pth')
                no_improve_count = 0
            else:
                no_improve_count += 1
            # Stop if not improvement after 15 epochs
            if no_improve_count >= 15:
                print("No improvements for 10 epochs. Breaking train loop")
                break

            # Output training stats
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item()}\tLoss_G:'
                  + f'{errG.item()}\tD(x): {D_x}\tD(G(z)): {D_G_z1} / {D_G_z2} - IS-mean: {is_mean}')

    if args.gen_images:
        vis.gen_plots(img_list, G_losses, D_losses, model_path)

    np.savez(model_path / f"training_data.npz",
             img_list=img_list,
             G_losses=G_losses,
             D_losses=D_losses,
             inc_scores=inc_scores,
             best_epoch=best_epoch,
             start_epoch=epoch)

    # Save model after training:
    # https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html#save-and-load-the-model
    torch.save(netD.state_dict(), model_path / f'model_weights_netD_last.pth')
    torch.save(netG.state_dict(), model_path / f'model_weights_netG_last.pth')

    # Save Inception Sore and FID (Frechet Inception Distance) of best checkpoint
    # Load best cp
    # Check if best cp of model exists
    best_cp_d_path = model_path / 'model_weights_netD_best.pth'
    best_cp_g_path = model_path / 'model_weights_netG_best.pth'

    if os.path.exists(best_cp_d_path) and os.path.exists(best_cp_g_path):
        netD.load_state_dict(torch.load(best_cp_d_path))
        netG.load_state_dict(torch.load(best_cp_g_path))

    # Calculate Inception score both self-implemented and torchmetrics implementation
    gen_imgs = me.gen_images(netG, device, nz)
#    inc_score_self = me.inception_score_own(gen_imgs, device, batch_size=32, upscale=True, splits=10)
    inc_score_torch = me.inception_score_torchmetrics(gen_imgs)
    # Generate real images
    dataloader = torch.utils.data.DataLoader(dataset_test,
                                             batch_size=1,
                                             num_workers=workers)
    reals = torch.empty(0)
#    with open('debug_cifar.txt', 'w+') as f:
#        for batch in dataloader:
#            for label in batch['label']:
#                f.write(str(label))
#                f.write('\n')
#
#    print("finished writing debug")

    # dumm aber ok
    counter_test_images = [0] * 10
    for batch in dataloader:
#        print(batch)
        feat = batch['feat'] # 1 x 3 x 32 x 32
#        print(feature_batch.shape)
        label = batch['label']
        all_done = False
        if all_done:
            break
        all_done = True
        for i in range(10):
            if counter_test_images[i] != 100:
                all_done = False
            if label == i and counter_test_images[i] < 100:
#                print(counter_test_images)
                counter_test_images[i] += 1
                reals = torch.cat((reals, feat), 0)

#    print(reals.shape)
    fid_score_torch = me.FID_torchmetrics(gen_imgs, reals)
    
    # Save best scores
    with open(model_path / 'final_inception_score.txt', 'w+') as f:
#        f.write(f"Inception scores self. Mean: {inc_score_self[0]}, std: {inc_score_self[1]}\n")
        f.write(f"Inception scores torchmetrics. Mean: {inc_score_torch[0]}, std: {inc_score_torch[1]}\n")
        f.write(f"FID-torchmetrics: {fid_score_torch}\n")
