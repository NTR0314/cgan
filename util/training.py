#!/usr/bin/env python
# coding: utf-8

import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch import nn

from . import metrics as me

def weights_init(m):
    # custom weights initialization called on netG and netD
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_model(model_path, num_epochs, batch_size, workers, netD, netG, nz, lr, beta1, dataset_train, dataset_dev, device,
                img_list=[], G_losses=[], D_losses=[], inc_scores=[], fid_scores=[], fid_scores_classes={},
                best_epoch=0, start_epoch=0, no_improve_count=0):
    # Set random seed for reproducibility
    manual_seed = 1337
    print("Using seed: ", manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    ngpu = torch.cuda.device_count()
    print(f"Number of GPUs = {ngpu}")

    # Create dir for saving stuff
    os.makedirs(model_path, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)


    netG.to(device)
    netD.to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    # only 8 because display is 8x8 = 64
    fixed_labels = torch.arange(8).unsqueeze(-1).repeat(1, 8).view(-1).to(device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

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
            # print(output, label)
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

        # Calc IS/FID/class-FID for each Epoch
        gen_imgs = me.gen_images(netG, device, nz)  # Generate 1000 random images from generator
        # Calc IS
        is_mean, is_std = me.inception_score_torchmetrics(gen_imgs)
        inc_scores.append((is_mean, is_std))
        # Calc FID for dev set
        reals = torch.stack([data['feat'] for data in dataset_dev])
        fid_dev = me.FID_torchmetrics(gen_imgs, reals)
        fid_scores.append(fid_dev)
        # Calc FID score for each class
        for class_i in range(10):
            gen_imgs_class = me.gen_images_class(netG, device, nz, 100, class_i)
            fid_i = me.FID_torchmetrics(gen_imgs_class, reals[class_i * 100:(class_i + 1) * 100])
            if not class_i in fid_scores_classes:
                fid_scores_classes[class_i] = [fid_i]
            else:
                fid_scores_classes[class_i].append(fid_i)

        # Save best pt, compare mean.
        if fid_dev >= max(fid_scores):
            # Save best
            torch.save(netD.state_dict(), model_path / f'model_weights_netD_best.pth')
            torch.save(netG.state_dict(), model_path / f'model_weights_netG_best.pth')
            no_improve_count = 0
        else:
            no_improve_count += 1
        # Stop if not improvement after 15 epochs
        if no_improve_count >= 15:
            print("No improvements for 15 epochs. Breaking train loop")
            break

        # Output training stats
        print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item()}\tLoss_G:'
              + f'{errG.item()}\tD(x): {D_x}\tD(G(z)): {D_G_z1} / {D_G_z2} - IS-mean: {is_mean} - FID: {fid_dev}')

    np.savez(model_path / f"training_data.npz",
             img_list=img_list,
             G_losses=G_losses,
             D_losses=D_losses,
             inc_scores=inc_scores,
             best_epoch=best_epoch,
             start_epoch=epoch,
             fid_scores=fid_scores,
             fid_scores_classes=fid_scores_classes,
             no_improve_count=no_improve_count)

    # Save model after training:
    # https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html#save-and-load-the-model
    torch.save(netD.state_dict(), model_path / f'model_weights_netD_last.pth')
    torch.save(netG.state_dict(), model_path / f'model_weights_netG_last.pth')
