#!/usr/bin/env python
# coding: utf-8

import os
import random
import time

import torch
import torch.utils.data
import torchvision.utils as vutils
from torch import nn

from . import metrics as me

def weights_init(m):
    # custom weights initialization called on netG and netD
    # This is not my, Oswald Zink, own idea but taken from some blog post.

    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_model(model_path, num_epochs, batch_size, workers, netD, netG, nz, dataset_train, dataset_dev, device,
                optimizerG, optimizerD,
                img_list=[], G_losses=[], D_losses=[], inc_scores=[],
                best_epoch=0, start_epoch=0, no_improve_count=0, ls_loss=True, sloppy=False):
    """
    Method for training the GAN model

    Args:
        model_path: Save path of the trained model
        num_epochs: max number of epochs to train the model
        batch_size: batch size
        workers: amount of cpu workers
        netD: Discriminator neural net
        netG: Generator neural net
        nz: hidden size of noise input for Generator
        dataset_train: Training datsaset
        dataset_dev: Validation dataset
        device: gpu/cpu device
        optimizerD: optimizer for discriminator
        optimizerG: optimizer for generator
        img_list: list of generated images while training for progess visualization
        G_losses: history of generator losses
        D_losses: history of discriminator losses
        inc_scores: history of incepotion scores for each epoch
        best_epoch: best epoch so far
        start_epoch: starting epoch used for continuing training
        no_improve_count: amount of epochs since the model has not improived on the dev dataset
        ls_loss: whether or not to use least square loss
        sloppy: flag for 50 isntead of 1000 images for calculating FID/IS scores each epoch

    """

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
            if not ls_loss:
                errD_real = criterion(output, label)
            else:
                errD_real = 0.5 * ((output - label) ** 2).mean()
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
            if not ls_loss:
                errD_fake = criterion(output, label)
            else:
                errD_fake = 0.5 * ((output - label) ** 2).mean()
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
            if not ls_loss:
                errG = criterion(output, label)
            else:
                errG = 0.5 * ((output - label) ** 2).mean()
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

        # Sloppy means that we only calculate 50 images per class instead of 1000 because of heavy computation time
        # This results in inaccurate scores though
        num_img = 50 if sloppy else 1000
        # Calc IS/FID/class-FID for each Epoch
        gen_imgs = me.gen_images(netG, device, nz, num_img_per_class=num_img)
        # Calc IS
        is_mean, is_std = me.inception_score_torchmetrics(gen_imgs)
        inc_scores.append((is_mean, is_std))
        best_epoch = epoch

        # Save best pt, compare mean.
        if is_mean >= max(inc_scores, key=lambda x: x[0])[0]:
            # Save best
            torch.save({
                'img_list': img_list,
                'netD_state_dict': netD.state_dict(),
                'netG_state_dict': netG.state_dict(),
                'optimizer_g_state_dict': optimizerG.state_dict(),
                'optimizer_d_state_dict': optimizerD.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses,
                'inc_scores': inc_scores,
                'start_epoch': epoch,
                'best_epoch': best_epoch,
                'no_improve_count': no_improve_count
            }, model_path / f'model_best.pth')
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Save training data every epoch.
        torch.save({
            'img_list': img_list,
            'netD_state_dict': netD.state_dict(),
            'netG_state_dict': netG.state_dict(),
            'optimizer_g_state_dict': optimizerG.state_dict(),
            'optimizer_d_state_dict': optimizerD.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses,
            'inc_scores': inc_scores,
            'start_epoch': epoch,
            'best_epoch': best_epoch,
            'no_improve_count': no_improve_count
        }, model_path / f'model_{epoch}.pth')


        # Output training stats
        print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item()}\tLoss_G:'
              + f'{errG.item()}\tD(x): {D_x}\tD(G(z)): {D_G_z1} / {D_G_z2}\tIS-mean: {is_mean}')
