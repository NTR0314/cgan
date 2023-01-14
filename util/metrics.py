import random
import numpy as np
import torch
from torchmetrics.image.inception import InceptionScore
from scipy.stats import entropy
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from torchmetrics.image.fid import FrechetInceptionDistance

def gen_images(generator, device, nz, num_img_per_class=100):
    # Set neccessary seeds for reproducability
    np.random.seed(1337)
    torch.manual_seed(1337)
    random.seed(1337)

    # Generate 1000 images 100 per class as the ex sheet states.
    gen_imgs = []
    labels = torch.arange(0, 10).repeat(num_img_per_class).to(device)
    generator.eval()
    for i in range(len(labels)):
        noise = torch.randn(1, nz, 1, 1, device=device)
        gen_imgs.append(generator(noise, labels[i].unsqueeze(-1)).squeeze().detach().cpu())
    generator.train()

    return torch.stack(gen_imgs)

def gen_images_class(generator, device, nz, num_imgs, class_id):
    np.random.seed(1337)
    torch.manual_seed(1337)
    random.seed(1337)

    gen_imgs = []
    labels = torch.tensor(class_id).repeat(num_imgs).to(device)
    generator.eval()
    for i in range(num_imgs):
        noise = torch.randn(1, nz, 1, 1, device=device)
        gen_imgs.append(generator(noise, labels[i].unsqueeze(-1)).squeeze().detach().cpu())
    generator.train()

    return torch.stack(gen_imgs)


def inception_score_own(imgs, device, cuda=True, batch_size=128, upscale=False, splits=1):
    # Evaluate model
    # Inception score
    # Idee: Nutze pretrained inceptionv3 Klassifikator und berechne zuerst die marginale Distribution von
    # random gesampleten images vom Generator p(y), wobei y die Label bezeichne
    # Berechne dann noch p(y|x) für jedes weitere generierte Bild x
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    # How should N be chosen? 50.000 as the original paper suggests: Improved Techniques for Training GANs
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    #     print(torch.cuda.memory_summary(device=None, abbreviated=True))

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    # Inceptionv3 was trained on 299x299 images
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if upscale:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=-1).data.cpu().numpy()

    # Get predictions: N x 1000, weil wir N viele Bilder reinfeeden
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batch.to(device)
        batch_size_i = batch.size()[0]
        # Save batch at correct positions.
        #         print(f"Before getting predictions of batch:")
        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batch)
    #         print(f"After getting batch predictions:")

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        # Ist py, weil fuer jedes Element aus axis!=0, also 1 (Was einem Label entspricht) der Mitterlwert genommen wird
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def inception_score_torchmetrics(imgs):
    """

    :param imgs: Tensor of images -> batch x rgb (3) x height x width
    :return: (mean, std)
    """
    inception = InceptionScore(normalize=True)
    # update images
    inception.update(imgs)

    # Returns tuple (mean, std)
    return inception.compute()


def FID_torchmetrics(imgs, reals):
    """

    :param imgs: Tensor of images -> batch x rgb (3) x height x width
    :return: FID scalar tensor
    """
    fid = FrechetInceptionDistance(normalize=True)
    # update images
    fid.update(imgs, real=False)
    fid.update(reals, real=True)

    # Returns tuple (mean, std)
    return fid.compute()


