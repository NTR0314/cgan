from torch import nn


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, lrelu=False, suppres_first_relu=False, down_sample=True, sn=True, do=False, learnable_sc=True, residual=True):
        super().__init__()
        self.in_ch = in_ch
        self.residual = residual
        self.out_ch = out_ch
        self.ds = down_sample
        self.lrelu = lrelu
        self.learnable_sc = learnable_sc
        self.suppres_first_relu = suppres_first_relu
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.dof = do
        if self.dof:
            self.do = nn.Dropout2d(p=0.5)
        if self.ds:
            self.c2 = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        else:
            self.c2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        if sn:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
        if self.learnable_sc:
            self.sc_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
            if sn:
                self.sc_conv = nn.utils.spectral_norm(self.sc_conv)

    def forward(self, x):
        residual = x
        if self.residual:
            if self.learnable_sc:
                residual = self.sc_conv(residual)
        if self.ds:
            residual = nn.AvgPool2d(kernel_size=2)(residual)


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
        if self.dof:
            x = self.do(x)

        return x + residual


class GeneratorBlock(nn.Module):
    def __init__(self, ngf, bn=True, tconv=True, residual=True, do=False, learnable_sc=True):
        super().__init__()
        self.bn = bn
        self.tconv = tconv
        self.residual = residual
        self.learnable_sc = learnable_sc

        self.tconv1 = nn.ConvTranspose2d(ngf, ngf, 4, 2, 1)

        self.bn1 = nn.BatchNorm2d(ngf)
        self.bn2 = nn.BatchNorm2d(ngf)

        self.c1 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1)
        self.sc_conv = nn.Conv2d(ngf, ngf, kernel_size=1, padding=0)
        self.dof = do
        if self.dof:
            self.do = nn.Dropout2d(p=0.5)

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
        if self.dof:
            x = self.do(x)
        if self.residual:
            if self.learnable_sc:
                residual = self.sc_conv(nn.Upsample(scale_factor=2)(orig))
                print(residual.shape)
                return x + residual
            else:
                return x + nn.Upsample(scale_factor=2)(x)
        else:
            return x


class UpsampleConv(nn.Module):
    def __init__(self, in_feat, out_feat, scale_factor=2, sn=False):
        super().__init__()
        self.us = nn.Upsample(scale_factor=scale_factor)
        self.c2d = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1)
        if sn:
            self.c2d = nn.utils.spectral_norm(self.c2d)

    def forward(self, x):
        return self.c2d(self.us(x))
