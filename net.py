import torch
from torch import nn

def _weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    elif class_name.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class SEMerge(nn.Module):

    def __init__(self, channels, reduction_ratio=16):
        super(SEMerge, self).__init__()
        self.channels = channels
        self.reduced_channels = max(channels // reduction_ratio, 8)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels * 2, self.reduced_channels, kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(self.reduced_channels, channels * 2, kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, x1, x2):
        assert x1.shape == x2.shape, 'Input feature maps must have the same shape!'
        x_cat = torch.cat([x1, x2], dim=1)
        se_weights = self.se(x_cat)
        (w1, w2) = torch.split(se_weights, [self.channels, self.channels], dim=1)
        out = x1 * w1 + x2 * w2
        return out

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, mid_channels or out_channels, kernel_size=3, padding=1, bias=True), nn.BatchNorm2d(mid_channels or out_channels), nn.ELU(), nn.Conv2d(mid_channels or out_channels, out_channels, kernel_size=3, padding=1, bias=True), nn.BatchNorm2d(out_channels), nn.ELU())

    def forward(self, x):
        return self.layers(x)

class SingleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True), nn.BatchNorm2d(out_channels), nn.ELU())

    def forward(self, x):
        return self.layers(x)

class CirBaseUnet(nn.Module):
    """
    Unet:
        default: 16, 32, 64, 128, 256
    """

    def __init__(self, in_channel, out_channel, activation=None, mode='single'):
        super().__init__()
        self.mode = mode
        filters = [in_channel]
        for i in range(5):
            filters.append(14 + i * 4)
        self.EncodeBlock = nn.ModuleList()
        self.DecodeBlock = nn.ModuleList()
        for i in range(len(filters) - 2):
            if mode == 'start' or mode == 'single' or i == 0:
                self.EncodeBlock.append(DoubleConv(filters[i], filters[i + 1]))
            else:
                self.EncodeBlock.append(DoubleConv(filters[i] + filters[i + 1], filters[i + 1]))
        for i in range(1, len(filters) - 1):
            self.DecodeBlock.append(nn.Upsample(scale_factor=2))
            self.DecodeBlock.append(nn.Sequential(SingleConv(filters[-i - 1] + filters[-i], filters[-i - 1] // 2), DoubleConv(filters[-i - 1] // 2, filters[-i - 1])))
        self.MaxPool = nn.MaxPool2d(2)
        self.middleConv = DoubleConv(filters[-2], filters[-1])
        self.finalConv = nn.Conv2d(filters[1], out_channel, kernel_size=3, stride=1, padding=1)
        if activation is not None:
            self.isActivation = True
            if activation == 'softmax':
                self.activation = nn.Softmax(dim=1)
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            else:
                raise
        else:
            self.isActivation = False

    def forward(self, x, above_features=None):
        features = []
        for i in range(len(self.EncodeBlock)):
            if self.mode == 'start' or self.mode == 'single' or i == 0:
                x = self.EncodeBlock[i](x)
            else:
                x = self.EncodeBlock[i](torch.cat([x, above_features[-i]], dim=1))
            features.append(x)
            x = self.MaxPool(x)
        x = self.middleConv(x)
        next_features = []
        for i in range(0, len(self.DecodeBlock), 2):
            x = self.DecodeBlock[i](x)
            x = self.DecodeBlock[i + 1](torch.cat([features[-i // 2 - 1], x], dim=1))
            next_features.append(x)
        next_features.pop()
        res = self.activation(self.finalConv(x)) if self.isActivation else self.finalConv(x)
        if self.mode == 'end' or self.mode == 'single':
            return res
        else:
            return (res, next_features)

class CirUnet(nn.Module):

    def __init__(self, in_channel, out_channel, depth):
        super(CirUnet, self).__init__()
        self.layers = nn.ModuleList()
        self.binary_out = nn.ModuleList()
        for idx in range(depth):
            self.layers.append(CirBaseUnet(in_channel, out_channel, activation='sigmoid', mode='start') if idx == 0 else CirBaseUnet(in_channel + out_channel, out_channel, activation='sigmoid', mode='middle') if idx != depth - 1 else CirBaseUnet(in_channel + out_channel, out_channel, activation='sigmoid', mode='end'))

    def forward(self, img):
        out = []
        (y, above_features) = (None, None)
        for (idx, layer) in enumerate(self.layers):
            if idx == 0:
                (y, above_features) = layer(img)
            elif idx < len(self.layers) - 1:
                (y, above_features) = layer(torch.cat([img, y], dim=1), above_features)
            else:
                y = layer(torch.cat([img, y], dim=1), above_features)
            out.append(y)
        return out

class TransformationBlock(nn.Module):

    def __init__(self, in_channels, depth):
        super().__init__()
        filters = [14 + d * 4 for d in range(depth)]
        self.depth = depth
        self.conv_list = nn.ModuleList()
        for ii in range(depth):
            self.conv_list.append(nn.Sequential(nn.Conv2d(in_channels, filters[ii], kernel_size=1), nn.LeakyReLU(0.1, inplace=True)))

    def forward(self, x_list):
        x_list = [conv(x) for (x, conv) in zip(x_list, self.conv_list)]
        return x_list

class ProgressiveBlock(nn.Module):

    def __init__(self, in_channel, out_channel, depth=5, alpha=0.5, progressive_mode='dual'):
        super().__init__()
        filters = [14 + d * 4 for d in range(depth + 1)]
        self.encoder = nn.ModuleList()
        for i in range(depth):
            self.encoder.append(DoubleConv(in_channel if not i else filters[i - 1], filters[i]))
        self.middle = DoubleConv(filters[-2], filters[-1])
        self.decoder = nn.ModuleList()
        self.up_list = nn.ModuleList()
        for i in range(depth):
            self.up_list.append(nn.Sequential(nn.Conv2d(filters[-1 - i], filters[-2 - i], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(filters[-2 - i]), nn.LeakyReLU(0.1, inplace=True)))
            self.decoder.append(GateBlockModule(filters[-2 - i], filters[-2 - i], alpha=alpha, progressive_mode=progressive_mode))
        self.out_conv = nn.Conv2d(in_channel + filters[0], out_channel, kernel_size=1, bias=False)

    def forward(self, x, y_list):
        residual_x = x
        q_list = []
        mid_result = []
        for layer in self.encoder:
            x = layer(x)
            q_list.append(x)
        x = self.middle(x)
        for (layer, up) in zip(self.decoder, self.up_list):
            x = up(x)
            x = layer(x, y_list.pop(), q_list.pop())
            mid_result.append(x)
        res = self.out_conv(torch.cat([x, residual_x], dim=1))
        return (res, mid_result)

class MultiFusion(nn.Module):

    def __init__(self, hidden_channel, alpha=0.5, progressive_mode='dual'):
        super().__init__()
        self.progressive_mode = progressive_mode
        self.create_qkv = nn.Conv2d(hidden_channel, 3 * hidden_channel, kernel_size=1)
        self.create_y = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1)
        self.create_z = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1)
        if progressive_mode not in ['dual', 'hard', 'soft']:
            self.mid_conv_v = nn.Conv2d(2 * hidden_channel, hidden_channel, kernel_size=1)
        else:
            self.mid_conv_v = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=1)
        self.mid_conv_v_2 = nn.Conv2d(2 * hidden_channel, hidden_channel, kernel_size=1)
        self.mid_conv_k = nn.Conv2d(2 * hidden_channel, hidden_channel, kernel_size=3, padding=1)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Tanh()
        self.act3 = nn.ReLU()
        self.conv = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=1)
        self.conv_end = nn.Conv2d(2 * hidden_channel, hidden_channel, kernel_size=1)
        self.SE = SEMerge(hidden_channel)
        self.alpha = alpha

    def forward(self, x, y, z):
        (b, _, h, w) = x.shape
        (q, k, v) = self.create_qkv(x).reshape(b, -1, 3, h, w).permute(2, 0, 1, 3, 4).unbind(0)
        (y, z) = (self.create_y(y), self.create_z(z))
        q = self.act1(q)
        k = self.act2(k)
        if self.progressive_mode == 'dual':
            v = self.act1(self.mid_conv_v_2(torch.cat([self.alpha * self.act3(self.mid_conv_v(v + y)), (1 - self.alpha) * self.act3(self.mid_conv_v(v * y))], dim=1)))
        elif self.progressive_mode == 'hard':
            v = self.act1(self.mid_conv_v(v + y))
        elif self.progressive_mode == 'soft':
            v = self.act1(self.mid_conv_v(v * y))
        elif self.progressive_mode == 'SE':
            v = self.SE(v, y)
        else:
            v = self.act1(self.mid_conv_v(torch.cat((v, y), dim=1)))
        res = self.conv(q * k + v)
        return self.conv_end(torch.cat([res, z], dim=1))

class GateBlockModule(nn.Module):

    def __init__(self, in_channels, out_channels, alpha=1.0, progressive_mode='dual'):
        super().__init__()
        self.layer1 = MultiFusion(in_channels, alpha=alpha, progressive_mode=progressive_mode)
        self.layer2 = SingleConv(in_channels, out_channels)

    def forward(self, q, k, v):
        x = self.layer1(q, k, v)
        return self.layer2(x)

class Progressive(nn.Module):

    def __init__(self, in_channel, out_channel, depth, alpha=0.5, progressive_mode='dual'):
        super().__init__()
        self.trans = TransformationBlock(in_channel, depth)
        self.progressive = ProgressiveBlock(in_channel, out_channel, depth, alpha=alpha, progressive_mode=progressive_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y_list):
        y_list = self.trans(y_list)
        (res, mid_res) = self.progressive(x, y_list)
        return (torch.sigmoid(res), mid_res)

class PCNet(nn.Module):

    def __init__(self, in_channel, out_channel, depth, alpha, progressive_mode='dual'):
        super().__init__()
        self.module1 = CirUnet(in_channel=in_channel, out_channel=out_channel, depth=depth)
        self.module2 = Progressive(in_channel=in_channel, out_channel=out_channel, depth=depth, alpha=alpha, progressive_mode=progressive_mode)
        self.apply(_weights_init)

    def forward(self, x):
        x_list = self.module1(x)
        (t, mid_res) = self.module2(x, x_list)
        return (x_list, t, mid_res)
