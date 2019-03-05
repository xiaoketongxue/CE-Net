import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class Block_D(nn.Module):
    def __init__(self, inplanes):
        super(Block_D, self).__init__()

        self.in_channel = inplanes
        self.conv_1x1 = nn.Conv2d(in_channels=32, out_channels=self.in_channel, kernel_size=1, stride=1,
                                  padding=0)

        self.conv_3x3_d1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1,
                                     stride=1, padding=0, dilation=1)
        self.conv_3x3_d3 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=3,
                                     stride=1, padding=3, dilation=3)
        self.conv_3x3_d5 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=3,
                                     stride=1, padding=5, dilation=5)

        self.bn = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()

        self.conv_c = nn.Conv2d(in_channels=self.in_channel, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)


        self._init_weight()

    def forward(self, x):
        x = self.conv_1x1(x)

        x_1 = self.conv_3x3_d1(x)
        x_2 = self.conv_3x3_d3(x)
        x_3 = self.conv_3x3_d5(x)

        x = x + x_1 + x_2 + x_3
        x = self.relu(self.bn(x))
        x = self.conv_c(x)
        x = self.max_pooling(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Block_U(nn.Module):
    def __init__(self, inplanes, size):
        super(Block_U, self).__init__()

        self.in_channel = inplanes
        self.size = size
        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=self.in_channel, kernel_size=1, stride=1,
                                  padding=0)

        self.conv_3x3_d1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1,
                                     stride=1, padding=0, dilation=1)
        self.conv_3x3_d3 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=3,
                                     stride=1, padding=3, dilation=3)
        self.conv_3x3_d5 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=3,
                                     stride=1, padding=5, dilation=5)

        self.bn = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()

        self.conv_c = nn.Conv2d(in_channels=self.in_channel, out_channels=32, kernel_size=1, stride=1, padding=0)


        self._init_weight()

    def forward(self, x):
        x = self.conv_1x1(x)

        x_1 = self.conv_3x3_d1(x)
        x_2 = self.conv_3x3_d3(x)
        x_3 = self.conv_3x3_d5(x)

        x = x + x_1 + x_2 + x_3
        x = self.relu(self.bn(x))
        x = self.conv_c(x)
        x = F.upsample(x, size=[self.size, self.size], mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class Block_decoder(nn.Module):
    def __init__(self, inplanes):
        super(Block_decoder, self).__init__()

        self.in_channel = inplanes
        self.conv_1x1 = nn.Conv2d(in_channels=32, out_channels=self.in_channel, kernel_size=1, stride=1,
                                  padding=0)

        self.conv_3x3_d1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1,
                                     stride=1, padding=0, dilation=1)
        self.conv_3x3_d3 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=3,
                                     stride=1, padding=3, dilation=3)
        self.conv_3x3_d5 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=3,
                                     stride=1, padding=5, dilation=5)

        self.bn = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv2d(in_channels=self.in_channel, out_channels=32, kernel_size=1, stride=1, padding=0)

        self._init_weight()

    def forward(self, x):
        x = self.conv_1x1(x)

        x_1 = self.conv_3x3_d1(x)
        x_2 = self.conv_3x3_d3(x)
        x_3 = self.conv_3x3_d5(x)

        x = x + x_1 + x_2 + x_3
        x = self.relu(self.bn(x))
        x = self.conv_c(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




class BRU_net(nn.Module):

    def __init__(self, n_classes):
        n_filters = [32, 64, 92, 160, 256]
        super(BRU_net, self).__init__()

        self.input_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.BRU_Dnet0 = Block_D(n_filters[0])
        self.BRU_Dnet1 = Block_D(n_filters[1])
        self.BRU_Dnet2 = Block_D(n_filters[2])
        self.BRU_Dnet3 = Block_D(n_filters[3])
        self.BRU_Dnet4 = Block_D(n_filters[4])

        self.encoder = Block_decoder(n_filters[4])

        self.BRU_Unet4 = Block_U(n_filters[4], 32)
        self.BRU_Unet3 = Block_U(n_filters[3], 64)
        self.BRU_Unet2 = Block_U(n_filters[2], 128)
        self.BRU_Unet1 = Block_U(n_filters[1], 256)
        self.BRU_Unet0 = Block_U(n_filters[0], 512)

        self.output_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv2d(in_channels=32, out_channels=n_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, input):

        input_feature = self.relu(self.bn(self.input_conv(input)))


        block0 = self.BRU_Dnet0(input_feature)

        block1 = self.BRU_Dnet1(block0)

        block2 = self.BRU_Dnet2(block1)

        block3 = self.BRU_Dnet3(block2)

        block4 = self.BRU_Dnet4(block3)

        encoder = self.encoder(block4)

        decoder4 = self.BRU_Unet4(torch.cat([encoder, block4], 1))


        decoder3 = self.BRU_Unet3(torch.cat([decoder4, block3], 1))

        decoder2 = self.BRU_Unet2(torch.cat([decoder3, block2], 1))

        decoder1 = self.BRU_Unet1(torch.cat([decoder2, block1], 1))

        decoder0 = self.BRU_Unet0(torch.cat([decoder1, block0], 1))

        output = self.conv1x1(self.relu(self.bn(self.output_conv(decoder0))))



        return output

if __name__ == "__main__":
    model = BRU_net(n_classes=12)
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        segmentation_output = model.forward(image)

    print(segmentation_output.size())



