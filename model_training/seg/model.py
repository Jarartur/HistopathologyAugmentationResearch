# ---------------------------------------------------------#
# Plain PyTorch implementation of the segmentation model. #
# ---------------------------------------------------------#

# internal imports
# %%
# pytorch ecosystem imports
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

# additional imports
import math


# %%
class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, stride=1, padding=1),
            nn.GroupNorm(output_size, output_size),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(output_size, output_size, 3, stride=1, padding=1),
            nn.GroupNorm(output_size, output_size),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.conv = nn.Sequential(nn.Conv2d(input_size, output_size, 1))

    def forward(self, x):
        return self.module(x) + self.conv(x)


# %%
class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 4, stride=2, padding=1),
            nn.GroupNorm(input_channels, input_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.encoder_2 = nn.Sequential(
            ResidualBlock(input_channels, 32),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.encoder_3 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.encoder_4 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder_4 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder_3 = nn.Sequential(
            ResidualBlock(192, 64),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder_2 = nn.Sequential(
            ResidualBlock(96, 32),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder_1 = nn.Sequential(
            ResidualBlock(32 + input_channels, output_channels),
            nn.ConvTranspose2d(
                output_channels,
                output_channels,
                4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.GroupNorm(output_channels, output_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 1),
        )

    def pad(self, image, template):
        pad_x = math.fabs(image.size(3) - template.size(3))
        pad_y = math.fabs(image.size(2) - template.size(2))
        b_x, e_x = math.floor(pad_x / 2), math.ceil(pad_x / 2)
        b_y, e_y = math.floor(pad_y / 2), math.ceil(pad_y / 2)
        image = F.pad(image, (b_x, e_x, b_y, e_y))
        return image

    def forward(self, x):
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.encoder_4(x3)
        d4 = self.decoder_4(x4)
        d4 = self.pad(d4, x3)
        d3 = self.decoder_3(tc.cat((d4, x3), dim=1))
        d3 = self.pad(d3, x2)
        d2 = self.decoder_2(tc.cat((d3, x2), dim=1))
        d2 = self.pad(d2, x1)
        d1 = self.decoder_1(tc.cat((d2, x1), dim=1))
        d1 = self.pad(d1, x)
        result = self.last_layer(d1)
        return result


# def load_network(weights_path=None):
#     """
#     Utility function to load the network.
#     """
#     model = UNet()
#     if weights_path is not None:
#         model.load_state_dict(tc.load(weights_path))
#         model.eval()
#     return model
# %%



if __name__ == "__main__":
    import torchsummary as ts
    def test_forward_pass():
        device = "cpu"
        model = UNet(1, 1).to(device)
        y_size, x_size = 64, 64
        no_channels = 1
        batch_size = 2
    
        example_input = tc.rand((batch_size, no_channels, y_size, x_size)).to(device)
        result = model(example_input)
        print("Result size: ", result.size())
        ts.summary(model, (no_channels, y_size, x_size), device=device)
    test_forward_pass()
