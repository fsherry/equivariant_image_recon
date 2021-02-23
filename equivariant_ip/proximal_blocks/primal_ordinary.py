import torch


class ResBlock(torch.nn.Module):
    def __init__(self, conv_params, channels=96, nonlin=None, init_as_id=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, **conv_params)
        self.nonlin = torch.nn.LeakyReLU() if nonlin is None else nonlin
        self.block = torch.nn.Sequential(
            self.conv,
            torch.nn.LeakyReLU() if nonlin is None else nonlin)
        if init_as_id:
            self._set_equal_to_identity()

    def forward(self, x):
        return x + self.block(x)

    def _set_equal_to_identity(self):
        self.block[0].weight.data.zero_()
        self.block[0].bias.data.zero_()


class PrimalProximalOrdinary(torch.nn.Module):
    def __init__(self,
                 conv_params,
                 in_channels=1,
                 n_memory=5,
                 n_res_blocks=1,
                 channels=96,
                 nonlin=None,
                 init_as_id=True):
        super().__init__()
        self.conv_params = conv_params
        self.in_channels = in_channels
        self.n_memory = n_memory
        self.first_part = torch.nn.Sequential(
            torch.nn.Conv2d(2 * in_channels + n_memory, channels,
                            **conv_params),
            torch.nn.LeakyReLU() if nonlin is None else nonlin)
        self.middle_part = torch.nn.Sequential(*[
            ResBlock(conv_params,
                     channels=channels,
                     nonlin=nonlin,
                     init_as_id=init_as_id) for _ in range(n_res_blocks)
        ])
        self.final_part = torch.nn.Conv2d(channels, in_channels + n_memory,
                                          **conv_params)
        if init_as_id:
            self._set_equal_to_identity()

    def forward(self, x, g, s):
        z = torch.cat((x, g, s), dim=1)
        z = self.first_part(z)
        z = self.middle_part(z)
        z = self.final_part(z)
        x_out = z[:, :self.in_channels, :, :]
        s_out = z[:, self.in_channels:, :, :]
        return (x_out, s_out)

    def _set_equal_to_identity(self):
        self.first_part[0].weight.data.zero_()
        self.first_part[0].bias.data.zero_()
        kernel_size = self.conv_params['kernel_size']
        for i in range(self.in_channels):
            self.first_part[0].weight.data[i, i, kernel_size // 2,
                                           kernel_size // 2] = 1.
        for i in range(self.n_memory):
            self.first_part[0].weight.data[self.in_channels + i,
                                           2 * self.in_channels + i,
                                           kernel_size // 2,
                                           kernel_size // 2] = 1.
        for block in self.middle_part:
            block._set_equal_to_identity()
        self.final_part.weight.data.zero_()
        self.final_part.bias.data.zero_()
        for i in range(self.in_channels + self.n_memory):
            self.final_part.weight.data[i, i, 1, 1] = 1.
