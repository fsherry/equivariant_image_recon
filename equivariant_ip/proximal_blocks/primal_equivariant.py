import e2cnn
import torch

_N = 6
default_base_space = e2cnn.gspaces.Rot2dOnR2(_N)
default_feat_type = e2cnn.nn.FieldType(default_base_space,
                                       16 * [default_base_space.regular_repr])


def leakyrelu_geometric(x, negative_slope=0.01):
    # Include warning if pointwise nonlinearity does not make sense for field type
    return e2cnn.nn.GeometricTensor(
        torch.nn.functional.leaky_relu(x.tensor, negative_slope), x.type)


class LeakyReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, negative_slope=0.01):
        return leakyrelu_geometric(x, negative_slope)


class ResBlock(torch.nn.Module):
    def __init__(self, conv_params, feat_type, nonlin=None, init_as_id=True):
        super().__init__()
        self.block = torch.nn.Sequential(
            e2cnn.nn.R2Conv(feat_type, feat_type, **conv_params),
            LeakyReLU() if nonlin is None else nonlin)
        if init_as_id:
            self._set_equal_to_identity()

    def forward(self, x):
        return x + self.block(x)

    def _set_equal_to_identity(self):
        self.block[0].weights.data.zero_()
        self.block[0].bias.data.zero_()


class PrimalProximalEquivariant(torch.nn.Module):
    def __init__(self,
                 conv_params,
                 in_channels=1,
                 n_memory=5,
                 n_res_blocks=1,
                 feat_type_intermed=default_feat_type,
                 nonlin=None,
                 init_as_id=True):
        super().__init__()
        self.in_channels = in_channels
        self.n_memory = n_memory
        self.base_space = feat_type_intermed.gspace
        self.feat_type_in = e2cnn.nn.FieldType(self.base_space,
                                               (2 * in_channels + n_memory) *
                                               [self.base_space.trivial_repr])
        self.feat_type_out = e2cnn.nn.FieldType(self.base_space,
                                                (in_channels + n_memory) *
                                                [self.base_space.trivial_repr])
        self.nonlin = LeakyReLU() if nonlin is None else nonlin

        self.first_part = torch.nn.Sequential(
            e2cnn.nn.R2Conv(self.feat_type_in, feat_type_intermed,
                            **conv_params), self.nonlin)
        self.middle_part = torch.nn.Sequential(*[
            ResBlock(conv_params,
                     feat_type_intermed,
                     nonlin=self.nonlin,
                     init_as_id=init_as_id) for _ in range(n_res_blocks)
        ])
        self.final_part = e2cnn.nn.R2Conv(feat_type_intermed,
                                          self.feat_type_out, **conv_params)
        if init_as_id:
            self._set_equal_to_identity()

    def forward(self, x, g, s):
        z = torch.cat((x, g, s), dim=1)
        z = e2cnn.nn.GeometricTensor(z, self.feat_type_in)
        z = self.first_part(z)
        z = self.middle_part(z)
        z = self.final_part(z)
        x_out = z.tensor[:, :self.in_channels, :, :]
        s_out = z.tensor[:, self.in_channels:, :, :]
        return (x_out, s_out)

    def _set_equal_to_identity(self):
        self.first_part[0].weights.data.zero_()
        self.first_part[0].bias.data.zero_()
        input_width = 2 * self.in_channels + self.n_memory
        order = self.first_part[0].out_type.representation.group.order()
        num_channels = self.first_part[0].out_type.representation.size // order
        for i in range(self.in_channels):
            self.first_part[0].weights.data[order * (input_width + 1) * i] = 1.
        for i in range(self.n_memory):
            self.first_part[0].weights.data[order *
                                            (self.in_channels +
                                             (input_width + 1) *
                                             (i + self.in_channels))] = 1.
        for block in self.middle_part:
            block._set_equal_to_identity()
        self.final_part.weights.data.zero_()
        self.final_part.bias.data.zero_()
        for i in range(self.in_channels + self.n_memory):
            self.final_part.weights.data[order * (num_channels + 1) * i] = 1.
