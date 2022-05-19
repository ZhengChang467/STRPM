import torch
import torch.nn as nn
from core.layers.STRPMCell import STRPMCell
import math


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        # print(configs.srcnn_tf)
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = configs.tau
        # self.time = 2
        cell_list = []

        width = configs.img_width // configs.patch_size // configs.sr_size
        height = configs.img_height // configs.patch_size // configs.sr_size

        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            cell_list.append(
                STRPMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                          configs.stride, self.frame_channel, self.tau)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(configs.sr_size))
        self.encoder_t = nn.Sequential()
        self.encoder_s = nn.Sequential()
        self.encoder_o = nn.Sequential()
        self.encoder_s.add_module(name='encoder_s_conv{0}'.format(-1),
                                  module=nn.Conv2d(in_channels=self.frame_channel,
                                                   out_channels=self.num_hidden[0],
                                                   stride=1,
                                                   padding=0,
                                                   kernel_size=1))
        self.encoder_s.add_module(name='relu_s_{0}'.format(-1),
                                  module=nn.ReLU())

        self.encoder_t.add_module(name='encoder_t_conv{0}'.format(-1),
                                  module=nn.Conv2d(in_channels=self.frame_channel,
                                                   out_channels=self.num_hidden[0],
                                                   stride=1,
                                                   padding=0,
                                                   kernel_size=1))
        self.encoder_t.add_module(name='relu_t_{0}'.format(-1),
                                  module=nn.ReLU())

        self.encoder_o.add_module(name='encoder_o_conv{0}'.format(-1),
                                  module=nn.Conv2d(in_channels=self.frame_channel,
                                                   out_channels=self.num_hidden[0],
                                                   stride=1,
                                                   padding=0,
                                                   kernel_size=1))
        self.encoder_o.add_module(name='relu_o_{0}'.format(-1),
                                  module=nn.ReLU())
        for i in range(n):
            self.encoder_t.add_module(name='encoder_t{0}'.format(i),
                                      module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                       out_channels=self.num_hidden[0],
                                                       stride=(2, 2),
                                                       padding=(1, 1),
                                                       kernel_size=(3, 3)
                                                       ))
            self.encoder_t.add_module(name='encoder_t_relu{0}'.format(i),
                                      module=nn.ReLU())

            self.encoder_s.add_module(name='encoder_s{0}'.format(i),
                                      module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                       out_channels=self.num_hidden[0],
                                                       stride=(2, 2),
                                                       padding=(1, 1),
                                                       kernel_size=(3, 3)
                                                       ))
            self.encoder_s.add_module(name='encoder_s_relu{0}'.format(i),
                                      module=nn.ReLU())

            self.encoder_o.add_module(name='encoder_o{0}'.format(i),
                                      module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                       out_channels=self.num_hidden[0],
                                                       stride=(2, 2),
                                                       padding=(1, 1),
                                                       kernel_size=(3, 3)
                                                       ))
            self.encoder_o.add_module(name='encoder_o_relu{0}'.format(i),
                                      module=nn.ReLU())

        # Decoder
        self.decoder_s = nn.Sequential()
        self.decoder_t = nn.Sequential()
        self.decoder_o = nn.Sequential()
        for i in range(n - 1):
            self.decoder_s.add_module(name='c_decoder{0}'.format(i),
                                      module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                                out_channels=self.num_hidden[-1],
                                                                stride=(2, 2),
                                                                padding=(1, 1),
                                                                kernel_size=(3, 3),
                                                                output_padding=(1, 1)
                                                                ))
            # self.decoder_s.add_module(name='gn_decoder_s{0}'.format(i),
            #                           module=nn.GroupNorm(4, self.frame_channel))
            self.decoder_s.add_module(name='c_decoder_relu{0}'.format(i),
                                      module=nn.ReLU())
            self.decoder_t.add_module(name='m_decoder{0}'.format(i),
                                      module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                                out_channels=self.num_hidden[-1],
                                                                stride=(2, 2),
                                                                padding=(1, 1),
                                                                kernel_size=(3, 3),
                                                                output_padding=(1, 1)
                                                                ))
            self.decoder_t.add_module(name='m_decoder_relu{0}'.format(i),
                                      module=nn.ReLU())
            self.decoder_o.add_module(name='o_decoder{0}'.format(i),
                                      module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                                out_channels=self.num_hidden[-1],
                                                                stride=(2, 2),
                                                                padding=(1, 1),
                                                                kernel_size=(3, 3),
                                                                output_padding=(1, 1)
                                                                ))
            # self.decoder_o.add_module(name='gn_decoder_o{0}'.format(i),
            #                           module=nn.GroupNorm(4, self.frame_channel))
            self.decoder_o.add_module(name='o_decoder_relu{0}'.format(i),
                                      module=nn.ReLU())

        if n > 0:
            self.decoder_s.add_module(name='c_decoder{0}'.format(n - 1),
                                      module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                                out_channels=self.num_hidden[-1],
                                                                stride=(2, 2),
                                                                padding=(1, 1),
                                                                kernel_size=(3, 3),
                                                                output_padding=(1, 1)
                                                                ))
            self.decoder_t.add_module(name='m_decoder{0}'.format(n - 1),
                                      module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                                out_channels=self.num_hidden[-1],
                                                                stride=(2, 2),
                                                                padding=(1, 1),
                                                                kernel_size=(3, 3),
                                                                output_padding=(1, 1)
                                                                ))
            # self.decoder_t.add_module(name='gn_decoder_t{0}'.format(n-1),
            #                           module=nn.GroupNorm(4, self.frame_channel))
            self.decoder_o.add_module(name='o_decoder{0}'.format(n - 1),
                                      module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                                out_channels=self.num_hidden[-1],
                                                                stride=(2, 2),
                                                                padding=(1, 1),
                                                                kernel_size=(3, 3),
                                                                output_padding=(1, 1)
                                                                ))
        self.c_srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.m_srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.o_srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )

        self.conv_last_sr = nn.Conv2d(self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, frames, mask_true):
        # print('ok')
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames.shape[0]
        height = frames.shape[3] // self.configs.sr_size
        width = frames.shape[4] // self.configs.sr_size
        frame_channels = frames.shape[2]
        next_frames = []
        h_t = []
        c_t = []
        c_net = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
        c_net.append(c_t)
        memory = torch.zeros([batch_size, self.num_hidden[0], height, width]).to(self.configs.device)
        m_net = []
        m_net.append(memory)
        for t in range(self.configs.total_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                time_diff = t - self.configs.input_length
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
            input_frm = net
            frames_feature_t = self.encoder_t(input_frm)
            frames_feature_s = self.encoder_s(input_frm)
            frames_feature_o = self.encoder_o(input_frm)

            # c_att
            c_att = []
            m_att = []

            if len(c_net) <= self.configs.tau:
                for idx in range(self.configs.tau):
                    if idx < self.configs.tau - len(c_net):
                        c_att.append(c_net[0][0])
                    else:
                        c_att.append(c_net[idx - self.configs.tau + len(c_net)][0])
            else:
                for idx in range(len(c_net) - self.configs.tau, len(c_net)):
                    c_att.append(c_net[idx][0])
            # m_att.append(m_t)
            if len(m_net) <= self.configs.tau:
                for idx in range(self.configs.tau):
                    if idx < self.configs.tau - len(c_net):
                        m_att.append(m_net[0])
                    else:
                        m_att.append(m_net[idx - self.configs.tau + len(c_net)])
            else:
                for idx in range(len(m_net) - self.configs.tau, len(m_net)):
                    m_att.append(m_net[idx])
            h_t[0], o_t, c_t[0], memory = self.cell_list[0](frames_feature_t,
                                                            frames_feature_s,
                                                            frames_feature_o,
                                                            h_t[0], c_t[0], memory, c_att, m_att)
            m_net.append(memory)
            for i in range(1, self.num_layers):
                # c_att
                c_att = []
                m_att = []

                if len(c_net) <= self.configs.tau:
                    for idx in range(self.configs.tau):
                        if idx < self.configs.tau - len(c_net):
                            c_att.append(c_net[0][i])
                        else:
                            c_att.append(c_net[idx - self.configs.tau + len(c_net)][i])
                else:
                    for idx in range(len(c_net) - self.configs.tau, len(c_net)):
                        c_att.append(c_net[idx][i])

                if len(m_net) <= self.configs.tau:
                    for idx in range(self.configs.tau):
                        if idx < self.configs.tau - len(c_net):
                            m_att.append(m_net[0])
                        else:
                            m_att.append(m_net[idx - self.configs.tau + len(c_net)])
                else:
                    for idx in range(len(m_net) - self.configs.tau, len(m_net)):
                        m_att.append(m_net[idx])

                h_t[i], o_t, c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i - 1], h_t[i - 1], h_t[i], c_t[i],
                                                                memory, c_att, m_att)
                m_net.append(memory)
            c_net.append(c_t)

            m_sr = self.decoder_s(memory)
            c_sr = self.decoder_t(c_t[-1])
            o_sr = self.decoder_o(o_t)

            m_sr = self.m_srcnn(m_sr)
            c_sr = self.c_srcnn(c_sr)
            o_sr = self.o_srcnn(o_sr)

            mem_sr = torch.cat((c_sr, m_sr), 1)
            h_sr = o_sr * torch.tanh(self.conv_last_sr(mem_sr))

            x_gen = h_sr

            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()

        return next_frames
