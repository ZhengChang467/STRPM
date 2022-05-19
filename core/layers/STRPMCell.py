import torch
import torch.nn as nn


class STRPMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, frame_channel, tau):
        super(STRPMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = (filter_size[0] // 2, filter_size[1] // 2)
        self._forget_bias = 1.0
        self.frame_channel = frame_channel
        self.en_t = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.en_s = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.en_o = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)
        self.att_c = nn.Sequential(
            nn.Conv2d(num_hidden * tau, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.att_m = nn.Sequential(
            nn.Conv2d(num_hidden * tau, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_sto = nn.Conv2d(num_hidden * 3, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, t_t, s_t, o_t, h_t, c_t, m_t, c_att, m_att):
        # print('yes')
        x_t = self.en_t(t_t)
        x_s = self.en_s(s_t)
        o_x = self.en_o(o_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x = x_t
        f_x = x_t
        g_x = x_t
        i_x_prime = x_s
        f_x_prime = x_s
        g_x_prime = x_s

        i_h = h_concat
        f_h = h_concat
        g_h = h_concat
        o_h = h_concat
        i_m = m_concat
        f_m = m_concat
        g_m = m_concat
        # Temporal module
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        # EAIFG_temporal
        c_att_merge = self.att_c(torch.cat(c_att, dim=1))
        c_new = i_t * g_t + f_t * c_att_merge
        # Spatial Module
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        # EIFAG_spatial
        m_att_merge = self.att_m(torch.cat(m_att, dim=1))
        m_new = i_t_prime * g_t_prime + f_t_prime * m_att_merge

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        # add resuial informaiton
        if (t_t - s_t).mean() == 0.0:
            output = h_new + t_t
        else:
            output = h_new + self.conv_sto(torch.cat([t_t, s_t, o_t], dim=1))
        return output, o_t, c_new, m_new


