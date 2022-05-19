import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import STRPM
from core.models.Discriminator import Discriminator
import torch.optim.lr_scheduler as lr_scheduler


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.patch_height = configs.img_height // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_channel = configs.img_channel * (configs.patch_size ** 2)
        self.num_layers = configs.num_layers
        networks_map = {
            'strpm': STRPM.RNN
        }
        num_hidden = []
        for i in range(configs.num_layers):
            num_hidden.append(configs.num_hidden)
        self.num_hidden = num_hidden
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        # print("Network state:")
        # for param_tensor in self.network.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        #     print(param_tensor, '\t', self.network.state_dict()[param_tensor].size())
        if configs.dataset == 'sjtu4k':
            from core.models.Discriminator_4k import Discriminator
        else:
            from core.models.Discriminator import Discriminator
        self.Discriminator = Discriminator(self.patch_height, self.patch_width, self.patch_channel,
                                           self.configs.D_num_hidden).to(self.configs.device)
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.optimizer_D = Adam(self.Discriminator.parameters(), lr=configs.lr_d)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=configs.lr_decay)
        self.scheduler_D = lr_scheduler.ExponentialLR(self.optimizer_D, gamma=configs.lr_decay)
        self.MSE_criterion = nn.MSELoss()
        self.D_criterion = nn.BCELoss()
        self.L1_loss = nn.L1Loss()

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model_pm.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save predictive model to %s" % checkpoint_path)

        stats = {}
        stats['net_param'] = self.Discriminator.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model_d.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save discriminator model to %s" % checkpoint_path)

    def load(self, pm_checkpoint_path, d_checkpoint_path):
        print('load predictive model:', pm_checkpoint_path)
        stats = torch.load(pm_checkpoint_path, map_location=torch.device(self.configs.device))
        self.network.load_state_dict(stats['net_param'])

        print('load discriminator model:', d_checkpoint_path)
        stats = torch.load(d_checkpoint_path, map_location=torch.device(self.configs.device))
        self.Discriminator.load_state_dict(stats['net_param'])

    def train(self, frames, mask, itr):
        # print(frames.shape)
        self.network.train()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)

        next_frames = self.network(frames_tensor, mask_tensor)
        ground_truth = frames_tensor[:, 1:]

        batch_size = next_frames.shape[0]
        zeros_label = torch.zeros(batch_size).cuda()
        ones_label = torch.ones(batch_size).cuda()

        # train D
        self.Discriminator.zero_grad()
        d_gen, _ = self.Discriminator(next_frames.detach())
        d_gt, _ = self.Discriminator(ground_truth)
        D_loss = self.D_criterion(d_gen, zeros_label) + self.D_criterion(d_gt, ones_label)
        D_loss.backward(retain_graph=True)
        self.optimizer_D.step()

        self.optimizer.zero_grad()
        d_gen_pre, features_gen = self.Discriminator(next_frames)
        _, features_gt = self.Discriminator(ground_truth)
        loss_l1 = self.L1_loss(next_frames, ground_truth)
        loss_l2 = self.MSE_criterion(next_frames, ground_truth)
        gen_D_loss = self.D_criterion(d_gen_pre, ones_label)
        loss_features = self.MSE_criterion(features_gen, features_gt)
        loss_gen = loss_l2 + 0.01 * loss_features + 0.001 * gen_D_loss
        loss_gen.backward()
        self.optimizer.step()
        if itr >= self.configs.sampling_stop_iter and itr % self.configs.delay_interval == 0:
            self.scheduler.step()
            self.scheduler_D.step()
            print('Lr decay to:%.8f', self.optimizer.param_groups[0]['lr'])
        return next_frames, loss_l1.detach().cpu().numpy(), loss_l2.detach().cpu().numpy(), D_loss.detach().cpu().numpy(), \
               gen_D_loss.detach().cpu().numpy(), loss_features.detach().cpu().numpy(), d_gt.mean().detach().cpu().numpy(), d_gen.mean().detach().cpu().numpy(), d_gen_pre.mean().detach().cpu().numpy(),\

    def test(self, frames, mask):
        self.network.eval()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()
