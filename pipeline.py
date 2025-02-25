import pdb

import h5py
import time
import wandb
import torch
import einops
import scipy.io as sio
from pathlib import Path
from datasets.data import create_loaders
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.util import AverageMeter, plot_feature_maps
from utils.calc_ssim import SSIM


import numpy as np
from collections import defaultdict

def adjust(init, fin, step, fin_step):
    if fin_step == 0:
        return  fin
    deta = fin - init
    adj = min(init + deta * step / fin_step, fin)
    return adj
def create_model(config, device):
    # config:来自configs/config.yaml 中设置好的一些信息
    dataset_name = config.dataset_name
    #in_channels, out_channels = (34, 31)
    in_channels, out_channels = (9, 8)
    if dataset_name in ('GF2', 'QB'):
        in_channels, out_channels = (5, 4)
    elif dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
        in_channels, out_channels = (34, 31)
    model_name = config.model_name # 设定模型的名称
    ms_cnum = out_channels # ms_cnum设置为输出的channels
    pan_cnum = in_channels - out_channels # pan_cnum设置为输入的通道数
    #max_step, hid_cnum = config.max_step, config.hidden_channel # 分别为max_step和hid_cnum赋值
    model = None
    ## TODO: To be fixed
    # if model_name == 'ARN':
    #     model = ARN(in_ms_cnum=ms_cnum, in_pan_cnum=pan_cnum,
    #                 max_step=max_step, hidden_channel=hid_cnum).to(device)
    # else:
    #     assert f'{model_name} not supported now.'
    return model


class Trainer:
    #训练器类
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.writer = None
        dataset_name = config.dataset_name
        self.debug = config.debug
        if not self.debug:
            run_time = logger.handlers[0].baseFilename.split('/')[-1][:-4]
            self.run_time = run_time
            weights_save_path = Path(self.config.weights_path) / dataset_name / run_time # 定义模型权重保存位置weights_save_path
            weights_save_path.mkdir(exist_ok=True, parents=True) # 建立文件夹
            self.weights_save_path = weights_save_path # 为类中元素赋值
            tb_log_path = Path(self.config.tb_log_path) / run_time
            tb_log_path.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(str(tb_log_path)) # 类似的,写入日志

        self.epoch_num = config.epoch_num
        self.train_loader, self.val_loader = create_loaders(config) # 详见create_loaders函数
        base_lr = float(config.base_lr) # 将config中的base_lr转换成float
        device = torch.device(config.device) # 设置device为第0张显卡
        self.device = device
        self.model = create_model(config, device)  # 创建模型

        self.criterion = nn.L1Loss().to(device)  # L1 loss
        #self.freq_loss = CharFreqLoss().to(device) # freq-domain loss for FFT
        self.ssim_loss = SSIM(size_average=True).to(device)
        # self.mi_loss = Mutual_info_reg(input_channels=64,channels=64).to(device)
        #self.mi_loss = Mutual_info_reg(8//2,8//2).to(device)
        # self.criterion = dual_domain_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr, betas=(0.9, 0.999))  # Adam优化器
        step_size, gamma = int(config.step_size), float(config.gamma)                         # 为step_size和gama赋值
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        # 为scheduler赋值
        return

    def train_all(self):
        print('Start training...')
        epoch_time = AverageMeter()
        end = time.time()

        ckpt = self.config.save_epoch # 设置断点checkpoint
        model, optimizer, device = self.model, self.optimizer, self.device
        for epoch in range(self.epoch_num):
            epoch += 1
            epoch_train_loss = []
            epoch_mi_loss = []

            model.train()   # 开始训练模式
            for iteration, batch in enumerate(self.train_loader, 1):
                # gt, lms, ms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                gt, lms, ms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[4].to(device)
                optimizer.zero_grad()  # fixed

                cond = pan.repeat(1,8,1,1) - lms
                out = model(lms, pan, cond) # For InvDemo

                loss = self.criterion(out, gt)  # compute L1-loss

                epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch
                loss.backward()
                optimizer.step()
            self.scheduler.step()
#scale
            t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
            self.logger.info('Epoch: {}/{} training loss:{:.7f}'.format(epoch, self.epoch_num, t_loss))

            if self.config.wandb:
                wandb.log({'train_loss': t_loss},step=epoch)

            if self.writer:
                self.writer.add_scalar('train/loss', t_loss, epoch)  # write to tensorboard to check
            self.validate()
            if epoch % ckpt == 0 and not self.debug:
                self.save_checkpoint(epoch)
            epoch_time.update(time.time() - end)
            end = time.time()
            remain_time = self.calc_remain_time(epoch, epoch_time)
            self.logger.info(f"remain {remain_time}")
        return

    def validate(self):
        epoch_val_loss = []
        model, device = self.model, self.device

        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(self.val_loader, 1):
                gt, lms, ms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[4].to(device)

                #TODO: Adjust input/output of your model here
                cond = pan.repeat(1,8,1,1) - lms
                out = model(lms, pan, cond) # for example
                ##
                loss = self.criterion(out, gt)
                epoch_val_loss.append(loss.item())
        v_loss = np.nanmean(np.array(epoch_val_loss))
        # writer.add_scalar('val/loss', v_loss, epoch)
        self.logger.info('validate loss: {:.7f}'.format(v_loss))
        if self.config.wandb:
            wandb.log({'validate_loss': v_loss})
        return

    def save_checkpoint(self, epoch):
        model_out_path = str(self.weights_save_path / f'CSNET{epoch}.pth')
        ckpt = {'state_dict': self.model.state_dict(), 'exp_timestamp': self.run_time}
        torch.save(ckpt, model_out_path)
        return

    def calc_remain_time(self, epoch, epoch_time):
        remain_time = (self.epoch_num - epoch) * epoch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        return remain_time


class Tester:
    def __init__(self, config):
        self.config = config
        dataset_name = config.dataset_name
        self.dataset_name = dataset_name
        self.model_name = config.model_name
        assert config.dataset_name in ('GF2', 'QB', 'WV3', 'WV2', 'CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8')
        assert config.test_mode in ('reduced', 'full')
        data_path = Path(config.data_path)
        if dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
            dname = dataset_name.lower()
            test_data_path = str(data_path / dname / f'test_{dname}_rgb.h5')
            self.dataset = h5py.File(test_data_path, 'r')
        else:
            if config.test_mode == 'reduced':
                tmp = f'test data/h5/{dataset_name}/reduce_examples/test_{dataset_name.lower()}_multiExm1.h5'
            else:
                tmp = f'test data/h5/{dataset_name}/full_examples/test_{dataset_name.lower()}_OrigScale_multiExm1.h5'
            test_data_path = str(data_path / tmp)

            self.dataset = h5py.File(test_data_path, 'r')
            # rgb channel indexes for each dataset
            if dataset_name in ('GF2', 'QB'):
                self.rgb_idx = [0, 1, 2]
            elif dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
                self.rgb_idx = [30, 19, 9]
            else:
                self.rgb_idx = [0, 2, 4]

        device = torch.device(config.device)
        self.device = device
        self.model = create_model(config, device)
        weight_path = config.test_weight_path
        ckpt = torch.load(weight_path, map_location=device)
        print(f"loading weight: {weight_path}")
        self.model.load_state_dict(ckpt['state_dict'])

        if dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
            save_path = Path(config.results_path) / f"{dataset_name}/{ckpt['exp_timestamp']}"
        else:
            save_path = Path(config.results_path) / f"{dataset_name}/{config.test_mode}/{ckpt['exp_timestamp']}"
        save_path.mkdir(exist_ok=True, parents=True)
        self.save_path = save_path
        return

    def test(self, analyse_fms=False):
        features = defaultdict(list)

        def get_features(name):
            def hook(model, input, output):
                if 'UE' in name:
                    features[name + '.fft_var'].append(output[0].detach().cpu().numpy())
                    features[name + '.fft_EU'].append(output[1].detach().cpu().numpy())
                    features[name + '.EU'].append(output[2].detach().cpu().numpy())
                    features[name + '.mean'].append(output[3].detach().cpu().numpy())
                else:
                    features[name].append(output.detach().cpu().numpy())

            return hook

        dataset, model = self.dataset, self.model
        dev = self.device
        scale = 1.0 if self.dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8'] else 2047.0

        if self.dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
            ms = np.array(dataset['LRHSI'], dtype=np.float32)
            lms = np.array(dataset['HSI_up'], dtype=np.float32)
            pan = np.array(dataset['RGB'], dtype=np.float32)
            gt = np.array(dataset['GT'], dtype=np.float32)
        else:
            ms = np.array(dataset['ms'], dtype=np.float32) / scale
            lms = np.array(dataset['lms'], dtype=np.float32) / scale
            pan = np.array(dataset['pan'], dtype=np.float32) / scale
            if self.config.test_mode == 'reduced':
                gt = np.array(dataset['gt'], dtype=np.float32)

        ms = torch.from_numpy(ms).float()
        if 'x8' in self.dataset_name and self.model_name != 'GuidedNet':
            ms = torch.nn.functional.interpolate(ms, scale_factor=2, mode='bicubic').squeeze()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()

        model.eval()
        print(f"save files to {self.save_path}")
        final_outs = []
        with torch.no_grad():
            # out = model(ms, pan)
            # out = model(lms, pan)
            # I_SR = torch.squeeze(out * 2047).cpu().detach().numpy()  # BxCxHxW
            # for i in range(len(I_SR)):
            #     sio.savemat(str(self.save_path / f'output_mulExm_{i}.mat'), {'I_SR': I_SR[i].transpose(1, 2, 0)})

            for i in range(len(pan)):
                # out = model(ms[i:i+1], lms[i:i+1], pan[i:i+1])
                # out = model(ms[i:i+1], lms[i:i+1], pan[i:i+1])
                #import pdb
                #pdb.set_trace()
                # cond, _ = einops.pack(
                #     [
                #         lms[i:i+1].to(self.device),
                #         pan[i:i+1].to(self.device),
                #     ],
                #     "b * h w",
                # )
                cond = pan[i:i+1].repeat(1,8,1,1).to(self.device) - lms[i:i+1].to(self.device)
                out = model(lms[i:i+1].to(self.device), pan[i:i+1].to(self.device), cond)
                #out = model(lms[i:i+1].to(self.device), pan=cond)
                #out = model(torch.from_numpy(gt[i:i+1]).to(self.device)-lms[i:i+1].to(self.device), pan=cond) # for InvNet_1208, reduced
                #out = out + lms[i:i+1].to(self.device)
                #out,_,_ = model(lms[i:i+1].to(self.device), pan[i:i+1].to(self.device)) # for HSFocal

                I_SR = torch.squeeze(out * scale).cpu().detach().numpy()  # BxCxHxW
                # I_GT = gt  # BxCxHxW
                # save H, W, C
                print(f'image {i}')
                if self.dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
                    final_outs.append(I_SR.transpose(1, 2, 0)[None, ...])
                else:
                    # save H, W, C
                    sio.savemat(str(self.save_path / f'output_mulExm_{i}.mat'), {'I_SR': I_SR.transpose(1, 2, 0)})
                    if analyse_fms and self.config.test_mode == 'reduced':
                        save_path = self.save_path / 'visualize_fms'
                        save_path.mkdir(exist_ok=True, parents=True)
                        plot_feature_maps(i, model.block_num, features, outs, gt[i], self.rgb_idx, save_path)


        if self.dataset_name in ['CAVE_x4', 'CAVE_x8', 'HARVARD_x4', 'HARVARD_x8']:
            final_outs = np.concatenate(final_outs, axis=0)
            if 'CAVE' in self.dataset_name:
                path = str(self.save_path / 'cave11-Ours.mat')
            else:
                path = str(self.save_path / 'harvard10-Ours.mat')
            sio.savemat(path, {'output': final_outs})
        #print('Total inference time: ', time.time() - end)
        return