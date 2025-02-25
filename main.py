import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
from loss_util import SSIM
from contrastive_dw_5mi import net

import scipy.io as sio
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter

import os
import pdb

import wandb
import datetime
import utils.util as u
from utils.global_config import parse_args, init_global_config
from models1.pipeline import Trainer

# ================== Pre-test =================== #
def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy((data['ms'] / 2047.0)).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy((data['pan'] / 2047.0))   # HxW = 256x256

    return lms, ms, pan

def load_gt_compared(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    test_gt = torch.from_numpy(data['gt'] / 2047.0)  # CxHxW = 8x256x256

    return test_gt

# ================== Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True

# ============= HYPER PARAMS(Pre-Defined) ==========#
# lr = 0.0003  # 0.001
# epochs = 500
ckpt = 20
# batch_size = 32
# lr = 1e-3/1024/64
# lr = 1e-3
lr = 6e-4
epochs = 1000
batch_size = 32

satellite = 'qb' # wv3, gf2, qb

if torch.cuda.device_count() > 1:
    print(f'共有{torch.cuda.device_count()}张GPU--使用第1张GPU训练')
device = torch.device('cuda:0') # 用第一张显卡
model = net(8, 64).to(device)


netname = 'contrastive'
criterion = nn.L1Loss().to(device)
# optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))   # optimizer 1
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

All_loss = {'TrainLoss': [], 'TrainL1': [], 'TrainSsim': [], 'TrainMi': [], 'TrainCl': [],
            'ValLoss': [], 'ValL1': [], 'ValSsim': [], 'ValMi': [], 'ValCl': []}

if os.path.exists('./train_logs'):  # for tensorboard: copy dir of train_logs
    shutil.rmtree('./train_logs')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs

writer = SummaryWriter('./train_logs/mutual_logs/log')


# model_out_path = './weights/mutual/wv3/bcd_2level'
model_out_path = './weights/qb/hfean'

if not os.path.exists(model_out_path):
    os.makedirs(model_out_path)
    

def save_checkpoint(model, epoch, model_out_path):  # save model function
    # model_out_path = './weights/mutual/wv3/contrastive_dw_ssim0.1'
    # model_out_path = './weights/mutual/wv3/contrastive_dw_withoutcl'
    torch.save(model.state_dict(), model_out_path + f'/{epoch}.pth')


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################

def train(training_data_loader, validate_data_loader):
    data = datetime.datetime.now()
    datastr = str(data.month) + '-' + str(data.day) + '-' + str(data.hour) + ':' + str(data.minute)

    print('Start training...')

    for epoch in range(epochs):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []
        trainl1_loss, trainmi_loss, trainssim_loss, traincl_loss = [], [], [], []
        vall1_loss, valmi_loss, valssim_loss, valcl_loss = [], [], [], []
        # mi_loss = []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            gt, lms, ms, _, pan = Variable(batch[0], requires_grad=False).to(device), \
                                     Variable(batch[1]).to(device), \
                                     Variable(batch[2]).to(device), \
                                     Variable(batch[3]).to(device), \
                                     Variable(batch[4]).to(device)
     	    #print(size(gt), lms.shape(), ms.shape(), pan_hr.shape(), pan.shape())
            optimizer.zero_grad()  # fixed
            #out = model(pan, lms)
            # out, clloss, mi = model(lms, pan, gt)
            out = model(lms, pan)

            # ssim_loss = pytorch_ssim.SSIM()
            # traincl_loss.append(clloss.item())
            # trainmi_loss.append(mi.item())
            # All_loss['TrainCl'].append(np.nanmean(np.array(clloss.item())))
            # All_loss['TrainMi'].append(np.nanmean(np.array(mi.item())))

            loss = criterion(out, gt)  # compute loss
            # All_loss['TrainL1'].append(np.nanmean(np.array(loss.item())))
            trainl1_loss.append(loss.item())

            ssim = SSIM()
            ssim = 1 - ssim(out, gt)
            # trainssim_loss.append(ssim.item())
            # All_loss['TrainSsim'].append(np.nanmean(np.array(ssim.item())))
            # loss_2 = frecriterion(out, gt)
            # loss = loss + 0.1*loss_2 + 0.2*(miloss3+miloss2+miloss1)
            # loss = loss + 0.1*clloss + mi + 0.2*ssim
            # loss = loss + 0.1*clloss + mi + 0.1*ssim
            loss_pcp = pcpcriterion(out[:, [0, 1, 2],...], gt[:, [0, 1, 2],...])
            loss = loss + 0.1*loss_pcp
            # loss = loss + mi + 0.1*ssim

            # All_loss['TrainLoss'].append(np.nanmean(np.array(loss.item())))

            # loss = loss
            # mi_loss.append(( miloss5+miloss4+miloss3+miloss2+miloss1).item())
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()  # fixed
            optimizer.step()  # fixed
        # scheduler.step()

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('train/loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss:{:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch
        # print('cl:{:.7f}'.format(np.nanmean(np.array(traincl_loss))))
        # print('mi:{:.7f}'.format(np.nanmean(np.array(trainmi_loss))))
        # print('ssim:{:.7f}'.format(np.nanmean(np.array(trainssim_loss))))

        All_loss['TrainLoss'].append(np.nanmean(np.array(epoch_train_loss)))
        All_loss['TrainL1'].append(np.nanmean(np.array(trainl1_loss)))
        # All_loss['TrainSsim'].append(np.nanmean(np.array(trainssim_loss)))
        # All_loss['TrainCl'].append(np.nanmean(np.array(traincl_loss)))
        # All_loss['TrainMi'].append(np.nanmean(np.array(trainmi_loss)))

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch, model_out_path)
        # if epoch % 10 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         output1, output2, output3 = model(test_ms, test_pan,test_lms)
        #         result_our = torch.squeeze(output3).permute(1, 2, 0)
        #         #sr = torch.squeeze(output3).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
        #         result_our = result_our * 2047
        #         result_our = result_our.type(torch.DoubleTensor).to(device)
        #
        #         our_SAM, our_ERGAS = compute_index(test_gt, result_our, 4)
        #         print('our_SAM: {} dmdnet_SAM: 2.9355'.format(our_SAM) ) # print loss for each epoch
        #         print('our_ERGAS: {} dmdnet_ERGAS:1.8119 '.format(our_ERGAS))
        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, _, _ ,pan= Variable(batch[0], requires_grad=False).to(device), \
                                         Variable(batch[1]).to(device), \
                                         batch[2], \
                                         batch[3], \
                                         Variable(batch[4]).to(device)

                # out = model(pan, lms)
                out = model(lms, pan)
                # out, clloss, mi = model(lms, pan, gt)
                # valcl_loss.append(clloss.item())
                # valmi_loss.append(mi.item())
                # All_loss['ValCl'].append(np.nanmean(np.array(clloss.item())))
                # All_loss['ValMi'].append(np.nanmean(np.array(mi.item())))

                loss = criterion(out, gt)
                vall1_loss.append(loss.item())
                # All_loss['ValL1'].append(np.nanmean(np.array(loss.item())))

                loss_2 = frecriterion(out, gt)
                ssim = SSIM()
                ssim = 1 - ssim(out, gt)
                # valssim_loss.append(ssim.item())
                # All_loss['ValSsim'].append(np.nanmean(np.array(ssim.item())))

                # loss = loss +  0.1*(miloss3+miloss2+miloss1+miloss4+miloss5
                # loss = loss + 0.1*clloss + mi + 0.2*ssim
                # loss = loss + 0.1*clloss + mi + 0.1*ssim
                loss_pcp = pcpcriterion(out[:, [0, 1, 2],...], gt[:, [0, 1, 2],...])
                loss = loss+0.1*loss_pcp
                # loss = loss  + mi + 0.1*ssim
                # All_loss['ValLoss'].append(np.nanmean(np.array(loss.item())))
                # mi_loss.append((miloss5+miloss4+miloss3+miloss2+miloss1).item())
                epoch_val_loss.append(loss.item())
        v_loss = np.nanmean(np.array(epoch_val_loss))
        # writer.add_scalar('val/loss', v_loss, epoch)
        print('validate loss: {:.7f}'.format(v_loss))
        # print('cl:{:.7f}'.format(np.nanmean(np.array(valcl_loss))))
        # print('mi:{:.7f}'.format(np.nanmean(np.array(valmi_loss))))
        # print('ssim:{:.7f}'.format(np.nanmean(np.array(valssim_loss))))

        All_loss['ValLoss'].append(np.nanmean(np.array(epoch_val_loss)))
        # All_loss['ValL1'].append(np.nanmean(np.array(vall1_loss)))
        # All_loss['ValSsim'].append(np.nanmean(np.array(valssim_loss)))
        # All_loss['ValCl'].append(np.nanmean(np.array(valcl_loss)))
        # All_loss['ValMi'].append(np.nanmean(np.array(valmi_loss)))

        sio.savemat('./train_logs/' + netname + '_loss_' + datastr + '.mat', All_loss)

    # writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":


    train_set = Dataset_Pro(f'/home/LJL/Proj/pansharpening_data/training_{satellite}/train_{satellite}.h5')
    # train_set = Dataset_Pro('/home/LJL/Proj/pansharpening_data/training_gf2/train_gf2.h5')
    # train_set = Dataset_Pro('./data/wv3/train_wv3.h5')  # create data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro(f'/home/LJL/Proj/pansharpening_data/training_{satellite}/valid_{satellite}.h5')  # create data for validation
    # validate_set = Dataset_Pro('./data/wv3/valid_wv3.h5')  # create data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches
    # ------------------- load_test ----------------------------------#
    # file_path = "test_data/new_data6.mat"
    # test_lms, test_ms, test_pan = load_set(file_path)
    # test_lms = test_lms.to(device).unsqueeze(dim=0).float()
    # test_ms = test_ms.to(device).unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
    # test_pan = test_pan.to(device).unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW
    # test_gt= load_gt_compared(file_path)  ##compared_result
    # test_gt = (test_gt * 2047).to(device).double()
    ###################################################################
    #train(training_data_loader, validate_data_loader)  # call train function (call: Line 53)
    train(training_data_loader, validate_data_loader)  # call train function (call: Line 66)


    ### Parsing Args
    # args = parse_args()
    # config = init_global_config(args)
    #
    # pdb.set_trace()
    # ### Wandb Settings (if args.wandb == True)
    # # TODO: Ad-justify your WANDB account here by replacing 'entity'
    # if args.wandb:
    #     wandb.init(project="Pan_proj", entity="HCT",
    #                name=str(data.month) + '-' + str(data.day) + '-' + str(
    #                    data.hour) + ':' + str(data.minute))
    #     wandb.config.update(args)
    #
    # ### CUDA Devices Settings
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in config.gpu_list)
    #
    # ### Logger Setting
    # logger = u.get_logger(config)
    # u.setup_seed(2023)
    # u.log_args_and_parameters(logger, args, config)
    #
    # ### Initialize Trainer
    # trainer = Trainer(config, logger)
    #
    # ### Start Training
    # trainer.train_all()
    #
    # if args.wandb:
    #     wandb.finish()
