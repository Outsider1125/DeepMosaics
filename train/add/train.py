import os
import random
import sys
import datetime
import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt

# 獲取當前文件的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
# 將專案根目錄添加到 sys.path
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)

# 確認專案根目錄是否正確
print(f"Project root: {project_root}")
print(f"sys.path: {sys.path}")

from cores import Options
from util import util, data
from util import image_processing as impro
from models import unet_model, BiSeNet_model
from torch import nn

# 設定選項
opt = Options()
opt.parser.add_argument('--dataset', type=str, default='C:/coding/DeepMosaics/datasets/draw/face/origin_image', help='your dataset dir')
opt.parser.add_argument('--savedir', type=str, default='C:\\coding\\DeepMosaics\\train\\add\\checkpoints', help='save dir')
opt.parser.add_argument('--maxload', type=int, default=1000, help='max number of images to load')
opt.parser.add_argument('--batchsize', type=int, default=16, help='batch size')
opt.parser.add_argument('--model', type=str, default='UNet', help='model type')
opt.parser.add_argument('--continue_train', type=bool, default=False, help='continue training from last checkpoint')
opt.parser.add_argument('--gpu_id', type=int, default=-1, help='GPU id to use')
opt = opt.getparse()

# 檢查並創建保存目錄
if not os.path.exists(opt.savedir):
    os.makedirs(opt.savedir)
print(f"checkpoints/face existed")

# 檢查數據集
dir_img = os.path.join(opt.dataset, 'images')
dir_mask = os.path.join(opt.dataset, 'C:/coding/DeepMosaics/datasets/draw/face/mask')
print('Dataset directory:', opt.dataset)
print('Image directory:', dir_img)
print('Mask directory:', dir_mask)

imagepaths = sorted(util.Traversal(dir_img))[:opt.maxload]
maskpaths = sorted(util.Traversal(dir_mask))[:opt.maxload]
print('Image paths:', imagepaths)
print('Mask paths:', maskpaths)

util.shuffledata(imagepaths, maskpaths)
if len(imagepaths) != len(maskpaths):
    print('dataset error!')
    exit(0)

img_num = len(imagepaths)
print('find images:', img_num)

if img_num == 0:
    print('No images found in the dataset directory.')
    exit(0)

# 創建日誌文件目錄
log_dir = os.path.join(opt.savedir, 'C:/coding/DeepMosaics/train/add/train/add/checkpoints/face')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 寫入日誌文件
util.writelog(os.path.join(log_dir, 'C:/coding/DeepMosaics/train/add/checkpoints/face/loss.txt'),
              str(time.asctime(time.localtime(time.time()))) + '\n' + util.opt2str(opt))

# 訓練和驗證數據集
imagepaths_eval = (imagepaths[int(img_num * 0.8):]).copy()
maskpaths_eval = (maskpaths[int(img_num * 0.8):]).copy()

'''

--------------------------Get options--------------------------
'''
opt.parser.add_argument('--lr',type=float,default=0.001, help='')
opt.parser.add_argument('--finesize',type=int,default=360, help='')
opt.parser.add_argument('--loadsize',type=int,default=400, help='')
opt.parser.add_argument('--batchsize',type=int,default=8, help='')
opt.parser.add_argument('--model',type=str,default='BiSeNet', help='BiSeNet or UNet')

opt.parser.add_argument('--maxepoch',type=int,default=100, help='')
opt.parser.add_argument('--savefreq',type=int,default=5, help='')
opt.parser.add_argument('--maxload',type=int,default=1000000, help='')
opt.parser.add_argument('--continue_train', action='store_true', help='')
opt.parser.add_argument('--startepoch',type=int,default=0, help='')
opt.parser.add_argument('--dataset',type=str,default='C:/coding/DeepMosaics/datasets/draw/face/origin_image', help='')
opt.parser.add_argument('--savename',type=str,default='face', help='')


'''
--------------------------Init--------------------------
'''
opt = opt.getparse()
dir_img = os.path.join(opt.dataset, 'C:/coding/DeepMosaics/datasets/draw/face/origin_image')
dir_mask = os.path.join(opt.dataset, 'C:/coding/DeepMosaics/datasets/draw/face/mask')
dir_checkpoint = os.path.join('C:/coding/DeepMosaics/train/add/checkpoints',opt.savename)
util.makedirs(dir_checkpoint)
util.writelog(os.path.join(dir_checkpoint,'C:/coding/DeepMosaics/train/add/checkpoints/face/loss.txt'), 
              str(time.asctime(time.localtime(time.time())))+'\n'+util.opt2str(opt))

def Totensor(img,gpu_id=True):
    size=img.shape[0]
    img = torch.from_numpy(img).float()
    if opt.gpu_id != -1:
        img = img.cuda()
    return img

def loadimage(imagepaths,maskpaths,opt,test_flag = False):
    batchsize = len(imagepaths)
    images = np.zeros((batchsize,3,opt.finesize,opt.finesize), dtype=np.float32)
    masks = np.zeros((batchsize,1,opt.finesize,opt.finesize), dtype=np.float32)
    for i in range(len(imagepaths)):
        img = impro.resize(impro.imread(imagepaths[i]),opt.loadsize)
        mask = impro.resize(impro.imread(maskpaths[i],mod = 'gray'),opt.loadsize)      
        img,mask = data.random_transform_pair_image(img, mask, opt.finesize, test_flag)
        images[i] = (img.transpose((2, 0, 1))/255.0)
        masks[i] = (mask.reshape(1,1,opt.finesize,opt.finesize)/255.0)
    images = data.to_tensor(images,opt.gpu_id)
    masks = data.to_tensor(masks,opt.gpu_id)

    return images,masks


'''
--------------------------checking dataset--------------------------
'''
print('checking dataset...')
imagepaths = sorted(util.Traversal(dir_img))[:opt.maxload]
maskpaths = sorted(util.Traversal(dir_mask))[:opt.maxload]
data.shuffledata(imagepaths, maskpaths)
if len(imagepaths) != len(maskpaths) :
    print('dataset error!')
    exit(0)
img_num = len(imagepaths)
print('find images:',img_num)
imagepaths_train = (imagepaths[0:int(img_num*0.8)]).copy()
maskpaths_train = (maskpaths[0:int(img_num*0.8)]).copy()
imagepaths_eval = (imagepaths[int(img_num*0.8):]).copy()
maskpaths_eval = (maskpaths[int(img_num*0.8):]).copy()

'''
--------------------------def network--------------------------
'''
if opt.model =='UNet':
    net = unet_model.UNet(n_channels = 3, n_classes = 1)
elif opt.model =='BiSeNet':
    net = BiSeNet_model.BiSeNet(num_classes=1, context_path='resnet18')

if opt.continue_train:
    if not os.path.isfile(os.path.join(dir_checkpoint,'last.pth')):
        opt.continue_train = False
        print('can not load last.pth, training on init weight.')
if opt.continue_train:
    net.load_state_dict(torch.load(os.path.join(dir_checkpoint,'last.pth')))
    f = open(os.path.join(dir_checkpoint,'epoch_log.txt'),'r')
    opt.startepoch = int(f.read())
    f.close()
if opt.gpu_id != -1:
    net.cuda()
    cudnn.benchmark = True

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

if opt.model =='UNet':
    criterion = nn.BCELoss()
elif opt.model =='BiSeNet':
    criterion = nn.BCELoss()
    # criterion = BiSeNet_model.DiceLoss()

'''
--------------------------train--------------------------
'''
loss_plot = {'train':[],'eval':[]}
print('begin training......')
for epoch in range(opt.startepoch,opt.maxepoch):
    random_save = random.randint(0, int(img_num*0.8/opt.batchsize))
    data.shuffledata(imagepaths_train, maskpaths_train)

    starttime = datetime.datetime.now()
    util.writelog(os.path.join(dir_checkpoint,'loss.txt'),'Epoch {}/{}.'.format(epoch + 1, opt.maxepoch),True)
    net.train()
    if opt.gpu_id != -1:
        net.cuda()
    epoch_loss = 0
    for i in range(int(img_num*0.8/opt.batchsize)):
        img,mask = loadimage(imagepaths_train[i*opt.batchsize:(i+1)*opt.batchsize], maskpaths_train[i*opt.batchsize:(i+1)*opt.batchsize], opt)

        if opt.model =='UNet':
            mask_pred = net(img)
            loss = criterion(mask_pred, mask)
            epoch_loss += loss.item()
        elif opt.model =='BiSeNet':
            mask_pred, mask_pred_sup1, mask_pred_sup2 = net(img)
            loss1 = criterion(mask_pred, mask)
            loss2 = criterion(mask_pred_sup1, mask)
            loss3 = criterion(mask_pred_sup2, mask)
            loss = loss1 + loss2 + loss3
            epoch_loss += loss1.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100 == 0:
            data.showresult(img,mask,mask_pred,os.path.join(dir_checkpoint,'result.png'),True)
        if  i == random_save:
            data.showresult(img,mask,mask_pred,os.path.join(dir_checkpoint,'epoch_'+str(epoch+1)+'.png'),True)
    epoch_loss = epoch_loss/int(img_num*0.8/opt.batchsize)
    loss_plot['train'].append(epoch_loss)

    #val
    epoch_loss_eval = 0
    with torch.no_grad():
    # net.eval()
        for i in range(int(img_num*0.2/opt.batchsize)):
            img,mask = loadimage(imagepaths_eval[i*opt.batchsize:(i+1)*opt.batchsize], maskpaths_eval[i*opt.batchsize:(i+1)*opt.batchsize], opt,test_flag=True)
            if opt.model =='UNet':
                mask_pred = net(img)
            elif opt.model =='BiSeNet':
                mask_pred, _, _ = net(img)
            # mask_pred = net(img)
            loss= criterion(mask_pred, mask)
            epoch_loss_eval += loss.item()
    epoch_loss_eval = epoch_loss_eval/int(img_num*0.2/opt.batchsize)
    loss_plot['eval'].append(epoch_loss_eval)
    # torch.cuda.empty_cache()

    #savelog
    endtime = datetime.datetime.now()
    util.writelog(os.path.join(dir_checkpoint,'loss.txt'),
                '--- Epoch train_loss: {0:.6f} eval_loss: {1:.6f} Cost time: {2:} s'.format(
                    epoch_loss,
                    epoch_loss_eval,
                    (endtime - starttime).seconds),
                True)
    #plot
    plt.plot(np.linspace(opt.startepoch+1,epoch+1,epoch+1-opt.startepoch),loss_plot['train'],label='train')
    plt.plot(np.linspace(opt.startepoch+1,epoch+1,epoch+1-opt.startepoch),loss_plot['eval'],label='eval')
    plt.xlabel('Epoch')
    plt.ylabel('BCELoss')
    plt.legend(loc=1)
    plt.savefig(os.path.join(dir_checkpoint,'loss.jpg'))
    plt.close()
    #save network
    torch.save(net.cpu().state_dict(),os.path.join(dir_checkpoint,'last.pth'))
    f = open(os.path.join(dir_checkpoint,'epoch_log.txt'),'w+')
    f.write(str(epoch+1))
    f.close()
    if (epoch+1)%opt.savefreq == 0:
        torch.save(net.cpu().state_dict(),os.path.join(dir_checkpoint,'epoch'+str(epoch+1)+'.pth'))
        print('network saved.')
