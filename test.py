import numpy as np
import os
import random
from scipy.io import savemat
import shutil
import torch
import torch.optim as optim
import torch.nn as nn

from dataset import PairedDeepFakeDataset, AllPairedDeepFakeDataset
from loss import MetricLoss, L2Loss

MODEL_DIR = '/media/user/deepfake/detect-fake-image/checkpoints/'
BACKBONE = 'xcp'
MAPTYPE = 'reg'#'none' #'tmp'
BATCH_SIZE = 128

from xception import Head_Model, Tail_Model
# from xception import Head_Model, Tail_Model
os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4, 5, 6'

# DATASET = "Face2Face"
# DATASET = "Deepfakes"
# DATASET = "FaceSwap"
# DATASET = "NeuralTextures"
DATASET = "All"

COMPRESSION = "c40"

if DATASET == "Deepfakes":
    BEST_EPOCH = 11
elif DATASET == "FaceSwap":
    BEST_EPOCH = 8
elif DATASET == "Face2Face":
    BEST_EPOCH = 9
elif DATASET == "NeuralTextures":
    BEST_EPOCH = 10
elif DATASET == "All":
    BEST_EPOCH = 13

# MODEL_NAME = '{0}_{1}_{2}_{3}'.format(BACKBONE, MAPTYPE, DATASET, COMPRESSION)
MODEL_DIR = MODEL_DIR + DATASET + '/'

TAIL_MODEL = Tail_Model(isFC=False)
# HQ_HEAD_MODEL = Head_Model(maptype=MAPTYPE)
LQ_HEAD_MODEL = Head_Model(maptype=MAPTYPE)

# HQ_HEAD_MODEL.model = torch.nn.DataParallel(HQ_HEAD_MODEL.model.cuda())
LQ_HEAD_MODEL.model = torch.nn.DataParallel(LQ_HEAD_MODEL.model.cuda())
TAIL_MODEL.model = torch.nn.DataParallel(TAIL_MODEL.model.cuda())

LOSS_M = MetricLoss(torch.zeros([2048]), 0.1, 18.0).cuda()
LOSS_L1 = nn.L1Loss().cuda()
# LOSS_CSE = nn.CrossEntropyLoss().cuda()
# LOSS_L1 = nn.L1Loss().cuda()
# MAXPOOL = nn.MaxPool2d(19).cuda()

def calculate_losses(batch):
    hq_img = batch[0].cuda()
    lq_img = batch[1].cuda()
    msk = batch[2].cuda()
    lab = batch[3].long().cuda()
    
    # hq_x, hq_mask, hq_f = HQ_HEAD_MODEL.model(hq_img)
    lq_x, lq_mask, lq_f = LQ_HEAD_MODEL.model(lq_img)
    # loss_cse = LOSS_CSE(x, lab)

    # hq_x = TAIL_MODEL.model(hq_x)
    lq_x = TAIL_MODEL.model(lq_x)

    # hq_loss_metric = LOSS_M(hq_x, lab)
    lq_loss_metric = LOSS_M(lq_x, lab)
    loss = lq_loss_metric
    
    d = torch.norm(lq_x, dim=1)
    pred = d > 5.0
    acc = (pred == lab).float().mean()

    # d = torch.norm(x, dim=1)
    # pred = d > 30.0
    # acc = (pred == lab).float().mean()
    # res = { 'lab': lab, 'msk': msk, 'score': x, 'pred': pred, 'mask': mask }
    res = { 'lab': lab, 'score': d, 'pred': pred, 'msk': msk, 'mask': lq_mask}

    results = {}
    for r in res:
        results[r] = res[r].squeeze().cpu().numpy()
    return { 'loss': loss, 'acc': acc }, results

def process_batch(batch, mode):
    TAIL_MODEL.model.eval()
    LQ_HEAD_MODEL.model.eval()

    with torch.no_grad():
        losses, results = calculate_losses(batch)
    return losses, results


def run_epoch(di, e, resultdir, test_loader):
    for s, batch in enumerate(test_loader):
        losses, results = process_batch(batch, 'test')

        savemat('{0}{1}_{2}.mat'.format(resultdir, di, s), results)

        if s % 10 == 0:
            print('\r{0} - '.format(s) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]), end='')


resultdir = '{0}results/{1}/'.format(MODEL_DIR, BEST_EPOCH)
if os.path.exists(resultdir):
    shutil.rmtree(resultdir)
os.makedirs(resultdir, exist_ok=True)
TAIL_MODEL.load(BEST_EPOCH, MODEL_DIR, "tail")
LQ_HEAD_MODEL.load(BEST_EPOCH, MODEL_DIR, "head")

if DATASET == "All":
    DATA_TEST = AllPairedDeepFakeDataset( 
        phase="test", 
        real_dataset_path= "/media/user/deepfake/ff/face/real/c23",
        fake_dataset_paths = [
                    "/media/user/deepfake/ff/face/Deepfakes/c23", 
                    "/media/user/deepfake/ff/face/FaceSwap/c23", 
                    "/media/user/deepfake/ff/face/Face2Face/c23", 
                    "/media/user/deepfake/ff/face/NeuralTextures/c23"])
else:
    DATA_TEST = PairedDeepFakeDataset( 
        phase="test",
        real_dataset_path = "/media/user/deepfake/ff/face/real/c23",
        fake_dataset_path = "/media/user/deepfake/ff/face/"+DATASET+"/c23")  
        # real_dataset_path = "/media/user/deepfake/data/Retinaface/real/c23",
        # fake_dataset_path = "/media/user/deepfake/data/Retinaface/"+DATASET+"/c23")

test_loader = torch.utils.data.DataLoader( 
    DATA_TEST,
    batch_size=128,
    shuffle=False,
    num_workers=8,
    drop_last=False
)

di = "c40"
run_epoch(di, BEST_EPOCH, resultdir, test_loader)
print()

print('Testing complete')
