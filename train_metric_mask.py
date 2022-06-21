import numpy as np
import os
import random
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
from loss import MetricLoss, L2Loss
# from torch.utils.data import Dataloader
from dataset import PairedDeepFakeDataset
from xception import Head_Model, Tail_Model
os.environ['CUDA_VISIBLE_DEVICES'] =  '0,1,2,3, 4, 5, 6, 7'#'4,5,6,7' #'0,1,2,3'
# from templates import get_templates

MODEL_DIR = './models/'
BACKBONE = 'xcp'
MAPTYPE = 'reg' # "none"# 'tmp'
BATCH_SIZE = 64
MAX_EPOCHS = 15

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

DATASET = "NeuralTextures"
COMPRESSION = "c40"
iter = 1

if DATASET == "All":
    sub_dbs = ["Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures"]
    dbs = []
    for sub_db in sub_dbs:
        db = PairedDeepFakeDataset( 
            phase="train",
            real_dataset_path = "/media/user/deepfake/ff/face/real/c23",
            fake_dataset_path = "/media/user/deepfake/ff/face/"+sub_db+"/c23")
        dbs.append(db)
    DATA_TRAIN = torch.utils.data.ConcatDataset(dbs)
else:
    DATA_TRAIN = PairedDeepFakeDataset( 
        phase="train",
        real_dataset_path = "/media/user/deepfake/ff/face/real/c23",
        fake_dataset_path = "/media/user/deepfake/ff/face/"+DATASET+"/c23") 

DATA_NUM = len(DATA_TRAIN)
STEPS_PER_EPOCH = int(DATA_NUM / BATCH_SIZE)
# DATA_EVAL = Dataset('eval', BATCH_SIZE, CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'], SEED)
train_loader =torch.utils.data.DataLoader(DATA_TRAIN, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

MODEL_NAME = '{0}_{1}_{2}_{3}'.format(BACKBONE, MAPTYPE, DATASET, COMPRESSION)
MODEL_DIR = MODEL_DIR + MODEL_NAME + '/'

# MODEL = Model(MAPTYPE, TEMPLATES, 2, False)
TAIL_MODEL = Tail_Model(isFC=False)
HQ_HEAD_MODEL = Head_Model(maptype=MAPTYPE)
LQ_HEAD_MODEL = Head_Model(maptype=MAPTYPE)

HQ_HEAD_MODEL.model = torch.nn.DataParallel(HQ_HEAD_MODEL.model.cuda())
LQ_HEAD_MODEL.model = torch.nn.DataParallel(LQ_HEAD_MODEL.model.cuda())
TAIL_MODEL.model = torch.nn.DataParallel(TAIL_MODEL.model.cuda())


# MODEL = Model(MAPTYPE, None, 2, load_pretrain=True)
# MODEL.model = torch.nn.DataParallel(MODEL.model.cuda())

OPTIM = optim.Adam([
  {"params": HQ_HEAD_MODEL.model.parameters()},
  {"params": LQ_HEAD_MODEL.model.parameters()},
  {"params": TAIL_MODEL.model.parameters()}],
  # MODEL.model.parameters(),
  lr=LEARNING_RATE, 
  weight_decay=WEIGHT_DECAY)

# LOSS_CSE = nn.CrossEntropyLoss().cuda()
LOSS_M = MetricLoss(torch.zeros([2048]), 0.01, 18.0).cuda()
LOSS_L1 = nn.L1Loss().cuda()
LOSS_BCE = nn.BCELoss().cuda()
LOSS_L2_NORM = L2Loss(norm=True).cuda()
LOSS_L2 = L2Loss(norm=False).cuda()

# MAXPOOL = nn.MaxPool2d(19).cuda()

def calculate_losses(batch, epoch):
  # img = batch['img']
  # msk = batch['msk']
  # lab = batch['lab']
  hq_img = batch[0].cuda()
  lq_img = batch[1].cuda()
  msk = batch[2].cuda()
  lab = batch[3].long().cuda()
  
  hq_x, hq_mask, hq_f = HQ_HEAD_MODEL.model(hq_img)
  lq_x, lq_mask, lq_f = LQ_HEAD_MODEL.model(lq_img)
  
  at_loss = LOSS_L2_NORM( 
    hq_mask.reshape(hq_mask.size(0), -1), 
    lq_mask.reshape(hq_mask.size(0), -1))
  # hq_l1 = LOSS_L1(hq_mask, msk)
  # lq_l1 = LOSS_L1(lq_mask, msk)
  hq_l1 = LOSS_BCE(hq_mask, msk)
  lq_l1 = LOSS_BCE(lq_mask, msk)

  hq_x = TAIL_MODEL.model(hq_x)
  lq_x = TAIL_MODEL.model(lq_x)

  hq_loss_metric = LOSS_M(hq_x, lab)
  lq_loss_metric = LOSS_M(lq_x, lab)

  # if epoch <= 1:
  #   loss_l2 = LOSS_L2_NORM(hq_x, lq_x)
  #   loss = 0.1*(hq_loss_metric + lq_loss_metric) + loss_l2 + (at_loss + hq_l1 + lq_l1)#
  # else:
  #   loss_l2 = LOSS_L2(hq_x, lq_x)
  #   loss = 0.1*(hq_loss_metric + lq_loss_metric + loss_l2) + (at_loss + hq_l1 )
  # loss = loss_l1 + loss_cse
  # loss = hq_loss_cse + lq_loss_cse \
  #   + 0.1 * (hq_loss_metric + lq_loss_metric + loss_l2) \
  #    + hq_loss_l1 + at_loss
  loss_l2 = 0.2*LOSS_L2(hq_x, lq_x)
  # global iter
  # iter += 1
  # # print(iter)
  # loss_l2 = LOSS_L2(hq_x, lq_x)
  # if iter < 1000:
  # loss_l2 = 0.1 * loss_l2
  # else:
  #   loss_l2 = 0.5 * loss_l2
  # # elif loss_l2 < 35000:
  #   loss_l2 =  loss_l2
  # else:
  #   loss_l2 = 10 * loss_l2
  loss = hq_loss_metric + lq_loss_metric + loss_l2 + (at_loss + hq_l1 + lq_l1)


  hq_d = torch.norm(hq_x, dim=1)
  hq_pred = hq_d > 9.0
  hq_acc = (hq_pred == lab).float().mean()

  lq_d = torch.norm(lq_x, dim=1)
  lq_pred = lq_d > 9.0
  lq_acc = (lq_pred == lab).float().mean()

  return{ 
    "loss":loss, 
    "hq_loss_metric": hq_loss_metric,
    "lq_loss_metric": lq_loss_metric,
    "loss_l2": loss_l2,
    "hq_loss": hq_l1,
    "lq_loss": lq_l1,
    "at_loss": at_loss,
    "hq_acc": hq_acc,
    "lq_acc": lq_acc
  }

  # hq_pred = torch.max(hq_x, dim=1)[1]
  # hq_acc = (hq_pred == lab).float().mean()
  # lq_pred = torch.max(lq_x, dim=1)[1]
  # lq_acc = (lq_pred == lab).float().mean()
  # return { 'loss': loss, 'loss_l1': loss_l1, 'loss_cse': loss_cse, 'acc': acc }
  # return { 'loss': loss, \
  #   'hq_loss_cse': hq_loss_cse, 'lq_loss_cse': lq_loss_cse, \
  #   'hq_loss_l1': hq_loss_l1, 'at_loss': at_loss, \
  #   'hq_loss_metric': hq_loss_metric, 'lq_loss_metric': lq_loss_metric, "loss_l2": loss_l2,\
  #   'hq_acc': hq_acc, 'lq_acc': lq_acc }

def process_batch(batch, mode, epoch):
  if mode == 'train':
    HQ_HEAD_MODEL.model.train()
    LQ_HEAD_MODEL.model.train()
    TAIL_MODEL.model.train()
    losses = calculate_losses(batch, epoch=epoch)
    OPTIM.zero_grad()
    losses['loss'].backward()
    OPTIM.step()
  elif mode == 'eval':
    LQ_HEAD_MODEL.model.eval()
    TAIL_MODEL.model.eval()
    with torch.no_grad():
      losses = calculate_losses(batch)
  return losses

SUMMARY_WRITER = SummaryWriter(MODEL_DIR + 'logs/')
def write_tfboard(item, itr, name):
  SUMMARY_WRITER.add_scalar('{0}'.format(name), item, itr)

def run_step(e, s):
  batch = DATA_TRAIN.get_batch()
  losses = process_batch(batch, 'train', epoch=e)

  if s % 10 == 0:
    print('\r{0} - '.format(s) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]), end='')
  if s % 100 == 0:
    print('\n', end='')
    [write_tfboard(losses[_], e * STEPS_PER_EPOCH + s, _) for _ in losses]

def run_epoch(e):
  print('Epoch: {0}'.format(e))
  for s, batch in enumerate(train_loader):
    # batch = DATA_TRAIN.get_batch()
    losses = process_batch(batch, 'train', e)

    if s % 10 == 0:
      print('\r{0} - '.format(s) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]), end='')
    if s % 100 == 0:
      print('\n', end='')
      [write_tfboard(losses[_], e * STEPS_PER_EPOCH + s, _) for _ in losses]
  TAIL_MODEL.save(e+1, OPTIM, MODEL_DIR, 'tail')
  HQ_HEAD_MODEL.save(e+1, OPTIM, MODEL_DIR, 'hq_head')
  LQ_HEAD_MODEL.save(e+1, OPTIM, MODEL_DIR, 'lq_head')

LAST_EPOCH = 0
for e in range(LAST_EPOCH, MAX_EPOCHS):
  run_epoch(e)

