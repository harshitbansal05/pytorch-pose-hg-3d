import os
import time
import datetime

import torch
import torch.utils.data
from opts import opts
import ref
from models.hg_3d import HourglassNet3D
from utils.utils import adjust_learning_rate
from datasets.fusion import Fusion
from datasets.h36m import H36M
from datasets.mpii import MPII
from utils.logger import Logger
from train import train, val

def main():
  opt = opts().parse()
  now = datetime.datetime.now()
  logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))

  if opt.loadModel != 'none':
    model = torch.load(opt.loadModel).cuda()
  else:
    model = HourglassNet3D(opt.nStack, opt.nModules, opt.nFeats, opt.nRegModules).cuda()
  
  criterion = torch.nn.MSELoss().cuda()
  optimizer = torch.optim.RMSprop(model.parameters(), opt.LR, 
                                  alpha = ref.alpha, 
                                  eps = ref.epsilon, 
                                  weight_decay = ref.weightDecay, 
                                  momentum = ref.momentum)

  val_loader = torch.utils.data.DataLoader(
      H36M(opt, 'val'), 
	  batch_size = 1, 
	  shuffle = False,
	  num_workers = int(ref.nThreads)
  )
  
  loss_val, acc_val, mpjpe_val, loss3d_val = val(0, opt, val_loader, model, criterion)
  logger.scalar_summary('loss_val', loss_val, 1)
  logger.scalar_summary('acc_val', acc_val, 1)
  logger.scalar_summary('mpjpe_val', mpjpe_val, 1)
  logger.scalar_summary('loss3d_val', loss3d_val, 1)
  
  logger.write('{:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} \n'.format(loss_val, acc_val, mpjpe_val, loss3d_val))
  print('Loss Val: ', loss_val)
  print('Accuracy Val: ', acc_val)
  print('MPJPE Val: ', mpjpe_val)
  print('Loss 3D Val: ', loss3d_val)

  logger.close()

if __name__ == '__main__':
  main()
