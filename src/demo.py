import sys
import torch
from opts import opts
import ref
from utils.debugger import Debugger
from utils.eval import getPreds
import cv2
import numpy as np

from functools import partial
import pickle

pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

def main():
  opt = opts().parse()
  # if opt.loadModel != 'none':
  #   model = torch.load(opt.loadModel)
  # else:
  #   model = torch.load('hgreg-3d.pth', pickle_module=pickle)
  img = cv2.imread(opt.demo)
  i = int(opt.demo.split('.')[0].split('/')[1])
  input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
  input = input.view(1, input.size(0), input.size(1), input.size(2))
  # input_var = torch.autograd.Variable(input).float()
  # output = model(input_var)
  pred = np.loadtxt('test1.txt')
  reg = np.loadtxt('reg1.out')
  pred = np.reshape(pred, (19, 16, 2))
  print(pred.shape, reg.shape)
  debugger = Debugger()
  debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
  debugger.addPoint2D(pred[i], (255, 0, 0))
  a = np.expand_dims(reg[i], 1)
  debugger.addPoint3D(np.concatenate([pred[i], (a + 1) / 2. * 256], axis = 1))
  debugger.showImg(pause=True)
  debugger.show3D()

if __name__ == '__main__':
  main()

# [ 148,  202.3125],
# [ 144,  146.25  ],
# [ 104,  121.625 ],
# [ 116,  117.8125],
# [ 148,  138.3125],
# [ 152,  190.375 ],
# [ 112,  121.75  ],
# [ 116,  69.8125],
# [ 116,  61.8125],
# [ 128,  26.0    ],
# [ 128,  122.0    ],
# [ 108,  97.6875],
# [ 116,  69.8125],
# [ 116,  69.8125],
# [ 116,  97.8125],
# [ 128,  122.0    ]

# [ 248.35119629
#   257.1585083
#   255.18921661 256.5622406 ,
#         262.81010437,  255.33033752,  255.55664825,  255.32247925,
#         253.74448395,  258.03192139,  233.26204681,  239.06238556,
#         249.04217529,  258.13237   ,  270.01130676,  272.71896362]

       	
#        144,146.25  ,257.1585083 
#        104,121.625 ,255.18921661
#        116,117.8125,256.5622406 
#        148,138.3125,262.81010437
#        152,190.375 ,255.33033752
#        112,121.75  ,255.55664825
#        116,69.8125,255.32247925
#        116,61.8125,253.74448395
#        128,26    ,258.03192139
#        128,122    ,233.26204681
#        108,97.6875,239.06238556
#        116,69.8125,249.04217529
#        116,69.8125,258.13237   
#        116,97.8125,270.01130676
#        128,122    ,272.71896362
