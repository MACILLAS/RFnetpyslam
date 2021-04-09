"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import os
import cv2 
import torch
import random
import argparse
from torchvision import transforms
import numpy as np

import config
config.cfg.set_lib('rfnet')

from utils.train_utils import parse_batch
from hpatch_dataset import *
from utils.eval_utils import *
from utils.common_utils import gct

from model.rf_des import HardNetNeiMask
from model.rf_det_so import RFDetSO
from model.rf_net_so import RFNetSO
from config_rfnet import cfg

from threading import RLock

from utils_sys import Printer

kVerbose = True   


class rfnetOptions:
    def __init__(self, do_cuda=True):
        #Pyslam is using config.py in it's directory rather than thirdparty (i think we need to change the file name of
        #config.py in rfnet to avoid conflict)
        #print(f"{gct()} : start time")
        cfg.MODEL.COO_THRSH = 5  # this is default ct argument for pixel distance threshold

        random.seed(cfg.PROJ.SEED)
        torch.manual_seed(cfg.PROJ.SEED)
        np.random.seed(cfg.PROJ.SEED)

        use_cuda = torch.cuda.is_available() & do_cuda
        device = torch.device('cuda' if use_cuda else 'cpu')
        print('rfnet using ', device)
        self.cuda = use_cuda


# image from pytorch tensor to cv2 format
def reverse_img(img):
    """
    reverse image from tensor to cv2 format
    :param img: tensor
    :return: RBG image
    """
    img = img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
    img = (img * 255).astype(np.uint8)  # change to opencv format
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # gray to rgb
    return img

# this is an old function point
def transpose_des(des):
    if des is not None: 
        return des.T 
    else: 
        return None 


# interface for pySLAM 
class rfnet:
    def __init__(self, do_cuda=True): 
        self.lock = RLock()
        self.opts = rfnetOptions(do_cuda)
        print(self.opts)        

        print('rfnet init')

        #print(f"{gct()} model init")
        print("model init")
        det = RFDetSO(
            cfg.TRAIN.score_com_strength,
            cfg.TRAIN.scale_com_strength,
            cfg.TRAIN.NMS_THRESH,
            cfg.TRAIN.NMS_KSIZE,
            cfg.TRAIN.TOPK,
            cfg.MODEL.GAUSSIAN_KSIZE,
            cfg.MODEL.GAUSSIAN_SIGMA,
            cfg.MODEL.KSIZE,
            cfg.MODEL.padding,
            cfg.MODEL.dilation,
            cfg.MODEL.scale_list,
        )
        des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
        model = RFNetSO(
            det, des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, cfg.TRAIN.TOPK
        )

        device = torch.device("cuda")
        model = model.to(device)
        #resume = args.resume
        #resume = "/home/cviss3/PycharmProjects/gensynth_dev_env/pyslam/thirdparty/rfnet/runs/10_24_09_25/model/e121_NN_0.480_NNT_0.655_NNDR_0.813_MeanMS_0.649.pth.tar"
        resume = "/content/RFnetpyslam/thirdparty/rfnet/runs/10_24_09_25/model/e121_NN_0.480_NNT_0.655_NNDR_0.813_MeanMS_0.649.pth.tar"

        print('==> Loading pre-trained network.')
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint["state_dict"])
        self.fe = model
        print('==> Successfully loaded pre-trained network.')

        self.device = device
        self.pts = []
        self.kps = []        
        self.des = []
        self.img = []
        self.heatmap = [] 
        self.frame = None 
        self.frameFloat = None 
        self.keypoint_size = 20  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint 

    # convert kp to cv2 keypointrs
    def to_cv2_kp(self, kp):
        # kp is like [batch_idx, y, x, channel]
        kp = kp.tolist()#added 04/06
        #print(kp)
        kps = [cv2.KeyPoint(p[2], p[1], 0) for p in kp]
        return kps

    # compute both keypoints and descriptors
    def detectAndCompute(self, frame, mask=None):  # mask is a fake input 
        with self.lock: 
            self.frame = frame 
            self.frameFloat = (frame.astype('float32') / 255.)
            # detectAndComputeNoLoad (pass in the image)
            #self.pts, self.des, self.img = self.fe.detectAndComputeNoLoad(im=self.frameFloat, device=self.device, output_size=(370, 1226))
            self.pts, self.des, self.img = self.fe.detectAndComputeNoLoad(im=self.frameFloat, device=self.device, output_size=frame.shape[0:2])


            self.kps = self.to_cv2_kp(self.pts)
            self.des = self.des.cpu().detach().numpy() # added 04/06

            if kVerbose:
                print('detector: rfnet, #features: ', len(self.kps), ', frame res: ', frame.shape[0:2])
            #I think the descriptors don't need transpose as in original superpoint code
            return self.kps, self.des
            
    # return keypoints if available otherwise call detectAndCompute()    
    def detect(self, frame, mask=None):  # mask is a fake input  
        with self.lock:         
            #if self.frame is not frame:
            self.detectAndCompute(frame)        
            return self.kps
    
    # return descriptors if available otherwise call detectAndCompute()  
    def compute(self, frame, kps=None, mask=None): # kps is a fake input, mask is a fake input
        with self.lock: 
            if self.frame is not frame:
                Printer.orange('WARNING: RFNET is recomputing both kps and des on last input frame', frame.shape)
                self.detectAndCompute(frame)
            return self.kps, self.des
