#!/usr/bin/env python
"""
# > Various modules for handling data 
#    - Paper: https://arxiv.org/pdf/2002.01155.pdf
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
"""
from __future__ import division
from __future__ import absolute_import
import os
import random
import fnmatch
import numpy as np
from scipy import misc
from glob import glob
import skimage.transform

def deprocess(x):
    # [-1,1] -> [0, 1]
    return (x+1.0)*0.5

def deprocess_uint8(x):
    # [-1,1] -> np.uint8 [0, 255]
    im = ((x+1.0)*127.5)
    return np.uint8(im)

def preprocess(x):
    # [0,255] -> [-1, 1]
    return (x/127.5)-1.0

def getPaths(data_dir):
    exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if (fnmatch.fnmatch(filename, pattern)):
                    fname_ = os.path.join(d,filename)
                    image_paths.append(fname_)
    return np.asarray(image_paths)

def read_and_resize(paths, res=(480, 640), mode_='RGB'):
    img = imageio.imread(paths, pilmode=mode_).astype(np.float)
    img = skimage.transform.resize(img, res)
    return img

def preprocess_mask(x):
    # attention mask [0,255] -> [0, 1]
    x = x*1.0 / 255.0
    x[x<0.2] = 0
    return np.expand_dims(x, axis=3)

def deprocess_mask(x):
    # [0,1] -> np.uint8 [0, 255]
    x *= 255.
    x[x<0.1] = 0
    return np.uint8(x)

def normalize_mask(img):
    # img [0, 1] => normaized [0, 255]
    img *= 255. 
    M, m = np.max(img), np.min(img)
    return (img-m)/(M-m)


def get_cmi(imgs, masks):
    cmis = []
    for i in range(imgs.shape[0]):
        mean = np.sum(imgs[i,:], axis=-1)
        mask = np.reshape(masks[i,:], mean.shape)
        F = mean*mask
        B = mean*(1-mask)
        cmis.append(F-B)
    return np.array(cmis)


class dataLoaderUFO():
    def __init__(self, data_path, SCALE=2):
        # SCALE = 2 (320, 240)  => (640,480)
        # SCALE = 3 (214, 160)  => (640,480)
        # SCALE = 4 (160, 120)  => (640,480) 
        self.SCALE = SCALE
        if (self.SCALE==2): self.lr_res_ = (240, 320)
        elif (self.SCALE==3): self.lr_res_ = (160, 214)
        else: self.lr_res_ = (120, 160)
        self.hr_folder = "hr/"
        self.mask_folder = "mask/"
        self.lr_dist_folder = "lrd/"  
        self.get_train_and_val_paths(data_path)
        

    def get_train_and_val_paths(self, data_path):
        self.num_train, self.num_val = 0, 0
        self.train_lrd_paths, self.val_lrd_paths= [], []
        self.train_hr_paths, self.val_hr_paths  = [], []
        self.train_mask_paths, self.val_mask_paths =  [], []

        data_dir = os.path.join(data_path, "train_val/")
        lrd_path = sorted(os.listdir(data_dir+self.lr_dist_folder))   
        hr_path = sorted(os.listdir(data_dir+self.hr_folder))
        mask_path = sorted(os.listdir(data_dir+self.mask_folder))
        num_paths = min(len(lrd_path), len(hr_path), len(mask_path))
        all_idx = range(num_paths)
        # 95% train-val splits
        random.shuffle(all_idx)
        self.num_train = int(num_paths*0.95)
        self.num_val = num_paths-self.num_train
        train_idx = set(all_idx[:self.num_train])
        # split data paths to training and validation sets
        for i in range(num_paths):
            if i in train_idx:
                self.train_lrd_paths.append(data_dir+self.lr_dist_folder+lrd_path[i])
                self.train_hr_paths.append(data_dir+self.hr_folder+hr_path[i]) 
                self.train_mask_paths.append(data_dir+self.mask_folder+mask_path[i])
            else:
                self.val_lrd_paths.append(data_dir+self.lr_dist_folder+lrd_path[i])
                self.val_hr_paths.append(data_dir+self.hr_folder+hr_path[i]) 
                self.val_mask_paths.append(data_dir+self.mask_folder+mask_path[i])
        print ("Loaded {0} samples for training".format(self.num_train))
        print ("Loaded {0} samples for validation".format(self.num_val)) 


    def load_batch(self, batch_size=1):
        self.n_batches = self.num_train//batch_size
        for i in range(self.n_batches-1):
            batch_lrd = self.train_lrd_paths[i*batch_size:(i+1)*batch_size]
            batch_hr = self.train_hr_paths[i*batch_size:(i+1)*batch_size]
            batch_m = self.train_mask_paths[i*batch_size:(i+1)*batch_size]

            imgs_lrd, imgs_lr, imgs_hr, imgs_mask = [], [], [], []
            for idx in range(len(batch_lrd)): 
                img_lrd = read_and_resize(batch_lrd[idx], res=self.lr_res_)
                img_lr = read_and_resize(batch_hr[idx], res=self.lr_res_)
                img_hr = read_and_resize(batch_hr[idx], res=(480, 640))
                img_mask = read_and_resize(batch_m[idx], res=self.lr_res_, mode_='L')
                imgs_lrd.append(img_lrd)
                imgs_lr.append(img_lr)
                imgs_hr.append(img_hr)
                imgs_mask.append(img_mask)
            imgs_lrd = preprocess(np.array(imgs_lrd))
            imgs_lr = preprocess(np.array(imgs_lr))
            imgs_hr = preprocess(np.array(imgs_hr))
            imgs_mask = preprocess_mask(np.array(imgs_mask))
            cmis = get_cmi(imgs_lr,imgs_mask)
            yield imgs_lrd, imgs_lr, imgs_hr, imgs_mask, cmis


    def load_val_data(self, batch_size=1):
        idx = np.random.choice(np.arange(self.num_val), batch_size, replace=False)
        paths_lrd = [self.val_lrd_paths[i] for i in idx]
        paths_hr = [self.val_hr_paths[i] for i in idx]
        paths_mask = [self.val_mask_paths[i] for i in idx]
        imgs_lrd, imgs_lr, imgs_hr, imgs_mask = [], [], [], []
        for idx in range(len(paths_hr)):
            img_lrd = read_and_resize(paths_lrd[idx], res=self.lr_res_)
            img_lr = read_and_resize(paths_hr[idx], res=self.lr_res_)
            img_hr = read_and_resize(paths_hr[idx], res=(480, 640))
            img_mask = read_and_resize(paths_mask[idx], res=self.lr_res_, mode_='L')
            imgs_lrd.append(img_lrd)
            imgs_lr.append(img_lr)
            imgs_hr.append(img_hr)
            imgs_mask.append(img_mask)
        imgs_lrd = preprocess(np.array(imgs_lrd))
        imgs_lr = preprocess(np.array(imgs_lr))
        imgs_hr = preprocess(np.array(imgs_hr))
        imgs_mask = preprocess_mask(np.array(imgs_mask))
        cmis = get_cmi(imgs_lr,imgs_mask)
        return imgs_lrd, imgs_lr, imgs_hr, imgs_mask, cmis


