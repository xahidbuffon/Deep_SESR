#!/usr/bin/env python
"""
# > Various modules for handling data 
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
from __future__ import division
from __future__ import absolute_import
import os
import random
import fnmatch
import numpy as np
from PIL import Image
from scipy import misc

def deprocess(x):
    # [-1,1] -> [0, 1]
    return (x+1.0)*0.5

def preprocess(x):
    # [0,255] -> [-1, 1]
    return (x/127.5)-1.0

def preprocess_mask(x):
    # attention mask [0,255] -> [0, 1]
    x = x*1.0 / 255.0
    x[x<0.2] = 0
    return np.expand_dims(x, axis=3)

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
    img = misc.imread(paths, mode=mode_).astype(np.float)
    img = misc.imresize(img, res)
    return img


def refine_mask(mask, th=0.5):
    if(np.max(mask) > 1): mask = mask/255
    mask[mask < th]  = 0
    mask[mask >= th] = 1
    return mask


def get_CMI(imgs, masks):
    cmis = []
    for i in range(imgs.shape[0]):
        mean = np.sum(imgs[i,:], axis=-1)
        mask = np.reshape(masks[i,:], mean.shape)
        F = mean*mask
        B = mean*(1-mask)
        cmis.append(F-B)
    return np.array(cmis)


##################################################################################
class dataLoaderUFO():
    def __init__(self, data_path="/mnt/data1/ImageSR/UFO/", SCALE=2, set_UFO=None):
        # SCALE = 2 (320, 240)  => (640,480)
        # SCALE = 3 (214, 160)  => (640,480)
        # SCALE = 4 (160, 120)  => (640,480) 
        self.SCALE = SCALE
        if (self.SCALE==2):   
            self.lr_res_ = (240, 320)
        elif (self.SCALE==3):   
            self.lr_res_ = (160, 214)
        else:   
            self.lr_res_ = (120, 160)
        self.hr_folder = "hr/"
        self.mask_folder = "mask/"
        self.lr_dist_folder = "lrd/"  
        self.get_train_and_val_paths(data_path, set_UFO)


    def get_train_and_val_paths(self, data_path, set_UFO):
        # set_UFO = "Set_U", "Set_F", or "Set_O"
        # if None, train all together
        if set_UFO not in ["Set_U", "Set_F", "Set_O"]: 
            all_sets = ["Set_U", "Set_F", "Set_O"]
        else:
            all_sets = [set_UFO]
        
        self.num_train, self.num_val = 0, 0
        self.train_lrd_paths, self.val_lrd_paths= [], [] 
        self.train_lr_paths, self.val_lr_paths  = [], []
        self.train_hr_paths, self.val_hr_paths  = [], []
        self.train_mask_paths, self.val_mask_paths =  [], []

        for p in all_sets:
            train_dir = os.path.join(data_path+p, "train/")
            all_train_paths = self.get_lr_hr_paths(train_dir)
            self.num_train += all_train_paths[0]
            self.train_lrd_paths += all_train_paths[1] 
            self.train_lr_paths  += all_train_paths[2] 
            self.train_hr_paths  += all_train_paths[3] 
            self.train_mask_paths += all_train_paths[4] 
        print ("Loaded {0} pairs of image-paths for training".format(self.num_train)) 

        for p in all_sets:
            val_dir = os.path.join(data_path+p, "val/")
            all_val_paths = self.get_lr_hr_paths(val_dir)
            self.num_val += all_val_paths[0]
            self.val_lrd_paths += all_val_paths[1]
            self.val_lr_paths += all_val_paths[2]
            self.val_hr_paths += all_val_paths[3]
            self.val_mask_paths += all_val_paths[4]
        print ("Loaded {0} pairs of image-paths for validation".format(self.num_val)) 


    def get_lr_hr_paths(self, data_dir):
        lrd_path = sorted(os.listdir(data_dir+self.lr_dist_folder))   
        hr_path = sorted(os.listdir(data_dir+self.hr_folder))
        mask_path = sorted(os.listdir(data_dir+self.mask_folder))
        num_paths = min(len(lrd_path), len(hr_path))
        all_hr_paths, all_lrd_paths, all_mask_paths = [], [], []
        for f in lrd_path[:num_paths]:
            all_lrd_paths.append(os.path.join(data_dir+self.lr_dist_folder, f))
            all_hr_paths.append(os.path.join(data_dir+self.hr_folder, f))
            all_mask_paths.append(os.path.join(data_dir+self.mask_folder, f))
        all_lr_paths = all_hr_paths # same hr images will be resized 
        return (num_paths, all_lrd_paths, all_lr_paths, all_hr_paths, all_mask_paths)


    def load_batch(self, batch_size=1, data_augment=True):
        self.n_batches = self.num_train//batch_size
        for i in range(self.n_batches-1):
            batch_lrd = self.train_lrd_paths[i*batch_size:(i+1)*batch_size]
            batch_lr = self.train_lr_paths[i*batch_size:(i+1)*batch_size]
            batch_hr = self.train_hr_paths[i*batch_size:(i+1)*batch_size]
            batch_m = self.train_mask_paths[i*batch_size:(i+1)*batch_size]

            imgs_lrd, imgs_lr, imgs_hr, imgs_mask = [], [], [], []
            for idx in range(len(batch_lr)): 
                img_lrd = read_and_resize(batch_lrd[idx], res=self.lr_res_)
                img_lr = read_and_resize(batch_lr[idx], res=self.lr_res_)
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
            cmis = get_CMI(imgs_lr,imgs_mask)
            yield imgs_lrd, imgs_lr, imgs_hr, imgs_mask, cmis


    def load_val_data(self, batch_size=1):
        idx = np.random.choice(np.arange(self.num_val), batch_size, replace=False)
        paths_lrd = [self.val_lrd_paths[i] for i in idx]
        paths_lr = [self.val_lr_paths[i] for i in idx]
        paths_hr = [self.val_hr_paths[i] for i in idx]
        paths_mask = [self.val_mask_paths[i] for i in idx]
        imgs_lrd, imgs_lr, imgs_hr, imgs_mask = [], [], [], []
        for idx in range(len(paths_hr)):
            img_lrd = read_and_resize(paths_lrd[idx], res=self.lr_res_)
            img_lr = read_and_resize(paths_lr[idx], res=self.lr_res_)
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
        cmis = get_CMI(imgs_lr,imgs_mask)
        return imgs_lrd, imgs_lr, imgs_hr, imgs_mask, cmis


