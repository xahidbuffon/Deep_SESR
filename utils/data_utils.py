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
import fnmatch
import numpy as np
from scipy import misc

def deprocess(x):
    # [-1,1] -> [0, 1]
    return (x+1.0)*0.5

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
    img = misc.imread(paths, mode=mode_).astype(np.float)
    img = misc.imresize(img, res)
    return img


