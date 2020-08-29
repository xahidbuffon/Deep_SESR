#!/usr/bin/env python
"""
# > Script for evaluating 2x SESR 
#    - Paper: https://arxiv.org/pdf/2002.01155.pdf
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
"""
import os
import time
import numpy as np
from glob import glob
from ntpath import basename
from os.path import join, exists
from PIL import Image
from keras.models import model_from_json
## local libs
from utils.data_utils import getPaths
from utils.data_utils import preprocess, deprocess
from utils.data_utils import deprocess_uint8, deprocess_mask

# input and output data shape
scale = 2 
hr_w, hr_h = 640, 480 # HR
lr_w, lr_h = 320, 240 # LR (1/2x)
lr_res, lr_shape = (lr_w, lr_h), (lr_h, lr_w, 3)
hr_res, hr_shape = (hr_w, hr_h), (hr_h, hr_w, 3)

## for testing arbitrary local data
data_dir = "data/sample_test_ufo/lrd/"
#data_dir = "data/test_mixed/"
test_paths = getPaths(data_dir)
print ("{0} test images are loaded".format(len(test_paths)))

## load specific model
ckpt_name =  "deep_sesr_2x_1d"
model_h5 = join("models/", ckpt_name+".h5")  
model_json = join("models/", ckpt_name + ".json")

# load model
assert (exists(model_h5) and exists(model_json))
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
generator = model_from_json(loaded_model_json)
generator.load_weights(model_h5)
print("\nLoaded data and model")

## create dir for output test data
samples_dir = join("data/output/", "keras_out")
if not exists(samples_dir): os.makedirs(samples_dir)

# testing loop
times = []; 
for img_path in test_paths:
    # prepare data
    img_name = basename(img_path).split('.')[0]
    img_lrd = np.array(Image.open(img_path).resize(lr_res))
    im = np.expand_dims(preprocess(img_lrd), axis=0)
    # get output
    s = time.time()
    gen_op = generator.predict(im)
    gen_lr, gen_hr, gen_mask = gen_op[0], gen_op[1], gen_op[2]
    tot = time.time()-s
    times.append(tot)
    # process raw outputs 
    gen_lr = deprocess_uint8(gen_lr).reshape(lr_shape)
    gen_hr = deprocess_uint8(gen_hr).reshape(hr_shape)
    gen_mask = deprocess_mask(gen_mask).reshape(lr_h, lr_w) 
    # save generated images
    Image.fromarray(img_lrd).save(join(samples_dir, img_name+'.png'))
    Image.fromarray(gen_lr).save(join(samples_dir, img_name+'_En.png'))
    Image.fromarray(gen_mask).save(join(samples_dir, img_name+'_Sal.png'))
    Image.fromarray(gen_hr).save(join(samples_dir, img_name+'_SESR.png'))
    print ("tested: {0}".format(img_path))

# some statistics    
num_test = len(test_paths)
if (num_test==0):
    print ("\nFound no images for test")
else:
    print ("\nTotal images: {0}".format(num_test)) 
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: {0} sec at {1} fps".format(Ttime, 1./Mtime))
    print("\nSaved generated images in in {0}\n".format(samples_dir))


