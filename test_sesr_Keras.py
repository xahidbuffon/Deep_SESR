#!/usr/bin/env python
"""
# > Script for evaluating 2x SESR 
#    - Paper: https://arxiv.org/pdf/2002.01155.pdf
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
"""
import os
import time
import ntpath
import numpy as np
import skimage.transform
import imageio
from PIL import Image
from keras.models import model_from_json
## local libs
from utils.data_utils import getPaths, preprocess, deprocess

# input and output data shape
scale = 2 
hr_width, hr_height = 640, 480 # HR
lr_width, lr_height = 320, 240 # LR (1/2x)
lr_shape = (lr_height, lr_width, 3)
hr_shape = (hr_height, hr_width, 3)

## for testing arbitrary local data
data_dir = "data/sample_test_ufo/lrd/"
#data_dir = "data/test_mixed/"
test_paths = getPaths(data_dir)
print ("{0} test images are loaded".format(len(test_paths)))

## load specific model
ckpt_name =  "deep_sesr_2x_1d"
model_h5 = os.path.join("models/", ckpt_name+".h5")  
model_json = os.path.join("models/", ckpt_name + ".json")

# load model
assert (os.path.exists(model_h5) and os.path.exists(model_json))
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
generator = model_from_json(loaded_model_json)
generator.load_weights(model_h5)
print("\nLoaded data and model")

## create dir for output test data
samples_dir = os.path.join("data/output/", "keras_out")
if not os.path.exists(samples_dir): os.makedirs(samples_dir)

# testing loop
times = []; 
for img_path in test_paths:
    # prepare data
    img_name = ntpath.basename(img_path).split('.')[0]
    img_lrd = imageio.imread(img_path, pilmode='RGB').astype(np.float)
    #img_lrd = misc.imread(img_path, mode='RGB').astype(np.float) 
    inp_h, inp_w, _  =  img_lrd.shape # save the input im-shape
    img_lrd = skimage.transform.resize(img_lrd, (lr_height,lr_width))
    im = preprocess(img_lrd)
    im = np.expand_dims(im, axis=0)
    # generate enhanced image
    s = time.time()
    gen_op = generator.predict(im)
    gen_lr, gen_hr, gen_mask = gen_op[0], gen_op[1], gen_op[2]
    tot = time.time()-s
    times.append(tot)
    # save sample images
    gen_lr = deprocess(gen_lr).reshape(lr_shape)
    gen_hr = deprocess(gen_hr).reshape(hr_shape)
    gen_mask = gen_mask.reshape(lr_height, lr_width) 
    # little clean-up of the saliency map
    # >> may add further post-processing for more informative map
    gen_mask[gen_mask<0.1] = 0 
    # reshape and save generated images for observation 
    img_lrd = skimage.transform.resize(img_lrd, (inp_h, inp_w))
    gen_lr = skimage.transform.resize(gen_lr, (inp_h, inp_w))
    gen_mask = skimage.transform.resize(gen_mask, (inp_h, inp_w))
    gen_hr = skimage.transform.resize(gen_hr, (inp_h*scale, inp_w*scale))
    imageio.imsave(os.path.join(samples_dir, img_name+'.png'), img_lrd)
    imageio.imsave(os.path.join(samples_dir, img_name+'_En.png'), gen_lr)
    imageio.imsave(os.path.join(samples_dir, img_name+'_Sal.png'), gen_mask)
    imageio.imsave(os.path.join(samples_dir, img_name+'_SESR.png'), gen_hr)
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


