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
import tensorflow as tf
## local libs
from utils.data_utils import getPaths
from utils.data_utils import preprocess, deprocess
from utils.data_utils import deprocess_uint8, deprocess_mask



class Deep_SESR_GraphOP:
    def __init__(self, frozen_model_path):
        # loads the graph  
        self.gen_graph = tf.Graph()
        with self.gen_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # graphOP handler for Deep SESR  
        ops = self.gen_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        self.output_dict = {}
        for key in ['conv2d_42/Tanh', 'conv2d_45/Tanh', 'conv2d_48/Sigmoid']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                self.output_dict[key] = self.gen_graph.get_tensor_by_name(tensor_name)
        self.inp_im_tensor = self.gen_graph.get_tensor_by_name('input_3:0')
        self.sess = tf.Session(graph=self.gen_graph)


    def predict(self, frame):
        """
          Given an input frame, returns: 
           - en_im: enhanced image (same size)
           - sesr_im: enhanced and super-resolution image (size * scale)
           - mask: saliency mask (same size) 
        """
        output_dict = self.sess.run(self.output_dict, 
                                   feed_dict={self.inp_im_tensor: np.expand_dims(frame, 0)})
        en_im = output_dict['conv2d_42/Tanh'][0]
        sesr_im = output_dict['conv2d_45/Tanh'][0]
        mask = output_dict['conv2d_48/Sigmoid'][0]
        return en_im, sesr_im, mask


## load specific model
ckpt_name =  "deep_sesr_2x_1d"
frozen_model_path = os.path.join("models/", ckpt_name+".pb")
assert (os.path.exists(frozen_model_path))
generator = Deep_SESR_GraphOP(frozen_model_path)

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

## create dir for output test data
samples_dir = os.path.join("data/output/", "tf_out")
if not os.path.exists(samples_dir): os.makedirs(samples_dir)

# testing loop
times = []; 
for img_path in test_paths:
    # prepare data
    img_name = basename(img_path).split('.')[0]
    img_lrd = np.array(Image.open(img_path).resize(lr_res))
    im = preprocess(img_lrd)
    # get output
    s = time.time()
    gen_lr, gen_hr, gen_mask = generator.predict(im)
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


