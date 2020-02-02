#!/usr/bin/env python
"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
#    - Underwater Image Quality Measure (UIQM)
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# // usage: for academic and educational purposes only
"""
## python libs
import os
import ntpath
import numpy as np
from scipy import misc
## local libs
from utils.data_utils import getPaths
from utils.uiqm_utils import getUIQM
from utils.ssm_psnr_utils import getSSIM, getPSNR

# measurement in a common dimension
im_w, im_h = 320, 240

## data paths
REAL_im_dir = "data/sample_test_ufo/lrd/"  # real/input im-dir with {f.ext}
GEN_im_dir  = "data/output/keras_out/"  # generated im-dir with {f_gen.ext}
GTr_im_dir  = "data/sample_test_ufo/hr/"  # ground truth im-dir with {f.ext}
REAL_paths, GEN_paths = getPaths(REAL_im_dir), getPaths(GEN_im_dir)

## mesures uqim for all images in a directory
def measure_UIQMs(dir_name, file_ext=None):
    """
      # measured in RGB
      Assumes:
        * dir_name contain generated images 
        * to evaluate on all images: file_ext = None 
        * to evaluate images that ends with "_SESR.png" or "_En.png"  
            * use file_ext = "_SESR.png" or "_En.png" 
    """
    if file_ext:
        paths = [p for p in getPaths(dir_name) if p.endswith(file_ext)]
    else: 
        paths = getPaths(dir_name)
    uqims = []
    for img_path in paths:
        #print (paths)
        im = misc.imresize(misc.imread(img_path), (im_h, im_w))
        uqims.append(getUIQM(im))
    return np.array(uqims)


def measure_SSIM(GT_dir, Gen_dir):
    """
      # measured in RGB
      Assumes:
        * GT_dir contain ground-truths {filename.ext}
        * Gen_dir contain generated images {filename_SESR.png} 
    """
    GT_paths, Gen_paths = getPaths(GT_dir), getPaths(Gen_dir)
    ssims = []
    for img_path in GT_paths:
        name_split = ntpath.basename(img_path).split('.')
        gen_path = os.path.join(Gen_dir, name_split[0]+'_SESR.png') 
        ## >> To evaluate only enhancement: use: 
        #gen_path = os.path.join(Gen_dir, name_split[0]+'_En.png') 
        if (gen_path in Gen_paths):
            r_im = misc.imresize(misc.imread(img_path), (im_h, im_w))
            g_im = misc.imresize(misc.imread(gen_path), (im_h, im_w))
            assert (r_im.shape==g_im.shape), "The images should be of same-size"
            ssim = getSSIM(r_im, g_im)
            ssims.append(ssim)
    return np.array(ssims)


def measure_PSNR(GT_dir, Gen_dir):
    """
      # measured in lightness channel 
      Assumes:
        * GT_dir contain ground-truths {filename.ext}
        * Gen_dir contain generated images {filename_SESR.png}
    """
    GT_paths, Gen_paths = getPaths(GT_dir), getPaths(Gen_dir)
    ssims, psnrs = [], []
    for img_path in GT_paths:
        name_split = ntpath.basename(img_path).split('.')
        gen_path = os.path.join(Gen_dir, name_split[0]+'_SESR.png') 
        ## >> To evaluate only enhancement: use: 
        #gen_path = os.path.join(Gen_dir, name_split[0]+'_En.png') 
        if (gen_path in Gen_paths):
            r_im = misc.imresize(misc.imread(img_path, mode='L'), (im_h, im_w))
            g_im = misc.imresize(misc.imread(gen_path, mode='L'), (im_h, im_w))
            assert (r_im.shape==g_im.shape), "The images should be of same-size"
            psnr = getPSNR(r_im, g_im)
            psnrs.append(psnr)
    return np.array(psnrs)

### compute SSIM and PSNR
SSIM_measures = measure_SSIM(GTr_im_dir, GEN_im_dir)
PSNR_measures = measure_PSNR(GTr_im_dir, GEN_im_dir)
print ("SSIM >> Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
print ("PSNR >> Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))

### compute and compare UIQMs
gen_uqims = measure_UIQMs(GEN_im_dir, file_ext="_En.png")  # or file_ext="_SESR.png"
print ("Generated UQIM >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))



