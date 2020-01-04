
# coding: utf-8

# In[ ]:


import numpy as np
import os
import skimage
import PIL
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from skimage.exposure import histogram
from skimage.util import random_noise
import math
import cv2
from skimage import io
from skimage import img_as_ubyte , img_as_int

from skimage.filters import threshold_otsu
from scipy.signal import find_peaks
import porespy as ps
import matplotlib.pyplot as plt
from  scipy import stats
from skimage.external import tifffile as tif
from skimage.filters import unsharp_mask


    
def sythetic_gaussian_image(im):
    thresholded_image=img_as_ubyte(im)

    inveted_image=np.where((thresholded_image==0.0), np.random.random_integers(30,40,1),thresholded_image)
    inveted_image=np.where((inveted_image==255.0),np.random.random_integers(100,130,1),inveted_image)
    noise=np.random.normal(size=inveted_image.shape)
    noise=img_as_ubyte(noise/np.maximum(np.absolute(noise.min()),noise.max()))
    blu=gaussian_filter(inveted_image-noise, sigma=5)
    result_3 = unsharp_mask(blu, radius=30, amount=3)

    sharped_image=img_as_ubyte(result_3/result_3.max())
    blu2=gaussian_filter((sharped_image), sigma=1)
    
    return blu2



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='generating sythentic porous images')

    parser.add_argument('--3D_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the 3D dataset')

    parser.add_argument('--grayscale_image_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the grayscale dataset')

    parser.add_argument('--GT_image_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the ground truth dataset')
    if not os.path.exists(3D_dir):
        os.makedirs(3D_dir)
    if not os.path.exists(grayscale_image_dir):
        os.makedirs(grayscale_image_dir)
    if not os.path.exists(GT_image_dir):
        os.makedirs(GT_image_dir)
    parser.add_argument("command",
                        metavar="<command>",
                        help="'blob' or 'overlapping_spheres'")
    args = parser.parse_args()


    if args.command == "blob":
        im1 = np.invert(ps.generators.blobs(shape=[20,256,256], porosity=0.6, blobiness=2))
        noise=sythetic_gaussian_image(im1)
        for i in range(20):
            name1='blob'+str(blob)+'_'+'p'+str(porosity)+str(i)+'.png'     
            io.imsave(os.path.join(porespy_image_path,name1),sythetic_gaussian_image(im1[i]))
            plt.imsave(os.path.join(porespy_ground_truth_path,name1),im1[i])

    if args.command == "overlapping_spheres":
        im1 = np.invert(ps.generators.overlapping_spheres(shape=[100,256,256], porosity=0.2, radius=30))
        noise=sythetic_gaussian_image(im1)

        for i in range(20):
            name1='overlapping_spheres'+str(radius)+'_'+'p'+str(porosity)+str(i)+'.png'     
            io.imsave(os.path.join(porespy_image_path,name1),sythetic_gaussian_image(im1[i]))
            plt.imsave(os.path.join(porespy_ground_truth_path,name1),im1[i])
        



