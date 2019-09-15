import boto3
import numpy as np
import os
import boto3
from math import floor

import sagemaker
from skimage import util 
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
from skimage import exposure,color, img_as_int
from skimage.io import imread as pngread
from skimage.io import imsave as pngsave
from skimage.segmentation import mark_boundaries
from skimage.segmentation import watershed
from skimage import color
from skimage.feature import peak_local_max
from skimage.color import label2rgb
import cv2
from rolling_ball_filter import rolling_ball_filter
import random
import boto3
import numpy as np
import threading

files = []
s3 = boto3.resource('s3')
s3_resource = boto3.resource('s3')
s3meadata = s3_resource.Bucket(name='meadata')
sess = sagemaker.Session()
bucket = sess.default_bucket()  
prefix = 'fresh_train_trial'
train_channel = prefix + '/train'
validation_channel = prefix + '/validation'
train_annotation_channel = prefix + '/train_annotation'
validation_annotation_channel = prefix + '/validation_annotation'

#Hela cell dataset
def proccesshelafiles(f, s3 = s3, sess = sess, bucket = bucket, prefix = prefix):
    train_channel = prefix + '/train'
    validation_channel = prefix + '/validation'
    train_annotation_channel = prefix + '/train_annotation'
    validation_annotation_channel = prefix + '/validation_annotation'
    if 'jpg' in f:
        file = f
        if 'segproj/hela_dataset_training_data/train/' in f:
            jpgpath = '/tmp/'+'hela_'+file.split('/')[-1].split('.')[0]+'_'+'raw.jpg'
            s3.meta.client.download_file('meadata', file, jpgpath)
            inverted_img = pngread(jpgpath)
            num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
            image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**num - 1))
            image =  cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA) 
            pngsave(jpgpath,img_as_ubyte(image))
            sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=train_channel)
        elif 'segproj/hela_dataset_training_data/val/' in f:
            jpgpath = '/tmp/'+'hela_'+file.split('/')[-1].split('.')[0]+'_'+'raw.jpg'
            s3.meta.client.download_file('meadata', file, jpgpath)
            inverted_img = pngread(jpgpath)
            num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
            image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**num - 1))
            image =  cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA) 
            pngsave(jpgpath,img_as_ubyte(image))
            sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=validation_channel)
    elif 'png' in f:
        file = f
        if 'segproj/hela_dataset_training_data/train_annotation/' in f:
            pngpath = '/tmp/'+'hela_'+file.split('/')[-1].split('.')[0]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath)
            im1 = pngread(pngpath)
            num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
            image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
            image = img_as_ubyte(image)
            im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
            im2 = img_as_int(im)
            im3 = np.zeros([im2.shape[0],im2.shape[1]])
            im3 = im2[:,:,0]+im2[:,:,1]+im2[:,:,2]
            im3 = np.uint8((im3>0))
            im3 =  cv2.resize(im3, (1024,1024), interpolation = cv2.INTER_AREA)                 
            pngsave(pngpath,im3, check_contrast=False)            
            sess.upload_data(path=pngpath, bucket=bucket, key_prefix=train_annotation_channel)
        elif 'segproj/hela_dataset_training_data/val_annotation/' in f:
            pngpath = '/tmp/'+'hela_'+file.split('/')[-1].split('.')[0]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath)
            im1 = pngread(pngpath)
            num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
            image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
            image = img_as_ubyte(image)
            im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
            im2 = img_as_int(im)
            im3 = np.zeros([im2.shape[0],im2.shape[1]])
            im3 = im2[:,:,0]+im2[:,:,1]+im2[:,:,2]
            im3 = np.uint8((im3>0))
            im3 =  cv2.resize(im3, (1024,1024), interpolation = cv2.INTER_AREA)                 
            pngsave(pngpath,im3, check_contrast=False)            
            sess.upload_data(path=pngpath, bucket=bucket, key_prefix=validation_annotation_channel)

#figure8

def proccessfigure8files(f, s3 = s3, sess = sess, bucket = bucket, prefix = prefix):
    train_channel = prefix + '/train'
    validation_channel = prefix + '/validation'
    train_annotation_channel = prefix + '/train_annotation'
    validation_annotation_channel = prefix + '/validation_annotation'
    if 'tif' in f:
        file = f
        if 'segproj/training_data/train/' in f:
            jpgpath = '/tmp/'+'fig8_raw_'+file.split('/')[-2]+'_'+'raw.jpg'
            s3.meta.client.download_file('meadata', file, jpgpath.replace('jpg','tif'))
            inverted_img = imread(jpgpath.replace('jpg','tif'))
            num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
            image = exposure.rescale_intensity(inverted_img, out_range=(2**num - 1, 0))
#             image = util.invert(image)
            imsave(jpgpath,img_as_ubyte(image))
            sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=train_channel)
        elif 'segproj/training_data/val/' in f:
            jpgpath = '/tmp/'+'fig8_raw_'+file.split('/')[-2]+'_'+'raw.jpg'
            s3.meta.client.download_file('meadata', file, jpgpath.replace('jpg','tif'))
            inverted_img = imread(jpgpath.replace('jpg','tif'))
            num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
            image = exposure.rescale_intensity(inverted_img, out_range=(2**num - 1, 0))
#             image = util.invert(image)
            imsave(jpgpath,img_as_ubyte(image))
            sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=validation_channel)
    elif 'instances_ids.png' in f:
        file = f
        if 'segproj/training_data/train/' in f:
            pngpath = '/tmp/'+'fig8_raw_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath)
            im1 = pngread(pngpath)
            num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
            image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
            image = img_as_ubyte(image)
            im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
            im2 = img_as_int(im)
            im3 = np.zeros([im2.shape[0],im2.shape[1]])
            im3 = im2[:,:,0]+im2[:,:,1]+im2[:,:,2]
            im3 = np.uint8((im3>0))
            im3 =  cv2.resize(im3, (1024,1024), interpolation = cv2.INTER_AREA)                 
            pngsave(pngpath,im3, check_contrast=False)            
            sess.upload_data(path=pngpath, bucket=bucket, key_prefix=train_annotation_channel)
        elif 'segproj/training_data/val/' in f:
            pngpath = '/tmp/'+'fig8_raw_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath)
            im1 = pngread(pngpath)
            num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
            image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
            image = img_as_ubyte(image)
            im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
            im2 = img_as_int(im)
            im3 = np.zeros([im2.shape[0],im2.shape[1]])
            im3 = im2[:,:,0]+im2[:,:,1]+im2[:,:,2]
            im3 = np.uint8((im3>0))
            im3 =  cv2.resize(im3, (1024,1024), interpolation = cv2.INTER_AREA)                 
            pngsave(pngpath,im3, check_contrast=False)            
            sess.upload_data(path=pngpath, bucket=bucket, key_prefix=validation_annotation_channel)
            
#usiigaci
def proccessusiigacifiles(f, s3 = s3, sess = sess, bucket = bucket, prefix = prefix):
    train_channel = prefix + '/train'
    validation_channel = prefix + '/validation'
    train_annotation_channel = prefix + '/train_annotation'
    validation_annotation_channel = prefix + '/validation_annotation'
    if 'tif' in f:
        file = f
        if 'segproj/usiigaci_train_data/train/' in f:
            jpgpath = '/tmp/'+'usiigaci_'+file.split('/')[-2]+'_'+'raw.jpg'
            s3.meta.client.download_file('meadata', file, jpgpath.replace('jpg','tif'))
            inverted_img = util.invert(imread(jpgpath.replace('jpg','tif')))
            num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
            image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**num - 1))
            image =  cv2.resize(img_as_ubyte(image), (1024,1024), interpolation = cv2.INTER_AREA)
            imsave(jpgpath,image)
            sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=train_channel)
        elif 'segproj/usiigaci_train_data/val/' in f:
            jpgpath = '/tmp/'+'usiigaci_'+file.split('/')[-2]+'_'+'raw.jpg'
            s3.meta.client.download_file('meadata', file, jpgpath.replace('jpg','tif'))
            inverted_img = util.invert(imread(jpgpath.replace('jpg','tif')))
            num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
            image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**num - 1))
            image =  cv2.resize(img_as_ubyte(image), (1024,1024), interpolation = cv2.INTER_AREA)
            imsave(jpgpath,image)
            sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=validation_channel)
    elif 'instances_ids.png' in f:
        file = f
        if 'segproj/usiigaci_train_data/train/' in f:
            pngpath = '/tmp/'+'usiigaci_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath)
            im1 = pngread(pngpath)
            num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
            image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
            image = img_as_ubyte(image)
            im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
            im2 = img_as_int(im)
            im3 = np.zeros([im2.shape[0],im2.shape[1]])
            im3 = im2[:,:,0]+im2[:,:,1]+im2[:,:,2]
            im3 = np.uint8((im3>0))
            im3 =  cv2.resize(im3, (1024,1024), interpolation = cv2.INTER_AREA)                 
            pngsave(pngpath,im3, check_contrast=False)
            sess.upload_data(path=pngpath, bucket=bucket, key_prefix=train_annotation_channel)
        elif 'segproj/usiigaci_train_data/val/' in f:
            pngpath = '/tmp/'+'usiigaci_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath)
            im1 = pngread(pngpath)
            num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
            image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
            image = img_as_ubyte(image)
            im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
            im2 = img_as_int(im)
            im3 = np.zeros([im2.shape[0],im2.shape[1]])
            im3 = im2[:,:,0]+im2[:,:,1]+im2[:,:,2]
            im3 = np.uint8((im3>0))
            im3 =  cv2.resize(im3, (1024,1024), interpolation = cv2.INTER_AREA)                 
            pngsave(pngpath,im3, check_contrast=False)
            sess.upload_data(path=pngpath, bucket=bucket, key_prefix=validation_annotation_channel)

def proccessliorfiles(f, s3 = s3, sess = sess, bucket = bucket, prefix = prefix):
    train_channel = prefix + '/train'
    validation_channel = prefix + '/validation'
    train_annotation_channel = prefix + '/train_annotation'
    validation_annotation_channel = prefix + '/validation_annotation'
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    if 'tif' in f:
        file = f
        file = file.replace('._','')
        if 'segproj/liorp_training_data/train/' in f:
            jpgpath = '/tmp/'+'liorp_raw_'+file.split('/')[-2]+'_'+'raw.jpg'
            s3.meta.client.download_file('meadata', file, jpgpath.replace('jpg','tif'))
            image = imread(jpgpath.replace('jpg','tif'))
            inverted_img =  cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA)
            num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
            image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**num - 1))
            imsave(jpgpath,img_as_ubyte(image))
            sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=train_channel)
        elif 'segproj/liorp_training_data/val/' in f:
            jpgpath = '/tmp/'+'liorp_raw_'+file.split('/')[-2]+'_'+'raw.jpg'
            s3.meta.client.download_file('meadata', file, jpgpath.replace('jpg','tif'))
            image = imread(jpgpath.replace('jpg','tif'))
            inverted_img =  cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA)
            num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
            image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**num - 1))
            imsave(jpgpath,img_as_ubyte(image))
            sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=validation_channel)
    elif 'instances_ids.png' in f:
        file = f
        file = file.replace('._','')
        if 'segproj/liorp_training_data/train/' in f:
            pngpath = '/tmp/'+'liorp_raw_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath)
            im1 = pngread(pngpath)
            num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
            image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
            image = img_as_ubyte(image)
            im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
            im2 = img_as_int(im)
            im3 = np.zeros([im2.shape[0],im2.shape[1]])
            im3 = im2[:,:,0]+im2[:,:,1]+im2[:,:,2]
            im3 = np.uint8((im3>0))
            im3 =  cv2.resize(im3, (1024,1024), interpolation = cv2.INTER_AREA)                 
            pngsave(pngpath,im3, check_contrast=False)
            sess.upload_data(path=pngpath, bucket=bucket, key_prefix=train_annotation_channel)
        elif 'segproj/liorp_training_data/val/' in f:
            pngpath = '/tmp/'+'liorp_raw_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath)
            im1 = pngread(pngpath)
            num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
            image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
            image = img_as_ubyte(image)
            im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
            im2 = img_as_int(im)
            im3 = np.zeros([im2.shape[0],im2.shape[1]])
            im3 = im2[:,:,0]+im2[:,:,1]+im2[:,:,2]
            im3 = np.uint8((im3>0))
            im3 =  cv2.resize(im3, (1024,1024), interpolation = cv2.INTER_AREA)                 
            pngsave(pngpath,im3, check_contrast=False)
            sess.upload_data(path=pngpath, bucket=bucket, key_prefix=validation_annotation_channel)

def proccessfig8preprocfiles(f, s3 = s3, sess = sess, bucket = bucket, prefix = prefix):
    train_channel = prefix + '/train'
    validation_channel = prefix + '/validation'
    train_annotation_channel = prefix + '/train_annotation'
    validation_annotation_channel = prefix + '/validation_annotation'
    if 'tif' in f:
        file = f
        if 'segproj/fig8_preprocessed/train/' in f:
            pngpath = '/tmp/'+'fig8_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath.replace('png','tif'))
            inverted_img = imread(pngpath.replace('png','tif'))
            num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
            image = exposure.rescale_intensity(inverted_img, out_range=(0,2**8 - 1))
            cv2.imwrite(pngpath.replace('png','jpg'),img_as_ubyte(image),[int(cv2.IMWRITE_JPEG_QUALITY), 100])
            sess.upload_data(path=pngpath.replace('png','jpg'), bucket=bucket, key_prefix=train_channel)
        elif 'segproj/fig8_preprocessed/val/' in f:
            pngpath = '/tmp/'+'fig8_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath.replace('png','tif'))
            inverted_img = imread(pngpath.replace('png','tif'))
            num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
            image = exposure.rescale_intensity(inverted_img, out_range=(0,2**8 - 1))
            cv2.imwrite(pngpath.replace('png','jpg'),img_as_ubyte(image),[int(cv2.IMWRITE_JPEG_QUALITY), 100])
            sess.upload_data(path=pngpath.replace('png','jpg'), bucket=bucket, key_prefix=validation_channel)
    elif 'instances_ids.png' in f:
        file = f
        if 'segproj/fig8_preprocessed/train/' in f:
            pngpath = '/tmp/'+'fig8_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath)
            im1 = imread(pngpath)
            num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
            image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
            image = img_as_ubyte(image)
            im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
            im2 = img_as_int(im)
            im3 = np.zeros([im2.shape[0],im2.shape[1]])
            im3 = im2[:,:,0]+im2[:,:,1]+im2[:,:,2]
            im3 = np.uint8((im3>0))
            im3 =  cv2.resize(im3, (1024,1024), interpolation = cv2.INTER_AREA)                 
            cv2.imwrite(pngpath,im3,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])          
            sess.upload_data(path=pngpath, bucket=bucket, key_prefix=train_annotation_channel)
        elif 'segproj/fig8_preprocessed/val/' in f:
            pngpath = '/tmp/'+'fig8_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath)
            im1 = imread(pngpath)
            num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
            image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
            image = img_as_ubyte(image)
            im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
            im2 = img_as_int(im)
            im3 = np.zeros([im2.shape[0],im2.shape[1]])
            im3 = im2[:,:,0]+im2[:,:,1]+im2[:,:,2]
            im3 = np.uint8((im3>0))
            im3 =  cv2.resize(im3, (1024,1024), interpolation = cv2.INTER_AREA)                 
            cv2.imwrite(pngpath,im3,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])            
            sess.upload_data(path=pngpath, bucket=bucket, key_prefix=validation_annotation_channel)

def proccessliorpreprocfiles(f, s3 = s3, sess = sess, bucket = bucket, prefix = prefix):
    train_channel = prefix + '/train'
    validation_channel = prefix + '/validation'
    train_annotation_channel = prefix + '/train_annotation'
    validation_annotation_channel = prefix + '/validation_annotation'
    if 'tif' in f:
        file = f
        file = file.replace('._','')
        if 'segproj/liorp_preprocessed/train/' in f:
            pngpath = '/tmp/'+'liorp_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath.replace('png','tif'))
            image = imread(pngpath.replace('png','tif'))
            inverted_img =  cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA)
            num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
            image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**8 - 1))
            cv2.imwrite(pngpath.replace('png','jpg'),img_as_ubyte(image),[int(cv2.IMWRITE_JPEG_QUALITY), 100])
            sess.upload_data(path=pngpath.replace('png','jpg'), bucket=bucket, key_prefix=train_channel)
        elif 'segproj/liorp_preprocessed/val/' in f:
            pngpath = '/tmp/'+'liorp_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath.replace('png','tif'))
            image = imread(pngpath.replace('png','tif'))
            inverted_img =  cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA)
            num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
            image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**8 - 1))
            cv2.imwrite(pngpath.replace('png','jpg'),img_as_ubyte(image),[int(cv2.IMWRITE_JPEG_QUALITY), 100])
            sess.upload_data(path=pngpath.replace('png','jpg'), bucket=bucket, key_prefix=validation_channel)
    elif 'instances_ids.png' in f:
        file = f
        file = file.replace('._','')
        if 'segproj/liorp_preprocessed/train/' in f:
            pngpath = '/tmp/'+'liorp_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath)
            im1 = imread(pngpath)
            num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
            image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
            image = img_as_ubyte(image)
            im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
            im2 = img_as_int(im)
            im3 = np.zeros([im2.shape[0],im2.shape[1]])
            im3 = im2[:,:,0]+im2[:,:,1]+im2[:,:,2]
            im3 = np.uint8((im3>0))
            im3 =  cv2.resize(im3, (1024,1024), interpolation = cv2.INTER_AREA)                 
            cv2.imwrite(pngpath,im3,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            sess.upload_data(path=pngpath, bucket=bucket, key_prefix=train_annotation_channel)
        elif 'segproj/liorp_preprocessed/val/' in f:
            pngpath = '/tmp/'+'liorp_'+file.split('/')[-2]+'_'+'raw.png'
            s3.meta.client.download_file('meadata', file, pngpath)
            im1 = imread(pngpath)
            num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
            image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
            image = img_as_ubyte(image)
            im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
            im2 = img_as_int(im)
            im3 = np.zeros([im2.shape[0],im2.shape[1]])
            im3 = im2[:,:,0]+im2[:,:,1]+im2[:,:,2]
            im3 = np.uint8((im3>0))
            im3 =  cv2.resize(im3, (1024,1024), interpolation = cv2.INTER_AREA)                 
            cv2.imwrite(pngpath,im3,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            sess.upload_data(path=pngpath, bucket=bucket, key_prefix=validation_annotation_channel)


def cropimage(image, best_box = None):
    try:
        imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    except:
        print(image.shape)
        imgray = image
        pass
    
    if best_box is None:
        ret,thresh = cv2.threshold(np.uint8(imgray>0),0,255,cv2.THRESH_BINARY_INV)
        dilated=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
        contours,_ = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        new_contours=[]
        for c in contours:
            if cv2.contourArea(c)<256*256:
                new_contours.append(c)
        best_box=[-1,-1,-1,-1]
        for c in new_contours:
            x,y,w,h = cv2.boundingRect(c)
            if best_box[0] < 0:
                best_box=[x,y,x+w,y+h]
            else:
                if x<best_box[0]:
                    best_box[0]=x
                if y<best_box[1]:
                    best_box[1]=y
                if x+w>best_box[2]:
                    best_box[2]=x+w
                if y+h>best_box[3]:
                    best_box[3]=y+h
    if (np.abs(best_box[2]-best_box[0])>100)&(np.abs(best_box[3]-best_box[1])>100):
        roi = imgray[best_box[1]:best_box[3], best_box[0]:best_box[2]]
        roi = roi[0:np.min(roi.shape), 0:np.min(roi.shape)]
    else:
        print(best_box)
        roi = imgray
    delta_w = 1024 - roi.shape[1]
    delta_h = 1024 - roi.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)
    return(roi, best_box)

def performcrop(key, bucket  = bucket, prefix = prefix ):
    train_channel = prefix + '/train'
    validation_channel = prefix + '/validation'
    train_annotation_channel = prefix + '/train_annotation'
    validation_annotation_channel = prefix + '/validation_annotation'
    imsavepath = "/tmp/"+key.split('/')[-1]
    maskkey = key.replace('train/',"train_annotation/").replace('val/','val_annotation/').replace('jpg','png')
    masksavepath = "/tmp/"+maskkey.split('/')[-1]
    try:
        s3.meta.client.download_file(bucket, key, imsavepath)
        s3.meta.client.download_file(bucket, maskkey , masksavepath)
    except:
        try:
            s3.meta.client.delete_object(Bucket = bucket, Key =  maskkey)
            s3.meta.client.delete_object(Bucket = bucket, Key =  key)
        except:
            pass
        pass
        
    if os.stat(imsavepath).st_size > 2000:
        mask,box = cropimage(cv2.imread(masksavepath))
        im,_ = cropimage(cv2.imread(imsavepath), best_box = box)
        cv2.imwrite(imsavepath,im)
        cv2.imwrite(masksavepath,mask)
        if 'train/' in key:
            sess.upload_data(path=imsavepath, bucket=bucket, key_prefix=train_channel)
            sess.upload_data(path=masksavepath, bucket=bucket, key_prefix=train_annotation_channel)
        elif 'validation/' in key:
            sess.upload_data(path=imsavepath, bucket=bucket, key_prefix=validation_channel)
            sess.upload_data(path=masksavepath, bucket=bucket, key_prefix=validation_annotation_channel)

def removeunmatched(s3_resource = s3_resource, prefix = prefix, bucket = bucket):
    keys = [obj.key for obj in s3_resource.Bucket(name=bucket).objects.all()]
    imkeys = set([str(key).split('/')[-1] for key in keys if ('jpg' in key and prefix in key)])
    maskkeys = set([str(key).split('/')[-1].replace('png','jpg') for key in keys if ('png' in key and prefix in key)])
    todelete = [l.replace('jpg','png') for l in list(maskkeys-(imkeys&maskkeys))]+list(imkeys-(imkeys&maskkeys))
    remove = [key for key in keys if any([t in key for t in todelete])]
    [boto3.client('s3').delete_object(Bucket=bucket, Key=r) for r in remove]
    
    
def merge_multiple_detections(masks):
    """

    :param masks:
    :return:
    """
    IOU_THRESHOLD = 0.6
    OVERLAP_THRESHOLD = 0.8
    MIN_DETECTIONS = 1
    def compute_iou(mask1, mask2):
        """
        Computes Intersection over Union score for two binary masks.
        :param mask1: numpy array
        :param mask2: numpy array
        :return:
        """
        intersection = np.sum((mask1 + mask2) > 1)
        union = np.sum((mask1 + mask2) > 0)

        return intersection / float(union)

    def compute_overlap(mask1, mask2):
        intersection = np.sum((mask1 + mask2) > 1)

        overlap1 = intersection / float(np.sum(mask1))
        overlap2 = intersection / float(np.sum(mask2))
        return overlap1, overlap2

    def sort_mask_by_cells(mask, min_size=50):
        """
        Returns size of each cell.
        :param mask:
        :return:
        """
        cell_num = np.unique(mask)
        cell_sizes = [(cell_id, len(np.where(mask == cell_id)[0]))
                      for cell_id in cell_num if cell_id != 0]

        cell_sizes = [x for x in sorted(
            cell_sizes, key=lambda x: x[1], reverse=True) if x[1 > min_size]]

        return cell_sizes
    
    cell_counter = 0
    final_mask = np.zeros(masks[0].shape)

    masks_stats = [sort_mask_by_cells(mask) for mask in masks]
    cells_left = sum([len(stats) for stats in masks_stats])

    while cells_left > 0:
        # Choose the biggest cell from available
        cells = [stats[0][1] if len(
            stats) > 0 else 0 for stats in masks_stats]
        reference_mask = cells.index(max(cells))

        reference_cell = masks_stats[reference_mask].pop(0)[0]

        # Prepare binary mask for cell chosen for comparison
        cell_location = np.where(masks[reference_mask] == reference_cell)

        cell_mask = np.zeros(final_mask.shape)
        cell_mask[cell_location] = 1

        masks[reference_mask][cell_location] = 0

        # Mask for storing temporary results
        tmp_mask = np.zeros(final_mask.shape)
        tmp_mask += cell_mask

        for mask_id, mask in enumerate(masks):
            # For each mask left
            if mask_id != reference_mask:
                # # Find overlapping cells on other masks
                overlapping_cells = list(np.unique(mask[cell_location]))

                try:
                    overlapping_cells.remove(0)
                except ValueError:
                    pass

                # # If only one overlapping, check IoU and update tmp mask if high
                if len(overlapping_cells) == 1:
                    overlapping_cell_mask = np.zeros(final_mask.shape)
                    overlapping_cell_mask[np.where(
                        mask == overlapping_cells[0])] = 1

                    iou = compute_iou(cell_mask, overlapping_cell_mask)
                    if iou >= IOU_THRESHOLD:
                        # Add cell to temporary results and remove from stats and mask
                        tmp_mask += overlapping_cell_mask
                        idx = [i for i, cell in enumerate(
                            masks_stats[mask_id]) if cell[0] == overlapping_cells[0]][0]
                        masks_stats[mask_id].pop(idx)
                        mask[np.where(mask == overlapping_cells[0])] = 0

                # # If more than one overlapping check area overlapping
                elif len(overlapping_cells) > 1:
                    overlapping_cell_masks = [
                        np.zeros(final_mask.shape) for _ in overlapping_cells]

                    for i, cell_id in enumerate(overlapping_cells):
                        overlapping_cell_masks[i][np.where(
                            mask == cell_id)] = 1

                    for cell_id, overlap_mask in zip(overlapping_cells, overlapping_cell_masks):
                        overlap_score, _ = compute_overlap(
                            overlap_mask, cell_mask)

                        if overlap_score >= OVERLAP_THRESHOLD:
                            tmp_mask += overlap_mask

                            mask[np.where(mask == cell_id)] = 0
                            idx = [i for i, cell in enumerate(masks_stats[mask_id])
                                   if cell[0] == cell_id][0]
                            masks_stats[mask_id].pop(idx)

                # # If none overlapping do nothing

        if len(np.unique(tmp_mask)) > 1:
            cell_counter += 1
            final_mask[np.where(tmp_mask >= MIN_DETECTIONS)] = cell_counter

        cells_left = sum([len(stats) for stats in masks_stats])

    bin_mask = np.zeros(final_mask.shape)
    bin_mask[np.where(final_mask > 0)] = 255
    return(final_mask)

def merge_two_masks(maskpaths):
    masks = []
    for mpath in maskpaths:
        binarymask = pngread(mpath)
        num_classes = 2
        distance = ndi.distance_transform_edt(binarymask)
        local_maxi = peak_local_max(distance, labels=binarymask, footprint=np.ones((3, 3)), indices=False)
        markers = ndi.label(local_maxi)[0]
        masks.append(watershed(-distance, markers, mask=binarymask))
    mask = merge_multiple_detections(masks)
    return(mask)

def merge_masks(s3_object, model_ids,batchid = ''):
    os.makedirs('/tmp/results/merge/merged/', exist_ok=True)
    outpaths=[]
    ismodelresult = any([m for m in model_ids if m in s3_object])
    if ismodelresult:
        for model_id in model_ids:
            if not s3_object.endswith("/") and "masks" in s3_object:
                    outpath = os.path.join('/tmp/results/'+batchid+'/merge/',model_id+'_'+s3_object.split('/')[-1])
                    s3.meta.client.download_file('sagemaker-eu-west-1-102554356212', s3_object, outpath)
                    outpaths.append(outpath)
        if outpaths:            
            mask = merge_two_masks(outpaths)
            pngsave(os.path.join('/tmp/results/'+batchid+'/merge/merged/','merged_'+s3_object.split('/')[-1]), np.uint8(mask>0))
