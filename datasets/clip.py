#!/usr/bin/python
# encoding: utf-8

##### THIS CODE WILL NOT BE CALLED FOR TESTING 
##### COMMENT TEST LOADER WHILE TRAINING 

import random
import os
import torch
from PIL import Image
import numpy as np
from core.utils import *
import cv2
from pathlib import Path 
import glob 
from skvideo.io import vread

def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, dhue, dsat, dexp):
    
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(clip, shape, jitter, hue, saturation, exposure):
    # Initialize Random Variables
    oh = clip[0].height  
    ow = clip[0].width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    flip = random.randint(1,10000)%2

    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)

    # Augment
    cropped = [img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1)) for img in clip]

    sized = [img.resize(shape) for img in cropped]

    if flip: 
        sized = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in sized]

    clip = [random_distort_image(img, dhue, dsat, dexp) for img in sized]
    
    return clip, flip, dx, dy, sx, sy 

# this function works for obtaining new labels after data augumentation
def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes,5))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        # print("truth detectopm",bs)
        # exit(1)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))

        for i in range(bs.shape[0]):
            cx = (bs[i][1] + bs[i][3]) / (2 * 320)
            cy = (bs[i][2] + bs[i][4]) / (2 * 240)
            imgw = (bs[i][3] - bs[i][1]) / 320
            imgh = (bs[i][4] - bs[i][2]) / 240
            bs[i][0] = bs[i][0] - 1
            bs[i][1] = cx
            bs[i][2] = cy
            bs[i][3] = imgw
            bs[i][4] = imgh

        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            
            bs[i][1] = (x1 + x2)/2
            bs[i][2] = (y1 + y2)/2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    # print("label shape", label.shape,np.max(label))
    return label

#j is the json containin the v_ids and the paths 
def load_data_detection(cfg, j,base_path, imgpath, train, train_dur, sampling_rate, shape, dataset_use='ucf24', jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
    # clip loading and  data augmentation

    im_split = imgpath.split('/')
    num_parts = len(im_split)
    im_ind = int(im_split[num_parts-1][0:5])
    labpath = os.path.join(base_path, 'labels', im_split[0], im_split[1] ,'{:05d}.txt'.format(im_ind))

    img_folder = os.path.join(base_path, 'rgb-images', im_split[0], im_split[1])
    
    # print("base path", "image_path",base_path,imgpath)
    # exit(1)
    if dataset_use == 'ucf24':
        max_num = len(os.listdir(img_folder))
    elif dataset_use == 'jhmdb21':
        max_num = len(os.listdir(img_folder)) - 1

    clip = []

    ### We change downsampling rate throughout training as a       ###
    ### temporal augmentation, which brings around 1-2 frame       ###
    ### mAP. During test time it is set to cfg.DATA.SAMPLING_RATE. ###
    d = sampling_rate
    if train:
        d = random.randint(1, 2)

    f_ids = []
    for i in reversed(range(train_dur)):
        # make it as a loop
        i_temp = im_ind - i * d
        if i_temp < 1:
            i_temp = 1
        elif i_temp > max_num:
            i_temp = max_num


        # print("i temp", i_temp,)
        if dataset_use == 'ucf24':
            path_tmp = os.path.join(base_path, 'rgb-images', im_split[0], im_split[1] ,'{:05d}.jpg'.format(i_temp))
        elif dataset_use == 'jhmdb21':
            path_tmp = os.path.join(base_path, 'rgb-images', im_split[0], im_split[1] ,'{:05d}.png'.format(i_temp))
        
        cls, v_name, f_id = path_tmp.split('/')[-3:]
        
        v_id = [cls, v_name]
        v_id = '/'.join(v_id)
        f_id = int(f_id.split('.')[0])-1
        f_ids.append(f_id)
    # print("path_tmp", path_tmp)
    
    occ_data_dir = cfg.TRAIN.OCCLUSION_DATA_DIR
    distribution = cfg.TRAIN.DISTRIBUTION_DIR
    # print(j.keys())
    # print("f_ids is", f_ids)
    # print("v_name", v_name)
    
    # video_id = video_name.split('/')[-1]
    paths = j[v_name]['train']
    # print("before", len(paths))
    if distribution=='all':
        pass
    else:
        new_paths = []
        for path in paths:
            if distribution in path:
                new_paths.append(path)
        paths = new_paths 
    # print("after filter", len(paths))           
    path = random.choice(paths)
    path = path.split('/')[3:]

    path = occ_data_dir +'/'+ '/'.join(path)
    
    # print("occ data dir", path)
    video = vread(str(path))
    # print("video shape", video.shape,f_ids)
    for f_id in f_ids:
        clip.append(Image.fromarray(video[f_id]))
    # exit(1)
    # clip.append(Image.open(path_tmp).convert('RGB'))

    if train: # Apply augmentation
        
        clip,flip,dx,dy,sx,sy = data_augmentation(clip, shape, jitter, hue, saturation, exposure)
        label = fill_truth_detection(labpath, clip[0].width, clip[0].height, flip, dx, dy, 1./sx, 1./sy)
        label = torch.from_numpy(label)

        pass

    else: # No augmentation
        label = torch.zeros(50*5)
        try:
            tmp = torch.from_numpy(read_truths_args(labpath, 8.0/clip[0].width).astype('float32'))
        except Exception:
            tmp = torch.zeros(1,5)

        tmp = tmp.view(-1)
        tsz = tmp.numel()

        if tsz > 50*5:
            label = tmp[0:50*5]
        elif tsz > 0:
            label[0:tsz] = tmp

    # print("in dataloader", type(clip[0]), type(label), label.shape)
    if train:
        return clip, label
    else:
        return im_split[0] + '_' +im_split[1] + '_' + im_split[2], clip, label

def load_data_detection_test(root, imgpath, train_dur, num_samples):

    clip,label = get_clip(root, imgpath, train_dur, num_samples)

    return clip, label

def get_clip(root, imgpath, train_dur, num_samples):

    im_split = imgpath.split('/')
    num_parts = len(im_split)
    im_ind = int(im_split[num_parts - 1][0:5])

    # for UCF101 dataset
    base_path = "/usr/home/sut/datasets/ucf24"
    labpath = os.path.join(base_path, 'labels', im_split[6], im_split[7], '{:05d}.txt'.format(im_ind))
    img_folder = os.path.join(base_path, 'rgb-images', im_split[6], im_split[7])

    # for arbitrary videos
    max_num = len(os.listdir(img_folder))

    clip = []
    for i in reversed(range(train_dur)):
        # the clip is created with the trained sample(image) being placed as the last image and 7 adjacent images before it
        i_temp = im_ind - i
        if i_temp < 1:
            i_temp = 1
        if i_temp > max_num:
            i_temp = max_num

        path_tmp = os.path.join(base_path, 'rgb-images', im_split[6], im_split[7] ,'{:05d}.jpg'.format(i_temp))
        
        clip.append(Image.open(path_tmp).convert('RGB'))

    label = torch.zeros(50 * 5)
    tmp = torch.zeros(1, 5)
    tmp = tmp.view(-1)
    tsz = tmp.numel()

    if tsz > 50 * 5:
        label = tmp[0:50 * 5]
    elif tsz > 0:
        label[0:tsz] = tmp

    return clip, label
