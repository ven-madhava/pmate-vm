# 4 Oct, 2018
# For VM pmate-beta

# Imports
# -------

import numpy as np
import cv2
import random
import copy
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tempfile
from tempfile import TemporaryFile
from google.cloud import storage
from tensorflow.python.lib.io import file_io
from io import BytesIO
import threading
import time

# Necessary Flask imports
# -----------------------
from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from json import dumps
from flask_jsonpify import jsonify


# # GCS functions
# Getting images from a "folder" in storage and returning that as a numpy array
# API ready ##
# -----------------------------------------------------------------------------
def get_images_from_storage(parent_dir,output_mode):

    """"For example, given these blobs:
        /a/1.txt
        /a/b/2.txt
    If you just specify prefix = '/a', you'll get back:
        /a/1.txt
        /a/b/2.txt
    However, if you specify prefix='/a' and delimiter='/', you'll get back:
        /a/1.txt"""

    # Create a storage client to use with bucket
    # ------------------------------------------
    bucket_name = 'ven-ml-project.appspot.com'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    # Getting all bobs under the mentioned "folder"
    # --------------------------------------------
    blobs = bucket.list_blobs(prefix=parent_dir + '/', delimiter='/')

    # Itering through blobs
    # ---------------------
    counter = 0
    if output_mode == 'list':
        xout = []

    for b in blobs:

        if '.jpg' in str(b.name) or '.jpeg' in str(b.name):

            counter += 1
            blob_curr = bucket.blob(str(b.name))

            # Using tempfile to retrieve
            # --------------------------
            with tempfile.NamedTemporaryFile() as temp:
                blob_curr.download_to_filename(temp.name)
                img = cv2.imread(temp.name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if output_mode == 'list':
                    xout.append(img)
                else:
                    img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
                    if counter == 1:
                        xout = img
                    else:
                        xout = np.concatenate((xout,img), axis = 0)

    return xout

# Function to saving a list or numpy array of images to storage folder
# API ready ##
# --------------------------------------------------------------------
def save_to_storage_from_array_list(x,storage_dir,image_prefix,update_progress,progress):

    # Create a storage client to use with bucket
    # ------------------------------------------
    bucket_name = 'ven-ml-project.appspot.com'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    xin = copy.deepcopy(x)
    xin = xin.astype('uint8')
    m = len(xin)

    # Itering through the list / array and saving them to storage
    # -----------------------------------------------------------
    for i in range(len(xin)):

        img = xin[i] # Works for both numpy array and list

        # Using temp file for storage ops
        # -------------------------------
        with tempfile.NamedTemporaryFile() as temp:

            # Etract name to the temp file
            # -----------------------------
            image_name = ''.join([str(temp.name),'.jpg'])

            # Save image to temp file
            # -----------------------
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_name,img.astype('uint8'))

            # Storing the image temp file inside the bucket
            # ---------------------------------------------
            destination_blob_name = storage_dir + '/' + image_prefix + '_' + str(i) + '.jpg'
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(image_name,content_type='image/jpeg')

        if update_progress == True:
            curr_prog_percent = int(round((i+1)/m,2)*100)
            progress['curr_message'] = str(progress['master_message']) + '..about ' + str(curr_prog_percent) + '% through'


        print('Done saving image ' + str(i) + '..')

# Getting images from a "folder" in storage and returning that as a numpy array
# API ready ##
# -----------------------------------------------------------------------------
def get_images_from_storage_by_names(parent_dir,output_mode,in_names):

    """"For example, given these blobs:
        /a/1.txt
        /a/b/2.txt
    If you just specify prefix = '/a', you'll get back:
        /a/1.txt
        /a/b/2.txt
    However, if you specify prefix='/a' and delimiter='/', you'll get back:
        /a/1.txt"""

    # Converting innames to string
    # ----------------------------
    in_names_list = in_names.split(',')

    # Create a storage client to use with bucket
    # ------------------------------------------
    bucket_name = 'ven-ml-project.appspot.com'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    # Itering through names
    # ---------------------
    counter = 0
    if output_mode == 'list':
        xout = []


    for i in in_names_list:

        # Setting blob
        # ------------
        blob_name = parent_dir + '/' + str(i)
        blob = bucket.blob(blob_name)

        curr_img = None
        curr_img_flag = 0

        try:
            with tempfile.NamedTemporaryFile() as temp:
                blob.download_to_filename(temp.name)
                img = cv2.imread(temp.name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if output_mode != 'list':
                    img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
                curr_img = copy.deepcopy(img)
                curr_img_flag = 1
                counter += 1
        except:

            'image not found'

        # Concatenating to output if image found
        # ---------------------------------------
        if curr_img_flag == 1:
            if output_mode == 'list':
                xout.append(curr_img)
            else:
                if counter == 1:
                    xout = curr_img
                else:
                    xout = np.concatenate((xout,curr_img), axis = 0)
    return xout


# # Protomate supportive functions
# protomate function to get block images
# API ready ##
# --------------------------------------
def protomatebeta_getfillimage_v1(datavec,labels,main_image,k,mode):

    # Getting the pixel locations for these labels
    # --------------------------------------------
    loc_list = list(np.argwhere(labels == k)[:,0])

    # Initialisations
    # ---------------
    main_image = np.array(main_image)
    h_indices = []
    w_indices = []

    # Getting h,w from datavec directly
    # ---------------------------------
    h_w = datavec[loc_list][:,3:5]


    # Looping through to create image
    # -------------------------------
    newim = np.ones((main_image.shape)) * 255
    newim_map = np.ones((main_image.shape[0],main_image.shape[1],1)) * 255
    for i in range(h_w.shape[0]):
        curr_w = int(h_w[i,0])
        curr_h = int(h_w[i,1])
        newim[curr_h,curr_w,:] = main_image[curr_h,curr_w,:]
        newim_map[curr_h,curr_w] = 0
        h_indices.append(curr_h)
        w_indices.append(curr_w)


    return newim.astype('uint8'), h_indices, w_indices, newim_map

# protomate kmeans function
# API ready ##
# -------------------------
def protomatebeta_cvkmeans_v1(imn,K,iters,mode,centers):

    # 1. Initialisations
    # ------------------
    imageH = imn.shape[0]
    imageW = imn.shape[1]
    dataVector = np.ndarray(shape=(imageH * imageW, 5), dtype=float) # r,g,b,x,y
    imin = Image.fromarray(imn)

    # 2. Building datavec for localisations
    # -------------------------------------
    for y in range(imageH):
        for x in range(imageW):
            xy = (x,y)
            rgb = imin.getpixel(xy)
            dataVector[x + y * imageW, 0] = rgb[0]
            dataVector[x + y * imageW, 1] = rgb[1]
            dataVector[x + y * imageW, 2] = rgb[2]
            dataVector[x + y * imageW, 3] = x #
            dataVector[x + y * imageW, 4] = y #

    # 3. k means locations settings
    # ------------------------------
    if mode == 'datavec':
        kin = dataVector
    else:
        kin = imn.reshape((-1,3))

    # 4. Running kmeans
    # -----------------
    Z = np.float32(kin) # Convert to np.float32

    # Define criteria, number of clusters(K) and apply kmeans()
    ##
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1.0)
    if centers == 'random':
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS) # KMEANS_RANDOM_CENTERS, KMEANS_PP_CENTERS
    else:
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS) # KMEANS_RANDOM_CENTERS, KMEANS_PP_CENTERS


    # 5. Setting up for output
    # ------------------------
    center = np.uint8(center)
    res = center[label.flatten()]
    resim = res[:,0:3]
    res2 = resim.reshape((imn.shape))

    return dataVector,label,res2,center

# protomate recurring kmeans function
# API ready ##
# -----------------------------------
def protomatebeta_recurr_kmeans_v1(img,start_k,end_k,cluster_by_location):

    # Initialisations
    # ---------------
    curr_im = copy.deepcopy(img)
    iter_flag = end_k - 1

    # Actual iter
    # -----------
    for i in range(start_k - iter_flag):
        curr_k = start_k - i
        if cluster_by_location == True:
            datavec,labels,curr_im,cen = protomatebeta_cvkmeans_v1(curr_im,curr_k,1500,'datavec','pp')  # For mood boards use datavec (localised), for product images use rdatavec(centralised)
        else:
            datavec,labels,curr_im,cen = protomatebeta_cvkmeans_v1(curr_im,curr_k,1500,'rdatavec','pp')  # For mood boards use datavec (localised), for product images use rdatavec(centralised)

    return curr_im,cen,datavec,labels


# # Protomate main functions
# 1
# API ready ##
# Stitch images together for blocks extraction
# --------------------------------------------
def protomatebeta_stitch_incoming_images_v1(inlist):

    # Some initialisations
    # --------------------
    insidecounter = 0
    xcurr = None
    xout = []
    new_h = 350
    thresh_max_w = 1000
    thresh_min_w = 400

    for i in range(len(inlist)):

        insidecounter += 1
        img = inlist[i]
        img_h = img.shape[0]
        img_w = img.shape[1]
        w_factor = img_w/img_h
        new_w = int(w_factor*new_h)

        # Resizing images to standard height of 500 px
        # --------------------------------------------
        img_res = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Concat ops
        # ----------
        if insidecounter == 1:
            xcurr = img_res
        else:
            # Checking threshold condition
            # ----------------------------
            if xcurr.shape[1] + new_w > thresh_max_w:

                # Appending stitched images to a list
                # -----------------------------------
                xout.append(xcurr)

                # Resetting xcurr and inside counter
                # ----------------------------------
                xcurr = img_res
                insidecounter = 2

            else:
                xcurr = np.concatenate((xcurr,img_res), axis = 1)

    # Final xcurr concatenation to xout
    # ---------------------------------
    try:
        if xcurr.shape[1] < thresh_min_w:

            # Need to concat the last bit image to the last xout image
            # --------------------------------------------------------
            xout_lastim = xout[len(xout)-1]
            xout_lastim = np.concatenate((xout_lastim,xcurr), axis = 1)
            xout[len(xout)-1] = xout_lastim

        else:
            xout.append(xcurr)
    except:
        'do nothing'


    return xout

# 2
# API ready ##
# Extracts blocks for building patterns
# ------------------------------------
def protomatebeta_extract_blocks_for_aop_v1(inlist,progress):

    # Early initialisations
    # ---------------------
    fullon_blocks_main = []
    fullon_blocks_map = []
    start_k_c = 150
    end_k_c = 150
    source_image = 'orig'
    localisation_factor = True


    # Initialisation
    # --------------
    counter = 0

    for i in range(len(inlist)):

        counter += 1
        img = inlist[i]
        orig_img = copy.deepcopy(img)
        print('At image ' + str(counter) + ' of around ' + str(len(inlist)) + '..')


        # 1. Smoothening images
        # ---------------------
        print('1. Smoothening image..')
        for _ in range(1): # was 5
            for _ in range(1):
                img = cv2.medianBlur(img, 5)
            for _ in range(5):
                img = cv2.edgePreservingFilter(img, flags=2, sigma_s=100, sigma_r=0.25)

        # 2. k means
        # ----------
        print('2. Applying recurring kmeans to segemnt..')
        kmimg,cen,dv,labels = protomatebeta_recurr_kmeans_v1(img,start_k_c,end_k_c,localisation_factor)
        plt.imshow(kmimg)
        plt.show()

        # Updating progress
        # -----------------
        curr_prog_percent = int(round((i+1)/(len(inlist)),2)*100)
        progress['curr_message'] = str(progress['master_message']) + '..about ' + str(curr_prog_percent) + '% through'

        # 3. Getting pattern blocks from segmented image as square blocks
        # ---------------------------------------------------------------
        print('3. Getting pattern blocks based on segmented image..')
        pt_blocks,pt_sqr_map,fullon_blocks,fullon_map = protomatebeta_cutout_blocks_v1(dv,labels,orig_img,cen,source_image) # can change orig to kmimg for placement patterns
        fullon_blocks_main  = fullon_blocks_main + fullon_blocks
        fullon_blocks_map = fullon_blocks_map + fullon_map

        if counter == 1:
            xout = pt_blocks
            pt_blocks_sqr_map = pt_sqr_map
        else:
            xout = np.concatenate((xout,pt_blocks), axis = 0)
            pt_blocks_sqr_map = np.concatenate((pt_blocks_sqr_map,pt_sqr_map), axis = 0)



    return fullon_blocks_main,fullon_blocks_map

# 2.1
# API ready ##
# Function to cut out block images from clustered mood board image for block extraction
# -------------------------------------------------------------------------------------
def protomatebeta_cutout_blocks_v1(datavec,labels,image,cen,image_mode):

    # Initialistations
    # ----------------
    no_patterns = cen.shape[0]
    pad_size = 0
    full_on_blocks = []
    full_on_blocks_map = []

    for i in range(no_patterns):

        nm,ht,wd,nm_map = protomatebeta_getfillimage_v1(datavec,labels,image,i,None)
        hh = np.max(ht) - np.min(ht)
        ww = np.max(wd) - np.min(wd)
        length = min(hh,ww)
        end_h = np.min(ht) + length
        end_w = np.min(wd) + length

        # Settings co-ordinates for full size blocks
        # -----------------------------------------
        start_h_full_on = np.min(ht)
        end_h_full_on = np.max(ht)
        start_w_full_on = np.min(wd)
        end_w_full_on = np.max(wd)


        if image_mode == 'orig':

            curr_full_on_block = image[start_h_full_on:end_h_full_on,start_w_full_on:end_w_full_on,:]
            try:
                curr_block = image[np.min(ht):end_h+pad_size,np.min(wd):end_w+pad_size,:]
                curr_block_sqr_map = nm_map[np.min(ht):end_h+pad_size,np.min(wd):end_w+pad_size] # Getting map of block
            except:
                curr_block = image[np.min(ht):end_h,np.min(wd):end_w,:]
                curr_block_sqr_map = nm_map[np.min(ht):end_h,np.min(wd):end_w] # Getting map of block

        else:

            curr_full_on_block = nm[start_h_full_on:end_h_full_on,start_w_full_on:end_w_full_on,:]
            try:
                curr_block = nm[np.min(ht)-pad_size:end_h+pad_size,np.min(wd)-pad_size:end_w+pad_size,:]
                curr_block_sqr_map = nm_map[np.min(ht)-pad_size:end_h+pad_size,np.min(wd)-pad_size:end_w+pad_size] # Getting map of block
            except:
                curr_block = nm[np.min(ht):end_h,np.min(wd):end_w,:]
                curr_block_sqr_map = nm_map[np.min(ht):end_h,np.min(wd):end_w] # Getting map of block


        try:
            curr_block_res = cv2.resize(curr_block, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
            curr_block_res = curr_block_res.reshape(1,50,50,3)

            # A small corrcetion to square block map due to resizing op
            # ---------------------------------------------------------
            curr_block_sqr_map_res = cv2.resize(curr_block_sqr_map, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
            curr_block_sqr_map_res[curr_block_sqr_map_res <= 150] = 0
            curr_block_sqr_map_res[curr_block_sqr_map_res > 150] = 255
            curr_block_sqr_map_res = curr_block_sqr_map_res.reshape(1,50,50)

            if i == 0:
                xout = curr_block_res
                xout_sqr_map = curr_block_sqr_map_res

            else:
                xout = np.concatenate((xout,curr_block_res), axis = 0)
                xout_sqr_map = np.concatenate((xout_sqr_map,curr_block_sqr_map_res), axis = 0)
        except:

            'do nothing'

        curr_full_on_block_map = nm_map[start_h_full_on:end_h_full_on,start_w_full_on:end_w_full_on]

        # Appending curr_full on block to list
        # ------------------------------------
        full_on_blocks.append(curr_full_on_block)
        full_on_blocks_map.append(curr_full_on_block_map)


    return xout,xout_sqr_map,full_on_blocks,full_on_blocks_map

# 3
# API ready ##
# Main function to build AOP patterns including height wise shift
# ---------------------------------------------------------------
def protomate_build_aop_patterns_v1(blocks,h,w,repeat_w):

    # Getting into direct iter
    # ------------------------
    counter = 0
    for i in range(len(blocks)):

        # 1. Current patch specific initializations
        # -----------------------------------------
        curr_orig_block = blocks[i]
        block_h = curr_orig_block.shape[0]
        block_w = curr_orig_block.shape[1]

        # 1.1 Doing a check to make sure the curr_block is 3 channels
        # -----------------------------------------------------------
        try:
            if curr_orig_block.shape[2] == 3:
                'do nothing'
            else:
                curr_orig_block = curr_orig_block.reshape(curr_orig_block.shape[0],curr_orig_block.shape[1],1)
                curr_orig_block = np.concatenate((curr_orig_block,curr_orig_block,curr_orig_block), axis = 2)
        except:
            curr_orig_block = curr_orig_block.reshape(curr_orig_block.shape[0],curr_orig_block.shape[1],1)
            curr_orig_block = np.concatenate((curr_orig_block,curr_orig_block,curr_orig_block), axis = 2)


        if block_h < 10 or block_w < 10:
            'do nothing'
            # For not having to worry about really thin stripes
        else:
            counter += 1
            h_w_fac = block_h/block_w
            repeat_h = int(h_w_fac * repeat_w)
            final_pat_h = 285
            final_pat_w = 221
            block_for_full_ht = final_pat_h + repeat_h

            # 2. Creating repeat block with new size
            # --------------------------------------
            curr_repeat_block = cv2.resize(curr_orig_block, dsize=(repeat_w, repeat_h), interpolation=cv2.INTER_CUBIC)

            # 3. Creating standard full repeat
            # --------------------------------
            block_for_full = curr_repeat_block.reshape(1,curr_repeat_block.shape[0],curr_repeat_block.shape[1],curr_repeat_block.shape[2])
            full_pat = protomate_build_std_aop_pattern_repeat_v1(block_for_full,block_for_full_ht,final_pat_w)

            # 4. Height shift ops
            # -------------------
            ht_fac = int(repeat_h/2)
            orig_full_block_h = full_pat[0].shape[0]
            orig_full_block_w = full_pat[0].shape[1]

            # 4.1 Creating a new image to perform height wise shift
            # ------------------------------------------------------
            rp_im_ht = (np.ones((1,orig_full_block_h + ht_fac + 1,orig_full_block_w,3))*255).astype('uint8')
            rp_im_ht[0,0:orig_full_block_h,0:orig_full_block_w,:] = full_pat[0]

            # 4.2 Actual height wise shift ops
            # --------------------------------
            no_iters = math.floor((orig_full_block_w/repeat_w)/2) + 1
            st_c = repeat_w

            st_r = ht_fac
            en_r = orig_full_block_h + ht_fac
            for niters in range(no_iters):
                if niters > 0:
                    st_c = en_c + repeat_w
                en_c = st_c + repeat_w

                try:
                    rp_im_ht[0][st_r:en_r,st_c:en_c] = rp_im_ht[0][0:orig_full_block_h,st_c:en_c]
                except:
                    rp_im_ht[0][st_r:,st_c:en_c] = rp_im_ht[0][0:orig_full_block_h,st_c:]

            # 4.3 Final smoothening
            # ---------------------
            ht_wise_final_image = rp_im_ht[0][ht_fac:ht_fac+final_pat_h,0:,:]
            for _ in range(1):
                ht_wise_final_image = cv2.medianBlur(ht_wise_final_image, 3)
            ht_wise_final_image = ht_wise_final_image.reshape(1,ht_wise_final_image.shape[0],ht_wise_final_image.shape[1],3)


            if counter == 1:
                xout = ht_wise_final_image
            else:
                xout = np.concatenate((xout,ht_wise_final_image), axis = 0)

    return xout

# 3.1
# API ready ##
# Function to make full images using extracted blocks for height wise shift
# --------------------------------------------------------------------------
def protomate_build_std_aop_pattern_repeat_v1(x,h,w):

    # Initialisations
    # ---------------
    block_h = x.shape[1]
    block_w = x.shape[2]
    no_rows = int(h/block_h) + 5
    no_cols = int(w/block_w) + 5

    for i in range(x.shape[0]):

        currim = x[i]

        # 1. Building top block row - concatenating along columns
        # -------------------------------------------------------
        for c in range(no_cols):
            if c == 0:
                colblock = currim
            else:
                colblock = np.concatenate((colblock,currim), axis = 1)

        # 2. Concatenating along rows
        # ---------------------------
        curr_im_colblock = copy.deepcopy(colblock)
        for r in range(no_rows):
            if r == 0:
                rowblock = curr_im_colblock
            elif r % 2 != 0:
                flipped_rowblock = np.fliplr(curr_im_colblock)
                rowblock = np.concatenate((rowblock,flipped_rowblock), axis = 0)
            elif r % 2 == 0:
                rowblock = np.concatenate((rowblock,curr_im_colblock), axis = 0)

        if i == 0:
            mout = rowblock.reshape(1,rowblock.shape[0],rowblock.shape[1],rowblock.shape[2])
        else:
            mout = np.concatenate((mout,rowblock.reshape(1,rowblock.shape[0],rowblock.shape[1],rowblock.shape[2])), axis = 0)

    return mout[:,0:h,0:w,:]

# 4
# API ready ##
# Main function that picks core colors from images
# ------------------------------------------------
def protomatebeta_pickcolors_v1(progress,inlist,ht,wd,similarity_distance=0.1):

    # Iterating through images
    # ------------------------
    counter = 0
    for i in range(len(inlist)):

        counter += 1
        img = inlist[i]
        orig_img = copy.deepcopy(img)
        print('At image ' + str(counter) + ' of around ' + str(len(inlist)) + '..')

        # 1. Smoothening the image
        # ------------------------
        print('1. Smoothening image..')
        for _ in range(5):
            for _ in range(10):
                img = cv2.medianBlur(img, 5)
            for _ in range(10):
                img = cv2.edgePreservingFilter(img, flags=2, sigma_s=10, sigma_r=0.10)

        # 2. Recurring k means extractin key 25 colors
        # --------------------------------------------
        print('2. Extracting key colors using kmeans..')
        kmimg_colors,cen_colrs,dv_clrs,lb_clrs = protomatebeta_recurr_kmeans_v1(img,30,25,False)


        # 3. Clustering similar colors
        # ----------------------------
        print('3. Clustering similar colors..')
        pick_color_dict = protomatebeta_cluster_colors_v1(cen_colrs,similarity_distance,False)

        # 4. Getting final colors for the image
        # -------------------------------------
        print('4. Filtering final colors..')
        fincolors = protomatebeta_getfinalcolors_v1(pick_color_dict,cen_colrs,lb_clrs,False,ht,wd)

        # Updating progress
        # -----------------
        curr_prog_percent = int(round((i+1)/(len(inlist)),2)*100)
        progress['curr_message'] = str(progress['master_message']) + '..about ' + str(curr_prog_percent) + '% through'


        if counter == 1:
            xout = fincolors
        else:
            xout = np.concatenate((xout,fincolors), axis = 0)

    return xout

# 4.1
# API ready ##
# Main function that clusters similar colors for theme images
# -----------------------------------------------------------
def protomatebeta_cluster_colors_v1(raw_colors,similarity_distance,print_colors):

    # Some initial master settings
    # ----------------------------
    total_cols = raw_colors.shape[0]
    master_all_ind = list(range(total_cols))

    # Recurrence settings - Initial one time set up
    # ---------------------------------------------
    colors_for_trav = copy.deepcopy(raw_colors)
    all_ind_for_trav = list(range(colors_for_trav.shape[0]))
    m = len(all_ind_for_trav)
    color_set_dict = {}
    set_num = 0


    # Actual iter
    # ----------
    while m > 0:

        set_num += 1
        curr_set_ind = []
        curr_color = colors_for_trav[0]
        curr_color_intensity = curr_color/255
        for i in range(m):
            trav_curr_intensity = colors_for_trav[i]/255
            diff_r = abs(curr_color_intensity[0] - trav_curr_intensity[0])
            diff_g = abs(curr_color_intensity[1] - trav_curr_intensity[1])
            diff_b = abs(curr_color_intensity[2] - trav_curr_intensity[2])

            if diff_r < similarity_distance and diff_g < similarity_distance and diff_b < similarity_distance:
                curr_set_ind.append(i)

        # Creating a new object in dictionary
        # -----------------------------------
        color_set_dict[set_num] = colors_for_trav[curr_set_ind]
        #print('Creating set ' + str(set_num) + '..')

        # Popping the indices from the traversing color set
        # -------------------------------------------------
        new_ind_to_retain_for_next_iter = [x for x in all_ind_for_trav if x not in curr_set_ind]
        colors_for_trav = copy.deepcopy(colors_for_trav[new_ind_to_retain_for_next_iter])
        all_ind_for_trav = list(range(colors_for_trav.shape[0]))
        m = len(all_ind_for_trav)
        #print('At set ' + str(set_num) + ', curr m size is ' + str(m))

    if print_colors == True:
        for keys in color_set_dict:
            print(str(keys) + '..' + str(color_set_dict[keys].shape))
            h = np.ones((25,25,3)).astype('uint8')
            h[:,:,0] = color_set_dict[keys][0][0]
            h[:,:,1] = color_set_dict[keys][0][1]
            h[:,:,2] = color_set_dict[keys][0][2]
            print(color_set_dict[keys][0])
            plt.figure(figsize=(1,1))
            plt.imshow(h)
            plt.show()

    return color_set_dict

# 4.2
# API ready ##
# Main function that takes in clustered dict to return final colors
# -----------------------------------------------------------------
def protomatebeta_getfinalcolors_v1(color_dict,cen,labels,print_colors,ht,wd):

    # Some initialisations
    # --------------------
    total_cen = cen.shape[0]
    counter = 0

    for keys in color_dict:

        # Iterating through color clusters and saving by max locs
        # -------------------------------------------------------
        counter += 1
        curr_colors = color_dict[keys]
        locs_counter = []
        m = curr_colors.shape[0]

        if m == 1:
            color_to_keep = curr_colors[0]
            #print('At single cluster ' + str(keys) + ', keeping same color')
            h = np.ones((ht,wd,3)).astype('uint8')
            h[:,:,0] = color_to_keep[0]
            h[:,:,1] = color_to_keep[1]
            h[:,:,2] = color_to_keep[2]
            if print_colors == True:
                plt.figure(figsize=(1,1))
                plt.imshow(h)
                plt.show()
        else:
            # Itertaing through every item of the cluster
            # -------------------------------------------
            for i in range(m):
                curr_color_in_cluster = curr_colors[i]

                # Find what centroid it belongs to
                # --------------------------------
                for j in range(total_cen):
                    if cen[j][0] == curr_color_in_cluster[0] and cen[j][1] == curr_color_in_cluster[1] and cen[j][2] == curr_color_in_cluster[2]:
                        curr_centroid = j

                # We have the curr cluster colors centroid, now to find the locs it appears in
                # ----------------------------------------------------------------------------
                loc_list = list(np.argwhere(labels == curr_centroid)[:,0])
                curr_centroid_tot_locs = len(loc_list)
                locs_counter.append(curr_centroid_tot_locs)

            # Now out of the cluster colors loop and deciding which color to keep
            # -------------------------------------------------------------------
            max_ind = locs_counter.index(max(locs_counter))
            color_to_keep = curr_colors[max_ind]
            #print('At cluster ' + str(keys) + ', keeping color number ' + str(max_ind) + ' from below locs list..')
            #print(locs_counter)
            h = np.ones((ht,wd,3)).astype('uint8')
            h[:,:,0] = color_to_keep[0]
            h[:,:,1] = color_to_keep[1]
            h[:,:,2] = color_to_keep[2]

            if print_colors == True:
                plt.figure(figsize=(1,1))
                plt.imshow(h)
                plt.show()

        # Concating current color for output
        # ----------------------------------

        if counter == 1:
            xout = h.reshape(1,h.shape[0],h.shape[1],3)
        else:
            xout = np.concatenate((xout,h.reshape(1,h.shape[0],h.shape[1],3)), axis = 0)

    return xout

# 5
# API ready ##
# Main function that returns textures
# -----------------------------------
def protomatebeta_build_textures_v1(x,hin,win,print_colorscale,progress,task_id):


    # Some initial initialisations
    # ----------------------------
    m = x.shape[0]
    cluster_threshold = 35

    # Iterating through images
    # ------------------------
    for i in range(m):

        print('At image ' + str(i+1) + ' of around ' + str(m) + '..')
        img = x[i]

        # 1. Bluring ops
        # --------------
        print('1. Blurring ops started..')
        for _ in range(1):
            img = cv2.medianBlur(img, 3)
        for _ in range(2):
            img = cv2.edgePreservingFilter(img, flags=2, sigma_s=10, sigma_r=0.12)

        # 2. kmeans ops
        # -------------
        print('2. Recurring k means started..')
        kmimg,centroids,_,_ = protomatebeta_recurr_kmeans_v1(img,30,30,True) # 75,75
        #plt.figure(figsize=(2,3))
        #plt.imshow(kmimg)
        #plt.show()

        # 3. Actual extraction using pillow method
        # -----------------------------------------
        resimg = Image.fromarray(kmimg)
        tu = resimg.getcolors()

        # 3.2 Removing minority colors
        # ----------------------------
        #tu = copy.deepcopy(new_tu)
        tot_vals = 0
        for tuc in range(len(tu)):
            tot_vals += tu[tuc][0]
        indx_counter = 0
        for tuc in range(len(tu)):
            if float(tu[indx_counter][0]/tot_vals) <= 0.001:
                tu.pop(indx_counter)
            else:
                indx_counter += 1
        no_colors = len(tu)
        #print('Number of colors extracted before clustering (tu length): ' + str(no_colors))


        # 4. Code to cluster similar colors & get out stripes and checks
        # ---------------------------------------------------------------
        oim_stripes,oim_checks = protomatebeta_cluster_colors_products_v1(tu,cluster_threshold,hin,win)
        oim_stripes = oim_stripes.reshape(1,oim_stripes.shape[0],oim_stripes.shape[1],3)
        oim_checks = oim_checks.reshape(1,oim_checks.shape[0],oim_checks.shape[1],3)
        if i == 0:
            out_stripes_final = oim_stripes
            out_checks_final = oim_checks

        else:
            out_stripes_final = np.concatenate((out_stripes_final,oim_stripes), axis = 0)
            out_checks_final = np.concatenate((out_checks_final,oim_checks), axis = 0)

        # Updating progress
        # -----------------
        curr_prog_percent = int(round((i+1)/m,2)*100)
        progress['curr_message'] = str(progress['master_message']) + '..about ' + str(curr_prog_percent) + '% through'

        # Saving textures to storage for keeping frontend progress
        # --------------------------------------------------------
        # CHECKS
        ##
        storage_dir = str(task_id) + '/texturespreview'
        image_prefix = str(task_id) + '_textures_preview_checks'
        save_to_storage_from_array_list(oim_checks,storage_dir,image_prefix,False,None)
        # STRIPES
        ##
        storage_dir = str(task_id) + '/texturespreview'
        image_prefix = str(task_id) + '_textures_preview_stripes'
        save_to_storage_from_array_list(oim_stripes,storage_dir,image_prefix,False,None)


        # 5. Printing coloscales
        # ----------------------
        if print_colorscale == True:
            #plt.figure(figsize=(2,2))
            plt.imshow(oim_stripes[0])
            plt.show()
            plt.imshow(oim_checks[0])
            plt.show()

    return out_stripes_final, out_checks_final

# 5.1
# API ready ##
# Actual function to create stripes, checks, other textures
# ---------------------------------------------------------
def protomatebeta_create_stripes_checks_v1(tokd,wkd,repeat_h,hout,wout):

    # Here input dicts represent color scales from a single processed image
    # ---------------------------------------------------------------------

    # 1. Getting total weight
    # -----------------------
    tot_weight = 0
    for k in wkd:
        tot_weight += wkd[k]

    # 2. Some initialisations
    # -----------------------
    stripe_block = np.zeros((repeat_h,wout,3))
    verticle_stripe_block = np.zeros((repeat_h,hout,3))
    st_h = 0



    # 3. Spliting the block row wise
    # ------------------------------
    for k in wkd:

        # As we process the incoming colors and its weights,
        # there is randomness implicit here as we dont process,
        # in any particular order.
        # -----------------------------------------------------
        weight_percent = (wkd[k] / tot_weight)
        num_rows = math.ceil(weight_percent * repeat_h)

        # Initialising col pal for curr color
        # -----------------------------------
        col_pal = np.zeros((1,1,3))
        col_pal[:,:,0],col_pal[:,:,1],col_pal[:,:,2] = tokd[k]

        # Setting strip_block values
        # --------------------------
        try:
            # Incase we are still within height bounds
            # ----------------------------------------
            stripe_block[st_h:st_h+num_rows,:,:] = col_pal
            verticle_stripe_block[st_h:st_h+num_rows,:,:] = col_pal

        except:
            # Incase we are outside height bounds
            # -----------------------------------
            stripe_block[st_h:,:,:] = col_pal
            verticle_stripe_block[st_h:,:,:] = col_pal

        st_h = st_h+num_rows

    # 4. Building a full on repeat for stripes
    # ----------------------------------------
    no_rows_for_repeat = math.ceil(hout / repeat_h) + 1
    built_repeat = np.concatenate((stripe_block,stripe_block), axis = 0)
    for _ in range(no_rows_for_repeat - 2):
        built_repeat = np.concatenate((built_repeat,stripe_block), axis = 0)
    built_repeat_stripes = built_repeat[0:hout,0:wout,:].astype('uint8')

    # 5. Building a full on repeat for checks
    # ---------------------------------------
    no_rows_for_repeat_checks = math.ceil(wout / repeat_h) + 1
    built_repeat_horz = np.concatenate((verticle_stripe_block,verticle_stripe_block), axis = 0)
    for _ in range(no_rows_for_repeat_checks - 2):
        built_repeat_horz = np.concatenate((built_repeat_horz,verticle_stripe_block), axis = 0)
    built_repeat_horz = built_repeat_horz[0:wout,0:hout,:]
    built_repeat_horz = np.rot90(built_repeat_horz)
    stripe_horz = copy.deepcopy(built_repeat_horz).astype(float)
    stripe_vert = copy.deepcopy(built_repeat_stripes).astype(float)
    built_repeat_checks = ((stripe_horz + stripe_vert)/2).astype('uint8')


    return built_repeat_stripes, built_repeat_checks

# 5.2
# API ready ##
# Main funtion that is used to cluster similar colors returned from smaller product/pattern images
# -------------------------------------------------------------------------------------------------
def protomatebeta_cluster_colors_products_v1(tu,similarity_distance,hout,wout):

    # Some initial master settings
    # ----------------------------
    no_colors = len(tu)
    raw_colors = np.zeros((no_colors,1,1,3))
    weightage_by_color_dict = {}
    for no_c in range(no_colors):
        raw_colors[no_c,:,:,0],raw_colors[no_c,:,:,1],raw_colors[no_c,:,:,2] = tu[no_c][1]
        weightage_by_color_dict[tu[no_c][1]] = tu[no_c][0] # Setting dict(RGB) = no_locations

    total_cols = raw_colors.shape[0]
    master_all_ind = list(range(total_cols))

    # Recurrence settings - Initial one time set up
    # ---------------------------------------------
    colors_for_trav = copy.deepcopy(raw_colors)
    all_ind_for_trav = list(range(colors_for_trav.shape[0]))
    m = len(all_ind_for_trav)
    color_set_dict = {}
    set_num = 0
    to_keep_dict = {}
    weight_keep_dict = {}

    # Actual iter
    # ----------
    while m > 0:

        set_num += 1
        curr_set_ind = []
        curr_color = colors_for_trav[0]
        curr_color_intensity = curr_color

        for i in range(m):
            trav_curr_intensity = colors_for_trav[i]
            diff_r = abs(curr_color_intensity[:,:,0] - trav_curr_intensity[:,:,0])
            diff_g = abs(curr_color_intensity[:,:,1] - trav_curr_intensity[:,:,1])
            diff_b = abs(curr_color_intensity[:,:,2] - trav_curr_intensity[:,:,2])

            if diff_r <= similarity_distance and diff_g <= similarity_distance and diff_b <= similarity_distance:
                curr_set_ind.append(i)

        # Creating a new object in dictionary
        # -----------------------------------
        color_set_dict[set_num] = colors_for_trav[curr_set_ind]
        #print('Creating set ' + str(set_num) + '..')

        # Working out which colors to keep using another dict
        # ---------------------------------------------------
        tot_weight = 0
        to_keep_curr_set_color = None
        curr_weight_value_to_keep = 0
        for m_r in range(color_set_dict[set_num].shape[0]):
            curr_color_in_tuple = tuple(np.squeeze(color_set_dict[set_num][m_r].astype(int)))
            curr_weight_value_to_keep += weightage_by_color_dict[curr_color_in_tuple]
            if m_r == 0:
                to_keep_curr_set_color = curr_color_in_tuple
                #curr_weight_value_to_keep = weightage_by_color_dict[curr_color_in_tuple]
            else:
                if weightage_by_color_dict[curr_color_in_tuple] > weightage_by_color_dict[to_keep_curr_set_color]:
                    to_keep_curr_set_color = curr_color_in_tuple
                    #curr_weight_value_to_keep = weightage_by_color_dict[curr_color_in_tuple]

        to_keep_dict[set_num] = to_keep_curr_set_color
        weight_keep_dict[set_num] = curr_weight_value_to_keep


        # Popping the indices from the traversing color set
        # -------------------------------------------------
        new_ind_to_retain_for_next_iter = [x for x in all_ind_for_trav if x not in curr_set_ind]
        colors_for_trav = copy.deepcopy(colors_for_trav[new_ind_to_retain_for_next_iter])
        all_ind_for_trav = list(range(colors_for_trav.shape[0]))
        m = len(all_ind_for_trav)
        #print('At set ' + str(set_num) + ', curr m size is ' + str(m))



    outimage_stripes,outimage_checks = protomatebeta_create_stripes_checks_v1(to_keep_dict,weight_keep_dict,30,hout,wout)

    return outimage_stripes,outimage_checks

# 6
# Getting lines,segments & categories from storage and returning them as a numpy array and list
# API ready ##
# ---------------------------------------------------------------------------------------------
def get_stylings_from_storage(in_names):

    """"For example, given these blobs:
        /a/1.txt
        /a/b/2.txt
    If you just specify prefix = '/a', you'll get back:
        /a/1.txt
        /a/b/2.txt
    However, if you specify prefix='/a' and delimiter='/', you'll get back:
        /a/1.txt"""

    # Converting innames to string
    # ----------------------------
    in_names_list = in_names.split(',')

    # Create a storage client to use with bucket
    # ------------------------------------------
    bucket_name = 'ven-ml-project.appspot.com'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    # Initialising categories
    # -----------------------
    categories = []

    # Itering through names list
    # --------------------------
    counter = 0
    for i in in_names_list:

        #print('At image..' + str(i))
        print(i)
        print(type(i))

        # Getting linemarking blob for curr name
        # ---------------------------------------
        line_blob_name = 'linemarkings/' + str(i)
        blob_line = bucket.blob(line_blob_name)

        # Getting Segments blob for curr name
        # -----------------------------------
        seg_blob_name = 'segments/' + str(i)
        blob_seg = bucket.blob(seg_blob_name)

        # Setting images in both folder flag
        # ----------------------------------
        image_in_lines = 0
        image_in_seg = 0
        curr_img_line = None
        curr_img_seg = None

        try:
            # Using tempfile to retrieve linemarking
            # --------------------------------------
            with tempfile.NamedTemporaryFile() as temp:
                blob_line.download_to_filename(temp.name)
                img_l = cv2.imread(temp.name)
                img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
                img_l = img_l.reshape(1,img_l.shape[0],img_l.shape[1],img_l.shape[2])
                curr_img_line = copy.deepcopy(img_l)
                image_in_lines = 1

            # Using tempfile to retrieve segments
            # -----------------------------------
            with tempfile.NamedTemporaryFile() as temp:
                blob_seg.download_to_filename(temp.name)
                img_s = cv2.imread(temp.name)
                img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
                img_s = img_s.reshape(1,img_s.shape[0],img_s.shape[1],img_s.shape[2])
                curr_img_seg = copy.deepcopy(img_s)
                image_in_seg = 1
        except:
            print('At exception: ' + str(i))
            'one/both of the folders is missing an image'

        # Concatenating to output arrays only if image present in both folders
        # --------------------------------------------------------------------
        if image_in_lines == 1 and image_in_seg == 1:
            counter += 1
            if counter == 1:
                xout_lines = curr_img_line
                xout_seg = curr_img_seg
            else:
                xout_lines = np.concatenate((xout_lines,curr_img_line), axis = 0)
                xout_seg = np.concatenate((xout_seg,curr_img_seg), axis = 0)

            # Getting catrgory label to be used in final generation
            # -----------------------------------------------------
            curr_cat_label = int(str(i)[0:str(i).index('_')])
            categories.append(curr_cat_label)
            print('Added!')


    return xout_lines,xout_seg,categories

# 6.1
# API ready ##
# Function to correct segments and linemarkings
# ----------------------------------------------
def protomatebeta_correct_segments_linemarkings(lines,seg):

    # Initialisations
    # ---------------
    m = lines.shape[0]
    h,w = lines.shape[1],lines.shape[2]

    # Iters
    # -----
    for i in range(m):

        # Setting curr images for correction
        # ----------------------------------
        curr_seg = seg[i]
        curr_lines = lines[i]

        # Correcting segments
        # -------------------
        bl_d = 50
        gl_u,gl_d = bl_d,200
        w_u = gl_d
        black_positions = (curr_seg < bl_d).astype(int)
        grey_positions_up = (curr_seg >= gl_u).astype(int)
        grey_positions_down = (curr_seg < gl_d).astype(int)
        grey_positions = (grey_positions_up == grey_positions_down).astype(int)
        white_positions = (curr_seg >= gl_d).astype(int)

        grey_value = 150
        black_value = 0
        white_value = 255

        new_seg = np.ones((h,w,3),dtype = 'uint8') * 255
        new_seg = black_positions * black_value + grey_positions * grey_value + white_positions * white_value
        new_seg = new_seg.astype('uint8')

        # Correcting linemarkings
        # -----------------------
        white_area = (curr_lines < 200).astype('int') * 0
        line_area = (curr_lines >= 200).astype('int') * 255
        new_line = white_area + line_area
        new_line = new_line.astype('uint8')

        # Concat ops
        # ----------
        if i == 0:
            xout_seg = new_seg.reshape(1,new_seg.shape[0],new_seg.shape[1],3)
            xout_lines = new_line.reshape(1,new_line.shape[0],new_line.shape[1],3)
        else:
            xout_seg = np.concatenate((xout_seg,new_seg.reshape(1,new_seg.shape[0],new_seg.shape[1],3)), axis = 0)
            xout_lines = np.concatenate((xout_lines,new_line.reshape(1,new_line.shape[0],new_line.shape[1],3)), axis = 0)


    return xout_lines,xout_seg

# 7
# API read ##
# Main function that generates ideas
# ----------------------------------
###########################################
# To include more textures and categories #
###########################################
def protomatebeta_create_ideas_v2(segments,linemarkings,categories,patterns,stripes,checks,colors,no_images,progress,task_id,gen_id):

    # Initialisations
    # ---------------
    seg_index = list(range(segments.shape[0]))
    patterns_index = list(range(patterns.shape[0]))
    stripes_index = list(range(stripes.shape[0]))
    checks_index = list(range(checks.shape[0]))
    colors_index = list(range(colors.shape[0]))
    m = segments.shape[0]

    inh_r = segments.shape[1]
    inw_r = segments.shape[2]
    list_combos = []

    # tup will be always of form (seg_index,pattern_index,color_index,stripes_index,check_index)
    # Will use 0 in case of specific categories or single segments

    # Iterating for number of required output
    # ---------------------------------------
    for i in range(no_images):

        print('Generating image..' + str(i+1))


        # First getting the seg_index to use for evenly displaying generations
        # --------------------------------------------------------------------
        if i < m:
            s_index = i
        else:
            s_index = int(i%m)

        # Picking segments & patterns
        # ---------------------------
        if i == 0:

            # Setting segment
            # ---------------
            segc = segments[s_index]
            category_curr_seg = categories[s_index,0]
            temp_grey_pos = (segc == 150).astype(int)
            temp_black_pos = (segc == 0).astype(int)

            # Finding proportions
            # -------------------
            grey_prop = np.sum(temp_grey_pos)
            black_prop = np.sum(temp_black_pos)
            total_prop = grey_prop + black_prop
            grey_fac = grey_prop / total_prop
            black_fac = black_prop / total_prop
            porp_threshold = 0.15


            # Figuring if the segment is a single or double
            # ---------------------------------------------
            if np.sum(temp_grey_pos) == 0:

                # No gray segments in the seg image
                # Having category rule here --
                # ---------------------------------
                if category_curr_seg == 0: # Girls dress

                    # Making a legitimate choice
                    # --------------------------
                    choice = random.choice([0,1])

                    if choice == 0: # Use patterns
                        b_index = random.choice(patterns_index)
                        #tup -- (seg_index,pattern_index,color_index,stripes_index,check_index)
                        tup = (s_index,b_index,0,0,0)
                        gblock = patterns[b_index]
                        bblock = patterns[b_index]

                    else: # Using Checks
                        b_index = random.choice(checks_index)
                        #tup -- (seg_index,pattern_index,color_index,stripes_index,check_index)
                        tup = (s_index,0,0,0,b_index)
                        gblock = checks[b_index]
                        bblock = checks[b_index]

                elif category_curr_seg == 1: # Girls top
                    'code'
                elif category_curr_seg == 2: # Girls Jeans
                    'code'
                elif category_curr_seg == 3: # Girls
                    'code'
                elif category_curr_seg == 4: # Girls top
                    'code'

            elif np.sum(temp_black_pos) == 0:

                # No black segments in the seg image
                # Having category rule here --
                # ---------------------------------
                if category_curr_seg == 0: # Girls dress

                    # Making a legitimate choice
                    # --------------------------
                    choice = random.choice([0,1])

                    if choice == 0: # Use patterns
                        g_index = random.choice(patterns_index)
                        #tup -- (seg_index,pattern_index,color_index,stripes_index,check_index)
                        tup = (s_index,g_index,0,0,0)
                        gblock = patterns[g_index]
                        bblock = patterns[g_index]

                    else: # Using Checks
                        g_index = random.choice(checks_index)
                        #tup -- (seg_index,pattern_index,color_index,stripes_index,check_index)
                        tup = (s_index,0,0,0,g_index)
                        gblock = checks[g_index]
                        bblock = checks[g_index]

                elif category_curr_seg == 1: # Girls top
                    'code'
                elif category_curr_seg == 2: # Girls Jeans
                    'code'
                elif category_curr_seg == 3: # Girls
                    'code'
                elif category_curr_seg == 4: # Girls top
                    'code'


            else:

                # Both segments are available
                # Having category rule here --
                # ---------------------------------
                if category_curr_seg == 0: # Girls dress

                    # Making a legitimate choice
                    # --------------------------
                    choice = random.choice([0,1])

                    if choice == 0: # Use patterns

                        # Checking to see if either of the segments is very tiny in proportion
                        # so that we dont end up assigning plain color to major portion
                        # --------------------------------------------------------------------
                        if grey_fac < porp_threshold:

                            # Grey seg is less
                            # ----------------
                            g_index = random.choice(colors_index)
                            b_index = random.choice(patterns_index)
                            #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                            tup = (s_index,b_index,g_index,0,0)
                            gblock = colors[g_index]
                            bblock = patterns[b_index]

                        elif black_fac < porp_threshold:

                            # Black seg is less
                            # ----------------
                            g_index = random.choice(patterns_index)
                            b_index = random.choice(colors_index)
                            #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                            tup = (s_index,g_index,b_index,0,0)
                            gblock = patterns[g_index]
                            bblock = colors[b_index]

                        else:
                            # Both grey and black available in good proportions
                            # -------------------------------------------------
                            seg_choice = random.choice([1,2])
                            if seg_choice == 1: # Make grey segment as color
                                g_index = random.choice(colors_index)
                                gblock = colors[g_index]

                                b_index = random.choice(patterns_index)
                                bblock = patterns[b_index]

                                #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                                tup = (s_index,b_index,g_index,0,0)

                            else: # Make black segment as color
                                b_index = random.choice(colors_index)
                                bblock = colors[b_index]

                                g_index = random.choice(patterns_index)
                                gblock = patterns[g_index]

                                #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                                tup = (s_index,g_index,b_index,0,0)

                    else: # Using Checks

                        # Checking to see if either of the segments is very tiny in proportion
                        # so that we dont end up assigning plain color to major portion
                        # --------------------------------------------------------------------
                        if grey_fac < porp_threshold:

                            # Grey seg is less
                            # ----------------
                            g_index = random.choice(colors_index)
                            b_index = random.choice(checks_index)
                            #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                            tup = (s_index,0,g_index,0,b_index)
                            gblock = colors[g_index]
                            bblock = checks[b_index]

                        elif black_fac < porp_threshold:

                            # Black seg is less
                            # ----------------
                            g_index = random.choice(checks_index)
                            b_index = random.choice(colors_index)
                            #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                            tup = (s_index,0,b_index,0,g_index)
                            gblock = checks[g_index]
                            bblock = colors[b_index]

                        else:
                            # Both grey and black available in good proportions
                            # -------------------------------------------------
                            seg_choice = random.choice([1,2])
                            if seg_choice == 1: # Make grey segment as color
                                g_index = random.choice(colors_index)
                                gblock = colors[g_index]

                                b_index = random.choice(checks_index)
                                bblock = checks[b_index]

                                #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                                tup = (s_index,0,g_index,0,b_index)

                            else: # Make black segment as color
                                b_index = random.choice(colors_index)
                                bblock = colors[b_index]

                                g_index = random.choice(checks_index)
                                gblock = checks[g_index]

                                #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                                tup = (s_index,0,b_index,0,g_index)

                elif category_curr_seg == 1: # Girls top
                    'code'
                elif category_curr_seg == 2: # Girls Jeans
                    'code'
                elif category_curr_seg == 3: # Girls
                    'code'
                elif category_curr_seg == 4: # Girls top
                    'code'


            list_combos.append(tup)

        else:
            start_time = time.time()

            while True:

                # Setting segment
                # ---------------
                segc = segments[s_index]
                category_curr_seg = categories[s_index,0]
                temp_grey_pos = (segc == 150).astype(int)
                temp_black_pos = (segc == 0).astype(int)

                # Finding proportions
                # -------------------
                grey_prop = np.sum(temp_grey_pos)
                black_prop = np.sum(temp_black_pos)
                total_prop = grey_prop + black_prop
                grey_fac = grey_prop / total_prop
                black_fac = black_prop / total_prop
                porp_threshold = 0.15


                # Figuring if the segment is a single or double
                # ---------------------------------------------
                if np.sum(temp_grey_pos) == 0:

                    # No gray segments in the seg image
                    # Having category rule here --
                    # ---------------------------------
                    if category_curr_seg == 0: # Girls dress

                        # Making a legitimate choice
                        # --------------------------
                        choice = random.choice([0,1])

                        if choice == 0: # Use patterns
                            b_index = random.choice(patterns_index)
                            #tup -- (seg_index,pattern_index,color_index,stripes_index,check_index)
                            tup = (s_index,b_index,0,0,0)
                            gblock = patterns[b_index]
                            bblock = patterns[b_index]

                        else: # Using Checks
                            b_index = random.choice(checks_index)
                            #tup -- (seg_index,pattern_index,color_index,stripes_index,check_index)
                            tup = (s_index,0,0,0,b_index)
                            gblock = checks[b_index]
                            bblock = checks[b_index]

                    elif category_curr_seg == 1: # Girls top
                        'code'
                    elif category_curr_seg == 2: # Girls Jeans
                        'code'
                    elif category_curr_seg == 3: # Girls
                        'code'
                    elif category_curr_seg == 4: # Girls top
                        'code'

                elif np.sum(temp_black_pos) == 0:

                    # No black segments in the seg image
                    # Having category rule here --
                    # ---------------------------------
                    if category_curr_seg == 0: # Girls dress

                        # Making a legitimate choice
                        # --------------------------
                        choice = random.choice([0,1])

                        if choice == 0: # Use patterns
                            g_index = random.choice(patterns_index)
                            #tup -- (seg_index,pattern_index,color_index,stripes_index,check_index)
                            tup = (s_index,g_index,0,0,0)
                            gblock = patterns[g_index]
                            bblock = patterns[g_index]

                        else: # Using Checks
                            g_index = random.choice(checks_index)
                            #tup -- (seg_index,pattern_index,color_index,stripes_index,check_index)
                            tup = (s_index,0,0,0,g_index)
                            gblock = checks[g_index]
                            bblock = checks[g_index]

                    elif category_curr_seg == 1: # Girls top
                        'code'
                    elif category_curr_seg == 2: # Girls Jeans
                        'code'
                    elif category_curr_seg == 3: # Girls
                        'code'
                    elif category_curr_seg == 4: # Girls top
                        'code'

                else:

                    # Both segments are available
                    # Having category rule here --
                    # ---------------------------------
                    if category_curr_seg == 0: # Girls dress

                        # Making a legitimate choice
                        # --------------------------
                        choice = random.choice([0,1])

                        if choice == 0: # Use patterns

                            # Checking to see if either of the segments is very tiny in proportion
                            # so that we dont end up assigning plain color to major portion
                            # --------------------------------------------------------------------
                            if grey_fac < porp_threshold:

                                # Grey seg is less
                                # ----------------
                                g_index = random.choice(colors_index)
                                b_index = random.choice(patterns_index)
                                #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                                tup = (s_index,b_index,g_index,0,0)
                                gblock = colors[g_index]
                                bblock = patterns[b_index]

                            elif black_fac < porp_threshold:

                                # Black seg is less
                                # ----------------
                                g_index = random.choice(patterns_index)
                                b_index = random.choice(colors_index)
                                #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                                tup = (s_index,g_index,b_index,0,0)
                                gblock = patterns[g_index]
                                bblock = colors[b_index]

                            else:
                                # Both grey and black available in good proportions
                                # -------------------------------------------------
                                seg_choice = random.choice([1,2])
                                if seg_choice == 1: # Make grey segment as color
                                    g_index = random.choice(colors_index)
                                    gblock = colors[g_index]

                                    b_index = random.choice(patterns_index)
                                    bblock = patterns[b_index]

                                    #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                                    tup = (s_index,b_index,g_index,0,0)

                                else: # Make black segment as color
                                    b_index = random.choice(colors_index)
                                    bblock = colors[b_index]

                                    g_index = random.choice(patterns_index)
                                    gblock = patterns[g_index]

                                    #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                                    tup = (s_index,g_index,b_index,0,0)

                        else: # Using Checks

                            # Checking to see if either of the segments is very tiny in proportion
                            # so that we dont end up assigning plain color to major portion
                            # --------------------------------------------------------------------
                            if grey_fac < porp_threshold:

                                # Grey seg is less
                                # ----------------
                                g_index = random.choice(colors_index)
                                b_index = random.choice(checks_index)
                                #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                                tup = (s_index,0,g_index,0,b_index)
                                gblock = colors[g_index]
                                bblock = checks[b_index]

                            elif black_fac < porp_threshold:

                                # Black seg is less
                                # ----------------
                                g_index = random.choice(checks_index)
                                b_index = random.choice(colors_index)
                                #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                                tup = (s_index,0,b_index,0,g_index)
                                gblock = checks[g_index]
                                bblock = colors[b_index]

                            else:
                                # Both grey and black available in good proportions
                                # -------------------------------------------------
                                seg_choice = random.choice([1,2])
                                if seg_choice == 1: # Make grey segment as color
                                    g_index = random.choice(colors_index)
                                    gblock = colors[g_index]

                                    b_index = random.choice(checks_index)
                                    bblock = checks[b_index]

                                    #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                                    tup = (s_index,0,g_index,0,b_index)

                                else: # Make black segment as color
                                    b_index = random.choice(colors_index)
                                    bblock = colors[b_index]

                                    g_index = random.choice(checks_index)
                                    gblock = checks[g_index]

                                    #tup -- (seg_index, pattern_index, color_index, stripes_index, check_index)
                                    tup = (s_index,0,b_index,0,g_index)

                    elif category_curr_seg == 1: # Girls top
                        'code'
                    elif category_curr_seg == 2: # Girls Jeans
                        'code'
                    elif category_curr_seg == 3: # Girls
                        'code'
                    elif category_curr_seg == 4: # Girls top
                        'code'

                # Code that runs until new combo is caught
                # ----------------------------------------
                if tup in list_combos:
                    # Old combo
                    ##
                    'do nothing'
                else:
                    # New combo
                    ##
                    list_combos.append(tup)
                    break

                # Code that times out this while loop incase it goes over board
                # -------------------------------------------------------------
                curr_time = time.time()
                if (end_time - start_time) > 5: # 5 secs
                    return genout # Returns genout and exits


        # Actual masking ops
        # ------------------
        grey_pos = (segc == 150).astype(int)
        black_pos = (segc == 0).astype(int)
        white_pos = (segc == 255).astype(int)
        ones = np.ones((inh_r,inw_r,3), dtype = 'uint8') * 255
        newim = grey_pos * gblock + black_pos * bblock + white_pos * ones
        newim = newim.astype('uint8')

        # Adding line markings
        # --------------------
        line_map = linemarkings[s_index]
        markings_main_image = (line_map > 0).astype(int).reshape(line_map.shape[0],line_map.shape[1],3)
        markings_line_map = (line_map == 0).astype(int).reshape(line_map.shape[0],line_map.shape[1],3)
        outline_dark_value = 5
        newim = markings_main_image * newim + markings_line_map * outline_dark_value

        # Final concatenation
        # -------------------
        newim_c = newim.reshape(1,newim.shape[0],newim.shape[1],3)
        newim_c = newim_c.astype('uint8')

        # Updating progress
        # -----------------
        curr_prog_percent = int(round((i+1)/no_images,2)*100)
        progress['curr_message'] = str(progress['master_message']) + '..about ' + str(curr_prog_percent) + '% through'

        # Saving current generation to storage for keeping frontend progress
        # -------------------------------------------------------------------
        # CHECKS
        ##
        storage_dir = str(task_id) + '/ideas/ideaspreview'
        image_prefix = str(task_id) + '_' + str(gen_id) + '_preview'
        save_to_storage_from_array_list(newim_c,storage_dir,image_prefix,False,None)

        if i == 0:
            genout = newim_c
        else:
            genout = np.concatenate((genout,newim_c), axis = 0)

    return genout

# # Ven_API functions
# API 1 Function
# API -- create_new_patterns
# creates new patterns for the first time using uploaded theme images at
# /task_id/themes/ and saves to /task_id/all_patterns/
# Creates all_patterns, colors, all_maps numpy arrays and stores them at /task_id/numpy/
# Creates segments, linemarkings, categories from input selected styling names and saves as numpy
# at /task_id/numpy for easy generation
# Returns OK, NOT OK
# ------------------------------------------------------------------------------
def api_create_new_patterns(task_id,selected_style_names,progress):

    progress['curr_step'] = 0
    progress['total_step'] = 10
    progress['master_message'] = 'Inside function..'
    progress['curr_message'] = 'Inside function..'

    #try:

    # Setting input folder name as per set format
    # -------------------------------------------
    inputfolder = task_id + '/themes'

    # Standard initialisations
    # ------------------------
    h,w,rp_size = 285,221,30

    # 1. Getting theme images from storage
    # ------------------------------------
    progress['curr_step'] = 1
    progress['master_message'] = 'Getting theme images from storage..'
    progress['curr_message'] = progress['master_message']
    theme_list = get_images_from_storage(inputfolder,'list')

    # 2. Stitching images together as list
    # ------------------------------------
    themes_stitched = protomatebeta_stitch_incoming_images_v1(theme_list)

    # 3. Extracting blocks from stitched images
    # -----------------------------------------
    progress['curr_step'] = 2
    progress['master_message'] = 'Learning objects in the theme images..this may take a while (around 15 seconds per image)..'
    progress['curr_message'] = progress['master_message']
    flblocks,flblocks_maps = protomatebeta_extract_blocks_for_aop_v1(themes_stitched,progress)

    # 4. Building patterns
    # --------------------
    progress['curr_step'] = 3
    progress['master_message'] = 'Building patterns based on learnt objects..'
    progress['curr_message'] = progress['master_message']
    built_patterns = protomate_build_aop_patterns_v1(flblocks,h,w,rp_size)

    # 5. Picking key colors
    # ---------------------
    progress['curr_step'] = 4
    progress['master_message'] = 'Learning core colors from theme images..'
    progress['curr_message'] = progress['master_message']
    keycolors = protomatebeta_pickcolors_v1(progress,themes_stitched,h,w,similarity_distance=0.1)

    # 6. Building maps
    # ----------------
    built_maps = protomate_build_aop_patterns_v1(flblocks_maps,h,w,rp_size)

    # 7. Saving all patterns to storage for front end retrieval
    # ---------------------------------------------------------
    progress['curr_step'] = 5
    progress['master_message'] = 'Saving built patterns for display..'
    progress['curr_message'] = progress['master_message']
    storage_dir = task_id + '/all_patterns'
    image_prefix = str(task_id) + '_all_patterns'
    save_to_storage_from_array_list(built_patterns,storage_dir,image_prefix,True,progress)

    # 8. Saving numpy formats of all patterns, all maps, all colors
    # -------------------------------------------------------------
    # 8.1 Saving all patterns
    # -----------------------
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_all_patterns.npy'
    np.save(file_io.FileIO(storage_address, 'w'), built_patterns)

    # 8.2 Saving all colors
    # -----------------------
    progress['curr_step'] = 6
    progress['master_message'] = 'Saving core colors for display..'
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_all_colors.npy'
    np.save(file_io.FileIO(storage_address, 'w'), keycolors)

    # 8.2 Saving all maps
    # -----------------------
    progress['curr_step'] = 7
    progress['master_message'] = 'Saving pattern maps for later use..'
    progress['curr_message'] = progress['master_message']
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_all_maps.npy'
    np.save(file_io.FileIO(storage_address, 'w'), built_maps)

    # 9. Picking selected stylings and saving them into temp arrays for correction
    # ----------------------------------------------------------------------------
    progress['curr_step'] = 8
    progress['master_message'] = 'Retrieving selected stylings for generation..'
    progress['curr_message'] = progress['master_message']
    x_lines,x_segs,cats = get_stylings_from_storage(selected_style_names)

    # 9.1. Correcting incoming lines and segments
    # -----------------------------------------
    xl_corr,xs_corr = protomatebeta_correct_segments_linemarkings(x_lines,x_segs)

    # 10. Saving lines, segs and categories under /task_id/numpy for generation
    # -------------------------------------------------------------------------
    # Saving Segments
    # ---------------
    progress['curr_step'] = 9
    progress['master_message'] = 'Saving stylings and categories for generation..'
    progress['curr_message'] = progress['master_message']
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_segments.npy'
    np.save(file_io.FileIO(storage_address, 'w'), xs_corr)

    # Saving Linemarkings
    # --------------------
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_linemarkings.npy'
    np.save(file_io.FileIO(storage_address, 'w'), xl_corr)

    # Saving Categories
    # -----------------
    categories_np = np.array(cats).reshape(len(cats), 1)
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_categories.npy'
    np.save(file_io.FileIO(storage_address, 'w'), categories_np)

    progress['curr_step'] = 10
    progress['master_message'] = 'All Done.'
    progress['curr_message'] = progress['master_message']

    # Reading numpy files from cloud
    ###
    #f = BytesIO(file_io.read_file_to_string('gs://ven-ml-project.appspot.com/PMTASK001/numpy/nptest.npy', binary_mode=True))
    #x = np.load(f)

    return 200 # OK

    #except:
    #
    #    return 500 # NOT OK

# API 2 function
# API -- create_textures
# Takes in task_id, selected_indices (as string) and creates textures
# saving them under /task_id/numpy/ And also saved picked indices as numpy array
# At same location
# Returns OK, NOT OK
# ------------------------------------------------------------------------------
def api_create_textures(task_id,picked_ind_string,progress):

    #try:
    progress['curr_step'] = 0
    progress['total_step'] = 5
    progress['master_message'] = 'Inside function..'
    progress['curr_message'] = 'Inside function..'

    # 1. Converting input indices to string
    # -------------------------------------
    picked_ind = [int(s) for s in picked_ind_string.split(',')]

    # 2. Getting all patterns npy file
    # --------------------------------
    progress['curr_step'] = 1
    progress['total_step'] = 5
    progress['master_message'] = 'Loading saved patterns..'
    progress['curr_message'] = progress['master_message']
    print('1. At np array load..')
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_all_patterns.npy'
    f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
    all_pats = np.load(f)
    picked_patterns = all_pats[picked_ind]
    h,w = picked_patterns.shape[1],picked_patterns.shape[2]

    # 3. Creating stripes and checks
    # ------------------------------
    progress['curr_step'] = 2
    progress['total_step'] = 5
    progress['master_message'] = 'Building textures..'
    progress['curr_message'] = progress['master_message']
    print('2. At textures..')
    picked_stripes,picked_checks = protomatebeta_build_textures_v1(picked_patterns,h,w,False,progress,task_id)

    # 4. Saving textures under /task_id/numpy/
    # ----------------------------------------
    # Saving Stripes
    # --------------
    progress['curr_step'] = 3
    progress['total_step'] = 5
    progress['master_message'] = 'Saving textures..stripes'
    progress['curr_message'] = progress['master_message']
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_stripes.npy'
    np.save(file_io.FileIO(storage_address, 'w'), picked_stripes)
    print('3. Saved stripes..')

    # Saving Checks
    # --------------
    progress['curr_step'] = 4
    progress['total_step'] = 5
    progress['master_message'] = 'Saving textures..checks'
    progress['curr_message'] = progress['master_message']
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_checks.npy'
    np.save(file_io.FileIO(storage_address, 'w'), picked_checks)
    print('4. Saved checks..')

    # Saving Picked_ind
    # -----------------
    picked_ind_np = np.array(picked_ind).reshape(len(picked_ind), 1)
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_picked_ind.npy'
    np.save(file_io.FileIO(storage_address, 'w'), picked_ind_np)
    print('5. Saved indices..')
    progress['curr_step'] = 5
    progress['total_step'] = 5
    progress['master_message'] = 'Done.'
    progress['curr_message'] = progress['master_message']

    return 200

    #except:

    #    return 500

# API 3 function
# API -- generate ideas
# Takes in task_id, gen_id, no_images as inputs
# Generates new ideas and saves them under /task_id/ideas/gen_id/
# Returns OK, NOT OK
# ------------------------------------------------------------------------------
def api_generate(task_id,gen_id,no_images,progress):

#    try:
    progress['curr_step'] = 0
    progress['total_step'] = 9
    progress['master_message'] = 'Inside function..'
    progress['curr_message'] = 'Inside function..'


    # 1. Collect required numpy files and load them locally for generation
    # --------------------------------------------------------------------
    # Collecting linemarkings
    # -----------------------
    progress['curr_step'] = 1
    progress['total_step'] = 9
    progress['master_message'] = 'Loading stylings..'
    progress['curr_message'] = progress['master_message']
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_linemarkings.npy'
    f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
    lines = np.load(f)
    print('Got lines..')

    # Collecting segments
    # -------------------
    progress['curr_step'] = 2
    progress['total_step'] = 9
    progress['master_message'] = 'Loading segments..'
    progress['curr_message'] = progress['master_message']
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_segments.npy'
    f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
    segs = np.load(f)
    print('Got segs..')

    # Collecting all pats
    # -------------------
    progress['curr_step'] = 3
    progress['total_step'] = 9
    progress['master_message'] = 'Loading patterns..'
    progress['curr_message'] = progress['master_message']
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_all_patterns.npy'
    f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
    all_patterns = np.load(f)
    print('Got all pats..')

    # Collecting picked_ind
    # ---------------------
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_picked_ind.npy'
    f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
    picked_ind = np.load(f)
    print('Got picked indices..')

    # Collecting colors
    # ---------------------
    progress['curr_step'] = 4
    progress['total_step'] = 9
    progress['master_message'] = 'Loading colors..'
    progress['curr_message'] = progress['master_message']
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_all_colors.npy'
    f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
    colors = np.load(f)
    print('Got colors..')

    # Collecting Checks
    # -----------------
    progress['curr_step'] = 5
    progress['total_step'] = 9
    progress['master_message'] = 'Loading checks..'
    progress['curr_message'] = progress['master_message']
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_checks.npy'
    f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
    checks = np.load(f)
    print('Got checks..')

    # Collecting Stripes
    # -----------------
    progress['curr_step'] = 6
    progress['total_step'] = 9
    progress['master_message'] = 'Loading stripes..'
    progress['curr_message'] = progress['master_message']
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_stripes.npy'
    f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
    stripes = np.load(f)
    print('Got stripes..')

    # Collecting Catagories
    # ----------------------
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_categories.npy'
    f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
    categories = np.load(f)
    print('Got categories..')

    # 2. Preparing picked patterns for generation
    # -------------------------------------------
    picked_patterns = all_patterns[list(picked_ind[:,0])]

    # 3. Actual generation
    # --------------------
    progress['curr_step'] = 7
    progress['total_step'] = 9
    progress['master_message'] = 'Generating ideas..'
    progress['curr_message'] = progress['master_message']
    ideas = protomatebeta_create_ideas_v2(segs,lines,categories,picked_patterns,stripes,checks,colors,no_images,progress,task_id,gen_id)

    # 4. Saving generated images under /task_id/ideas/gen_id/
    # -------------------------------------------------------
    progress['curr_step'] = 8
    progress['total_step'] = 9
    progress['master_message'] = 'Saving ideas..'
    progress['curr_message'] = progress['master_message']
    storage_dir = task_id + '/ideas/' + str(gen_id)
    image_prefix = str(task_id) + '_' + str(gen_id) + '_ideas'
    save_to_storage_from_array_list(ideas,storage_dir,image_prefix,True,progress)

    progress['curr_step'] = 9
    progress['total_step'] = 9
    progress['master_message'] = 'Done.'
    progress['curr_message'] = progress['master_message']

#        return 200

#    except:
#
#        return 500


# # Actual Ven API endpoints
# ### 1. create new patterns external API
# ------------------------------------------------------------------------------
class create_new_patterns_threaded_task(threading.Thread):
    def __init__(self,p_task_id,p_selected_style_names):
        super().__init__()
        self.progress = {}
        self.progress['curr_step'] = 0
        self.progress['total_step'] = 0
        self.progress['master_message'] = 'Starting soon..'
        self.progress['curr_message'] = 'Starting soon..'
        self.p_task_id = p_task_id
        self.p_selected_style_names = p_selected_style_names

    def run(self): # Run method probably overrides the inherited Threads run method

        # 1. Running the create new patterns function
        # -------------------------------------------
        api_create_new_patterns(self.p_task_id,self.p_selected_style_names,self.progress)


# Creating a Main global dictionary to track progress of create new pattern task
# ------------------------------------------------------------------------------
global new_pattern_threads
new_pattern_threads = {}

## MAIN function TO BE CALLED ON API
# ----------------------------------
class externalAPI_create_new_patterns(Resource):

    def put(self):

        # Setting up key values to accept
        # -------------------------------
        parser = reqparse.RequestParser()
        parser.add_argument('task_id')
        parser.add_argument('selected_style_names')
        args = parser.parse_args()

        p_task_id = args['task_id']
        p_selected_style_names = args['selected_style_names']

        # This function really just starts the create new pattern thread class and assigns it to a global dict
        # ----------------------------------------------------------------------------------------------------
        global new_pattern_threads
        try:
            del new_pattern_threads[p_task_id]
        except:
            'do nothing'
        new_pattern_threads[p_task_id] = create_new_patterns_threaded_task(p_task_id,p_selected_style_names)
        new_pattern_threads[p_task_id].start()

        return 'Thread started', 200


# ### 2. create new texture external API
# ------------------------------------------------------------------------------
class create_textures_threaded_task(threading.Thread):
    def __init__(self,p_task_id,p_picked_ind_string):
        super().__init__()
        self.progress = {}
        self.progress['curr_step'] = 0
        self.progress['total_step'] = 0
        self.progress['master_message'] = 'Starting soon..'
        self.progress['curr_message'] = 'Starting soon..'
        self.p_task_id = p_task_id
        self.p_picked_ind_string = p_picked_ind_string

    def run(self): # Run method probably overrides the inherited Threads run method

        # 1. Running the create textures function
        # -------------------------------------------
        api_create_textures(self.p_task_id,self.p_picked_ind_string,self.progress)

# Creating a Main global dictionary to track progress of create textures task
# ---------------------------------------------------------------------------
global create_texture_threads
create_texture_threads = {}

## MAIN function TO BE CALLED ON API for creating texture
# -------------------------------------------------------
class externalAPI_create_textures(Resource):

    def put(self):

        # Setting up key values to accept
        # -------------------------------
        parser = reqparse.RequestParser()
        parser.add_argument('task_id')
        parser.add_argument('picked_ind_string')
        args = parser.parse_args()

        p_task_id = args['task_id']
        p_picked_ind_string = args['picked_ind_string']

        # This function really just starts the create new pattern thread class and assigns it to a global dict
        # ----------------------------------------------------------------------------------------------------
        global create_texture_threads
        try:
            del create_texture_threads[p_task_id]
        except:
            'do nothing'
        create_texture_threads[p_task_id] = create_textures_threaded_task(p_task_id,p_picked_ind_string)
        create_texture_threads[p_task_id].start()

        return 'Thread started', 200


# ### 3. generate ideas external API
# ------------------------------------------------------------------------------
class generate_ideas_threaded_task(threading.Thread):
    def __init__(self,p_task_id,p_gen_id,p_no_images):
        super().__init__()
        self.progress = {}
        self.progress['curr_step'] = 0
        self.progress['total_step'] = 0
        self.progress['master_message'] = 'Starting soon..'
        self.progress['curr_message'] = 'Starting soon..'
        self.p_task_id = p_task_id
        self.p_gen_id = p_gen_id
        self.p_no_images = p_no_images

    def run(self): # Run method probably overrides the inherited Threads run method

        # 1. Running the generate ideas function
        # --------------------------------------
        api_generate(self.p_task_id,self.p_gen_id,self.p_no_images,self.progress)


# Creating a Main global dictionary to track progress of generate ideas
# ---------------------------------------------------------------------
global generate_ideas_threads
generate_ideas_threads = {}

## MAIN function TO BE CALLED ON API
# ----------------------------------
class externalAPI_generate_ideas(Resource):

    def put(self):

        # Setting up key values to accept
        # -------------------------------
        parser = reqparse.RequestParser()
        parser.add_argument('task_id')
        parser.add_argument('gen_id')
        parser.add_argument('no_images')
        args = parser.parse_args()

        p_task_id = args['task_id']
        p_gen_id = args['gen_id']
        p_no_images = int(args['no_images'])


        # This function really just starts the create new pattern thread class and assigns it to a global dict
        # ----------------------------------------------------------------------------------------------------
        global generate_ideas_threads
        try:
            del generate_ideas_threads[p_task_id]
        except:
            'do nothing'
        generate_ideas_threads[p_task_id] = generate_ideas_threaded_task(p_task_id,p_gen_id,p_no_images)
        generate_ideas_threads[p_task_id].start()

        return 'Thread started', 200


# # running the external api functions
## MAIN function TO BE CALLED for progress
# ----------------------------------------
class externalAPI_get_progress(Resource):

    def put(self):

        # Initiating global params
        # ------------------------
        global create_texture_threads
        global new_pattern_threads
        global generate_ideas_threads

        # Setting up key values to accept
        # -------------------------------
        parser = reqparse.RequestParser()
        parser.add_argument('task_id')
        parser.add_argument('progress_for')
        args = parser.parse_args()

        p_task_id = args['task_id']
        p_progress_for = args['progress_for']

        # Using diff dicts based on request
        # ---------------------------------
        try:
            if p_progress_for == 'new_patterns':
                new_pattern_threads[p_task_id].progress['thread_status'] = new_pattern_threads[p_task_id].isAlive()
                return jsonify(new_pattern_threads[p_task_id].progress)

            elif p_progress_for == 'create_textures':
                create_texture_threads[p_task_id].progress['thread_status'] = create_texture_threads[p_task_id].isAlive()
                return jsonify(create_texture_threads[p_task_id].progress)

            elif p_progress_for == 'generate_ideas':
                generate_ideas_threads[p_task_id].progress['thread_status'] = generate_ideas_threads[p_task_id].isAlive()
                return jsonify(generate_ideas_threads[p_task_id].progress)
            else:
                return 'Invalid task decription.'

        except KeyError:
            return 'Invalid task id.', 500

# RUNNING FLASK APP
###############################################################################
#del api
app = Flask(__name__)
api = Api(app)

# Adding resource
# ---------------
api.add_resource(externalAPI_create_new_patterns, '/newpatterns') # Route
api.add_resource(externalAPI_create_textures, '/createtextures') # Route
api.add_resource(externalAPI_generate_ideas, '/generateideas') # Route
api.add_resource(externalAPI_get_progress, '/getprogress') # Route

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=8000)
