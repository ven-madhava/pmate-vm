# 30 Oct
# Added return all patterns URL API
# --------------------------------

# Imports
# -------

import numpy as np
import cv2
import random
import copy
import math
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tempfile
from tempfile import TemporaryFile
from google.cloud import storage
from tensorflow.python.lib.io import file_io
from io import BytesIO
import threading
from threading import Thread
import time
from PIL import Image,ImageFont,ImageDraw
import datetime
import string


# Necessary Flask imports
# -----------------------
from flask import Flask, request, send_file
from flask_restful import Resource, Api, reqparse
from json import dumps
from flask_jsonpify import jsonify


# # GCS functions

# In[2]:


'SWITCH BETWEEN LOCAL AND VM HERE'

global vm_or_local
vm_or_local = 'vm'

if vm_or_local == 'local':
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/venkateshmadhava/Documents/ml_projects/protomate_master/code/cloud_storage_apis/ven-ml-project-387fdf3f596f.json"
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/venkateshmadhava/ven-ml-project-387fdf3f596f.json"


# In[3]:


# Getting images from a "folder" in storage and returning that as a numpy array
# API ready ##
# -----------------------------------------------------------------------------

def get_images_from_storage(parent_dir,output_mode):

    global vm_or_local

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

        if '.jpg' in str(b.name) or '.jpeg' in str(b.name) or '.png' in str(b.name):

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


# In[4]:


# Function to saving a list or numpy array of images to storage folder
# API ready ##
# --------------------------------------------------------------------

def save_to_storage_from_array_list(x,storage_dir,image_prefix,update_progress,progress):

    global vm_or_local

    # Create a storage client to use with bucket
    # ------------------------------------------
    bucket_name = 'ven-ml-project.appspot.com'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    xin = copy.deepcopy(x)
    xin = xin.astype('uint8')
    m = len(xin)
    start_time = time.time()
    d = {}
    task_id = image_prefix[:image_prefix.index('_all_patterns')]

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

            # Making public and saving public URL
            # -----------------------------------
            if '_all_patterns' in image_prefix:
                blob.make_public()
                curr_ind = str(i)
                d[curr_ind] = {}
                d[curr_ind]['index'] = curr_ind
                d[curr_ind]['p_url'] = str(blob.public_url)


        if update_progress == True:
            curr_prog_percent = int(round((i+1)/m,2)*100)
            #progress.curr_message = str(progress.master_message) + '..about ' + str(curr_prog_percent) + '% through'
            progress.process_percent = curr_prog_percent
            # Eta params, initialise start_time
            ##
            try:
                time_counter += 1
            except:
                time_counter = 1
            curr_time = time.time()
            progress.process_start_time = start_time
            eta_remaining = telleta(start_time,curr_time,time_counter,m)
            progress.process_eta_end_time = curr_time + eta_remaining


        global vm_or_local
        if vm_or_local == 'local': print('Saving an array of size: ' + str(m) + '. Image prefix: ' + str(image_prefix) + '. Done saving image ' + str(i) + '..')


    # Saving numpy all patterns URLS and indices
    # ------------------------------------------
    if '_all_patterns' in image_prefix:
        if vm_or_local == 'local': print('Saving public URL data..')
        np_d = np.array(d)
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_all_patterns_urls.npy'
        np.save(file_io.FileIO(storage_address, 'w'), np_d)





# In[5]:


# Getting images from a "folder" in storage and returning that as a numpy array
# API ready ##
# -----------------------------------------------------------------------------

def get_images_from_storage_by_names(parent_dir,output_mode,in_names):

    global vm_or_local

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

# In[6]:


# protomate functions to get  progress eta
# API ready ##
# ----------------------------------------

def telleta(start_time,end_time,counter,m):

    # Some initialisations
    # --------------------
    time_per_iter = (end_time - start_time)/counter
    eta_left = int((m-counter) * time_per_iter)

    return eta_left

# ----------------------------------------

def printeta(eta_left):

    # Checking if eta left is in minutes or seconds
    # ---------------------------------------------
    if int(eta_left/60) >= 1: # In seconds
        if int(eta_left/60) == 1:
            minstr = 'minute'
        else:
            minstr = 'minutes'
        eta_out = str(int(eta_left/60)) + ' ' + str(minstr) + ' and ' + str(int(eta_left % 60)) + ' seconds remaining..'
    else:
        eta_out = str(int(eta_left)) + ' seconds remaining..'

    return 'Around ' + eta_out

# ----------------------------------------

# Getting key from storage
# -----------------------

def get_api_key():

    # 1. Initialising bucket details
    # ------------------------------
    bucket_name = 'ven-ml-project.appspot.com'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    destination_blob_name = 'admin/secret_key_vm_apis.txt'
    blob = bucket.blob(destination_blob_name)

    # 2. Getting content and processing
    # ---------------------------------
    key = blob.download_as_string().decode()

    return key


# In[7]:


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


# In[8]:


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




# In[9]:


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

# In[10]:


# 1
# API ready ##
# Stitch images together for blocks extraction
# --------------------------------------------

def protomatebeta_stitch_incoming_images_v1(inlist):

    # Some initialisations
    # --------------------
    global vm_or_local
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
            if len(xout) > 0: #  If all theme images are very very small and never attain max_width even after going through all images
                xout_lastim = xout[len(xout)-1]
                xout_lastim = np.concatenate((xout_lastim,xcurr), axis = 1)
                xout[len(xout)-1] = xout_lastim
            else:
                xout.append(xcurr)

        else:
            xout.append(xcurr)
    except:
        'do nothing'


    return xout


# In[11]:


# 2
# API ready ##
# Extracts blocks for building patterns
# ------------------------------------

def protomatebeta_extract_blocks_for_aop_v1(inlist,progress,ht,wd,similarity_distance=0.1):

    # Early initialisations
    # ---------------------
    global vm_or_local
    fullon_blocks_main = []
    fullon_blocks_map = []
    start_k_c = 150
    end_k_c = 150
    source_image = 'orig'
    localisation_factor = True

    # Initialisation
    # --------------
    counter = 0
    start_time = time.time()

    for i in range(len(inlist)):

        counter += 1
        img = inlist[i]
        orig_img = copy.deepcopy(img)
        if vm_or_local == 'local': print('At image ' + str(counter) + ' of around ' + str(len(inlist)) + '..')


        # 1. Smoothening images
        # ---------------------
        if vm_or_local == 'local': print('1. Smoothening image..')
        for _ in range(1): # was 5
            for _ in range(1):
                img = cv2.medianBlur(img, 5)
            for _ in range(5):
                img = cv2.edgePreservingFilter(img, flags=2, sigma_s=100, sigma_r=0.25)

        # 2. k means
        # ----------
        if vm_or_local == 'local': print('2. Applying recurring kmeans to segment..')
        kmimg,cen,dv,labels = protomatebeta_recurr_kmeans_v1(img,start_k_c,end_k_c,localisation_factor)
        if vm_or_local == 'local': plt.imshow(kmimg)
        if vm_or_local == 'local': plt.show()

        # 3. Including color picking code as well here
        # --------------------------------------------
        if vm_or_local == 'local': print('3. Picking core colors as well..')
        cen_colors = copy.deepcopy(cen)
        lb_colors = copy.deepcopy(labels)

        # 3.1 Clustering similar colors
        # -----------------------------
        if vm_or_local == 'local': print('3.1 Clustering similar colors..')
        pick_color_dict = protomatebeta_cluster_colors_v1(cen_colors,similarity_distance,False)

        # 3.2 Getting final colors for the image
        # --------------------------------------
        if vm_or_local == 'local': print('3.2 Filtering final colors..')
        fincolors = protomatebeta_getfinalcolors_v1(pick_color_dict,cen_colors,lb_colors,False,ht,wd)

        if counter == 1:
            xout_colors = fincolors
        else:
            xout_colors = np.concatenate((xout_colors,fincolors), axis = 0)


        # Updating progress
        # -----------------
        # Master message
        ##
        curr_prog_percent = int(round((i+1)/(len(inlist)),2)*100)
        #progress.curr_message = str(progress.master_message) + '..about ' + str(curr_prog_percent) + '% through'
        progress.process_percent = curr_prog_percent
        # Eta params, initialise start_time
        ##
        try:
            time_counter += 1
        except:
            time_counter = 1
        curr_time = time.time()
        progress.process_start_time = start_time
        eta_remaining = telleta(start_time,curr_time,time_counter,len(inlist))
        progress.process_eta_end_time = curr_time + eta_remaining


        # 3. Getting pattern blocks from segmented image as square blocks
        # ---------------------------------------------------------------
        if vm_or_local == 'local': print('3. Getting pattern blocks based on segmented image..')
        pt_blocks,pt_sqr_map,fullon_blocks,fullon_map = protomatebeta_cutout_blocks_v1(dv,labels,orig_img,cen,source_image) # can change orig to kmimg for placement patterns
        fullon_blocks_main  = fullon_blocks_main + fullon_blocks


    return fullon_blocks_main,xout_colors



# In[12]:


# 2.1
# API ready ##
# Function to cut out block images from clustered mood board image for block extraction
# -------------------------------------------------------------------------------------

def protomatebeta_cutout_blocks_v1(datavec,labels,image,cen,image_mode):

    # Initialistations
    # ----------------
    global vm_or_local
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





# In[13]:


# 3
# API ready ##
# Main function to build AOP patterns including height wise shift
# ---------------------------------------------------------------

def protomate_build_aop_patterns_v1(blocks,h,w,repeat_w):

    # Getting into direct iter
    # ------------------------
    global vm_or_local
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


# In[14]:


# 3.1
# API ready ##
# Function to make full images using extracted blocks for height wise shift
# --------------------------------------------------------------------------

def protomate_build_std_aop_pattern_repeat_v1(x,h,w):

    # Initialisations
    # ---------------
    global vm_or_local
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


# In[15]:


# 4
# API ready ##
# Main function that picks core colors from images
# ------------------------------------------------

def protomatebeta_pickcolors_v1(progress,inlist,ht,wd,similarity_distance=0.1):

    # Iterating through images
    # ------------------------
    global vm_or_local
    counter = 0
    start_time = time.time()

    for i in range(len(inlist)):

        counter += 1
        img = inlist[i]
        orig_img = copy.deepcopy(img)
        if vm_or_local == 'local': print('At image ' + str(counter) + ' of around ' + str(len(inlist)) + '..')

        # 1. Smoothening the image
        # ------------------------
        if vm_or_local == 'local': print('1. Smoothening image..')
        for _ in range(5):
            for _ in range(10):
                img = cv2.medianBlur(img, 5)
            for _ in range(10):
                img = cv2.edgePreservingFilter(img, flags=2, sigma_s=10, sigma_r=0.10)

        # 2. Recurring k means extractin key 25 colors
        # --------------------------------------------
        if vm_or_local == 'local': print('2. Extracting key colors using kmeans..')
        kmimg_colors,cen_colrs,dv_clrs,lb_clrs = protomatebeta_recurr_kmeans_v1(img,30,25,False)


        # 3. Clustering similar colors
        # ----------------------------
        if vm_or_local == 'local': print('3. Clustering similar colors..')
        pick_color_dict = protomatebeta_cluster_colors_v1(cen_colrs,similarity_distance,False)

        # 4. Getting final colors for the image
        # -------------------------------------
        if vm_or_local == 'local': print('4. Filtering final colors..')
        fincolors = protomatebeta_getfinalcolors_v1(pick_color_dict,cen_colrs,lb_clrs,False,ht,wd)

        # Updating progress
        # -----------------
        curr_prog_percent = int(round((i+1)/(len(inlist)),2)*100)
        #progress.curr_message = str(progress.master_message) + '..about ' + str(curr_prog_percent) + '% through'
        progress.process_percent = curr_prog_percent
        # Eta params, initialise start_time
        ##
        try:
            time_counter += 1
        except:
            time_counter = 1
        curr_time = time.time()
        progress.process_start_time = start_time
        eta_remaining = telleta(start_time,curr_time,time_counter,len(inlist))
        progress.process_eta_end_time = curr_time + eta_remaining


        if counter == 1:
            xout = fincolors
        else:
            xout = np.concatenate((xout,fincolors), axis = 0)

    return xout



# In[16]:


# 4.1
# API ready ##
# Main function that clusters similar colors for theme images
# -----------------------------------------------------------

def protomatebeta_cluster_colors_v1(raw_colors,similarity_distance,print_colors):

    # Some initial master settings
    # ----------------------------
    global vm_or_local
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




# In[17]:


# 4.2
# API ready ##
# Main function that takes in clustered dict to return final colors
# -----------------------------------------------------------------

def protomatebeta_getfinalcolors_v1(color_dict,cen,labels,print_colors,ht,wd):

    # Some initialisations
    # --------------------
    global vm_or_local
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


# In[18]:


# 5
# API ready ##
# Main function that returns textures
# -----------------------------------

def protomatebeta_build_textures_v1(x,hin,win,print_colorscale,progress,task_id,save_preview):


    # Some initial initialisations
    # ----------------------------
    global vm_or_local
    m = x.shape[0]
    cluster_threshold = 35
    start_time = time.time()

    # Iterating through images
    # ------------------------
    for i in range(m):

        if vm_or_local == 'local': print('At image ' + str(i+1) + ' of around ' + str(m) + '..')
        img = x[i]

        # 1. Bluring ops
        # --------------
        if vm_or_local == 'local': print('1. Blurring ops started..')
        for _ in range(1):
            img = cv2.medianBlur(img, 3)
        for _ in range(2):
            img = cv2.edgePreservingFilter(img, flags=2, sigma_s=10, sigma_r=0.12)

        # 2. kmeans ops
        # -------------
        if vm_or_local == 'local': print('2. Recurring k means started..')
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
        oim_stripes,oim_checks,oim_melange,oim_grainy = protomatebeta_cluster_colors_products_v1(tu,cluster_threshold,hin,win)
        oim_stripes = oim_stripes.reshape(1,oim_stripes.shape[0],oim_stripes.shape[1],3)
        oim_checks = oim_checks.reshape(1,oim_checks.shape[0],oim_checks.shape[1],3)
        if i == 0:
            out_stripes_final = oim_stripes
            out_checks_final = oim_checks
            out_mel_final = oim_melange
            out_grainy_final = oim_grainy

        else:
            out_stripes_final = np.concatenate((out_stripes_final,oim_stripes), axis = 0)
            out_checks_final = np.concatenate((out_checks_final,oim_checks), axis = 0)
            out_mel_final = np.concatenate((out_mel_final,oim_melange), axis = 0)
            out_grainy_final = np.concatenate((out_grainy_final,oim_grainy), axis = 0)


        # Updating progress
        # -----------------
        curr_prog_percent = int(round((i+1)/m,2)*100)
        #progress.curr_message = str(progress.master_message) + '..about ' + str(curr_prog_percent) + '% through'
        progress.process_percent = curr_prog_percent
        # Eta params, initialise start_time
        ##
        try:
            time_counter += 1
        except:
            time_counter = 1
        curr_time = time.time()
        progress.process_start_time = start_time
        eta_remaining = telleta(start_time,curr_time,time_counter,m)
        progress.process_eta_end_time = curr_time + eta_remaining

        if save_preview == True:
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

    return out_stripes_final.astype('uint8'), out_checks_final.astype('uint8'), out_mel_final.astype('uint8'), out_grainy_final.astype('uint8')



# In[19]:


# 5.1
# API ready ##
# Main funtion that is used to cluster similar colors returned from smaller product/pattern images
# -------------------------------------------------------------------------------------------------

def protomatebeta_cluster_colors_products_v1(tu,similarity_distance,hout,wout):

    # Some initial master settings
    # ----------------------------
    global vm_or_local
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

    outimage_stripes,outimage_checks,outimage_mel,outimage_grain = protomatebeta_create_textures_v1(to_keep_dict,weight_keep_dict,30,hout,wout)

    return outimage_stripes,outimage_checks,outimage_mel,outimage_grain


# In[20]:


# 5.2
# API ready ##
# Actual function to create stripes, checks, other textures
# ---------------------------------------------------------

def protomatebeta_create_textures_v1(tokd,wkd,repeat_h,hout,wout):

    # Here input dicts represent color scales from a single processed image
    # ---------------------------------------------------------------------
    global vm_or_local

    # 1. Getting total weight
    # -----------------------
    tot_weight = 0
    for k in wkd:
        tot_weight += wkd[k]

    # 1.1 Some initialisations for melange and grain
    # ----------------------------------------------
    mean_weight = tot_weight / (len(wkd.keys()))
    mel_grain = []

    # 2. Some initialisations
    # -----------------------
    stripe_block = np.zeros((repeat_h,wout,3))
    verticle_stripe_block = np.zeros((repeat_h,hout,3))
    st_h = 0


    # 3. Spliting the block row wise
    # ------------------------------
    for k in wkd:

        # Building a list of colors to pass to melange and grainy function
        # ----------------------------------------------------------------
        if wkd[k] >= mean_weight:
            mel_grain.append(tokd[k])


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

    # 6. Building Melange
    # -------------------
    mel_tex,grain_tex = protomatebeta_create_mel_grainy_v1(mel_grain,hout,wout)


    return built_repeat_stripes, built_repeat_checks, mel_tex, grain_tex




# In[21]:


# 5.3
# API ready ##
# Function that builds melange and grainy accepting a list of colors as tuple
# ---------------------------------------------------------------------------

def protomatebeta_create_mel_grainy_v1(inlist,h,w):

    # Accepts a list of color values as tuple and outputs full size melange and grainy blocks
    # ---------------------------------------------------------------------------------------
    global vm_or_local
    counter = 0

    for l in inlist:

        counter += 1
        if vm_or_local == 'local': print('At image ' + str(counter) + '..')

        # l of format (R,G,B). Example (202,123,34). Initialising colors
        # --------------------------------------------------------------
        orig = l
        dkthresh = 30
        dark = tuple([x-dkthresh if x-dkthresh > 0 else 0 for x in orig])

        # Building melange
        # ----------------
        mel = np.ones((h,w,3))
        spot = np.ones((h,w,3))
        mj = 0.6
        mn = 1 - mj
        width_ind = list(range(w))
        orig_count = int(mj * w)

        # Line wise melange
        # -----------------
        no_pos = orig_count
        min_len = int(0.02*mj*w)
        max_len = int(0.10*mj*w)
        len_list = list(range(min_len,max_len+1))

        # Itering through the rows and randomly setting pixels
        # ----------------------------------------------------
        for i in range(h):

            # Building occurances lengths
            # ---------------------------
            no_occurance_len = []
            no_occurance_counter = 0
            while True:
                #print('At 1st while..')
                oc = random.choice(len_list)
                if no_occurance_counter + oc < no_pos:
                    no_occurance_counter += oc
                    no_occurance_len.append(oc)
                else:
                    oc = no_pos - no_occurance_counter
                    no_occurance_counter += oc
                    no_occurance_len.append(oc)
                    break

            # Determining orig color positions
            # --------------------------------
            len_list_current = copy.deepcopy(width_ind)
            sub_lists = []
            for j in no_occurance_len:

                # Looping to randomly find a position
                # -----------------------------------
                while True:
                    #print('At 2nd while..')
                    start_postion = random.choice(len_list_current)
                    if start_postion + j > max(len_list_current):
                        'do nothing and repeat loop'
                    else:
                        break

                # We now have a start position that occupies legitimate indices
                # -------------------------------------------------------------
                curr_sub_list = list(range(start_postion,start_postion+j))
                sub_lists += curr_sub_list
                len_list_current = [x for x in len_list_current if x not in curr_sub_list]

            #print('Out of loops')
            # 'Set'ing the lists
            # -------------------
            orig_positions_mel = list(set(sub_lists))
            dark_positions_mel = list(set(len_list_current))
            mel[i,[orig_positions_mel]] = orig
            mel[i,[dark_positions_mel]] = dark

            # Spot texture
            # -------------
            mj_spot = 0.85
            orig_count_spot = int(mj_spot * w)
            orig_positions_spot = list(random.sample(width_ind,orig_count_spot))
            dark_positions_spot = [x for x in width_ind if x not in orig_positions_spot]
            spot[i,[orig_positions_spot]] = orig
            spot[i,[dark_positions_spot]] = dark

        if counter == 1:
            melout = mel.reshape(1,mel.shape[0],mel.shape[1],mel.shape[2])
            spotout = spot.reshape(1,spot.shape[0],spot.shape[1],spot.shape[2])
        else:
            melout = np.concatenate((melout,mel.reshape(1,mel.shape[0],mel.shape[1],mel.shape[2])), axis = 0)
            spotout = np.concatenate((spotout,spot.reshape(1,spot.shape[0],spot.shape[1],spot.shape[2])), axis = 0)

    return melout.astype('uint8'), spotout.astype('uint8')


# In[22]:


# 6
# Getting lines,segments & categories from storage and returning them as a numpy array and list
# API ready ##
# ---------------------------------------------------------------------------------------------

def get_stylings_from_storage(in_names,update_progress,progress):

    global vm_or_local

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
    len_namelist = len(in_names_list)
    start_time = time.time()

    for i in in_names_list:

        #print('At image..' + str(i))
        #print(i)
        #print(type(i))

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
            #print('Added!')
            if update_progress == True:
                curr_prog_percent = int(round(counter/len_namelist,2)*100)
                #progress.curr_message = str(progress.master_message) + '..about ' + str(curr_prog_percent) + '% through'
                progress.process_percent = curr_prog_percent
                # Eta params, initialise start_time
                ##
                try:
                    time_counter += 1
                except:
                    time_counter = 1
                curr_time = time.time()
                progress.process_start_time = start_time
                eta_remaining = telleta(start_time,curr_time,time_counter,len_namelist)
                progress.process_eta_end_time = curr_time + eta_remaining



    return xout_lines,xout_seg,categories


# In[23]:


# 6.1
# API ready ##
# Function to correct segments and linemarkings
# ----------------------------------------------

def protomatebeta_correct_segments_linemarkings(lines,seg):

    # Initialisations
    # ---------------
    global vm_or_local
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
        bl_d = 100
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


# In[24]:


# 7
# API read ##
# Main function that generates ideas
# ----------------------------------

def protomatebeta_create_ideas_v2(segments_in,linemarkings_in,categories_in,patterns,stripes,checks,melange,grainy,colors,progress,task_id,gen_id,save_preview,no_options):



    # Repeating segments, linemarkings & cats for no_options
    # ------------------------------------------------------
    global vm_or_local
    for _ in range(int(no_options)):
        try:
            segments = np.concatenate((segments,segments_in), axis = 0)
            linemarkings = np.concatenate((linemarkings,linemarkings_in), axis = 0)
            categories = np.concatenate((categories,categories_in), axis = 0)
        except:
            segments = copy.deepcopy(segments_in)
            linemarkings = copy.deepcopy(linemarkings_in)
            categories = copy.deepcopy(categories_in)

    # Initialisations
    # ---------------
    seg_index = list(range(segments.shape[0]))
    start_time = time.time()

    # Setting no_images to number of input stylings
    # ---------------------------------------------
    no_images = segments.shape[0]

    m = segments.shape[0]
    cats_out = []

    inh_r = segments.shape[1]
    inw_r = segments.shape[2]
    list_combos = []

    # tup will be always of form (seg_index,pattern_index,color_index,stripes_index,check_index,melange_index,grainy_index)
    # Will use 0 in case of specific categories or single segments

    # Iterating for number of required output
    # ---------------------------------------
    for i in range(no_images):

        if vm_or_local == 'local': print('Generating image..' + str(i+1))

        # First getting the seg_index to use for evenly displaying generations
        # --------------------------------------------------------------------
        if i < m:
            s_index = i
        else:
            s_index = int(i%m)

        # Picking segments & patterns
        # ---------------------------
        if i == 0:

            tup = None

            # Setting segment
            # ---------------
            segc = segments[s_index]
            category_curr_seg = categories[s_index,0]
            cats_out.append(category_curr_seg)
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
            single_segment_threshold = 0.05

            # Using external function to return combos
            # ----------------------------------------
            if np.sum(temp_grey_pos) == 0 or np.sum(temp_black_pos) == 0 or grey_fac < single_segment_threshold or black_fac < single_segment_threshold: ## Single segment ##

                # return combo function
                # ----------------------
                gblock,bblock,tup = returncombo(True,False,None,category_curr_seg,s_index,patterns,stripes,checks,melange,grainy,colors)

            else:

                if grey_fac < porp_threshold: ## Minority segment - gray ##
                    gblock,bblock,tup = returncombo(False,True,'gray',category_curr_seg,s_index,patterns,stripes,checks,melange,grainy,colors)

                elif black_fac < porp_threshold: ## Minority segment - black ##
                    gblock,bblock,tup = returncombo(False,True,'black',category_curr_seg,s_index,patterns,stripes,checks,melange,grainy,colors)

                else: ## Equal segments ##
                    gblock,bblock,tup = returncombo(False,False,None,category_curr_seg,s_index,patterns,stripes,checks,melange,grainy,colors)


            # Appending to list tup
            # ---------------------
            list_combos.append(tup)

        else:
            start_time_break_loop = time.time()

            while True:

                tup = None

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

                # Using external function to return combos
                # ----------------------------------------
                if np.sum(temp_grey_pos) == 0 or np.sum(temp_black_pos) == 0: ## Single segment ##

                    # return combo function
                    # ----------------------
                    gblock,bblock,tup = returncombo(True,False,None,category_curr_seg,s_index,patterns,stripes,checks,melange,grainy,colors)

                else:

                    if grey_fac < porp_threshold: ## Minority segment - gray ##
                        gblock,bblock,tup = returncombo(False,True,'gray',category_curr_seg,s_index,patterns,stripes,checks,melange,grainy,colors)

                    elif black_fac < porp_threshold: ## Minority segment - black ##
                        gblock,bblock,tup = returncombo(False,True,'black',category_curr_seg,s_index,patterns,stripes,checks,melange,grainy,colors)

                    else: ## Equal segments ##
                        gblock,bblock,tup = returncombo(False,False,None,category_curr_seg,s_index,patterns,stripes,checks,melange,grainy,colors)

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
                    cats_out.append(category_curr_seg)
                    break

                # Code that times out this while loop incase it goes over board
                # -------------------------------------------------------------
                curr_time = time.time()
                if (curr_time - start_time_break_loop) > 8: # 8 secs
                    return genout,cats_out  # Returns genout and exits


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
        #progress.curr_message = str(progress.master_message) + '..about ' + str(curr_prog_percent) + '% through'
        progress.process_percent = curr_prog_percent
        # Eta params, initialise start_time
        ##
        try:
            time_counter += 1
        except:
            time_counter = 1
        curr_time = time.time()
        progress.process_start_time = start_time
        eta_remaining = telleta(start_time,curr_time,time_counter,no_images)
        progress.process_eta_end_time = curr_time + eta_remaining

        # Saving current generation to storage for keeping frontend progress
        # -------------------------------------------------------------------
        if save_preview == True:
            storage_dir = str(task_id) + '/ideas/ideaspreview'
            image_prefix = str(task_id) + '_' + str(gen_id) + '_preview'
            save_to_storage_from_array_list(newim_c,storage_dir,image_prefix,False,None)

        if i == 0:
            genout = newim_c
        else:
            genout = np.concatenate((genout,newim_c), axis = 0)

    return genout , cats_out


# In[25]:


# 7.1
# API read ##
# Function that return combos for generating ideas
# ------------------------------------------------

#Girls  Womens
#Cat#   Cat#
#0	    20	Woven Dresses
#1	    21	Knit Dresses
#2	    22	Woven Tops
#3	    23	Knit Tops
#4	    24	Knit Polos
#5	    25	Woven Jumpers
#6	    26	Knit Jumpers
#7	    27	Woven Pants & Capris
#8	    28	Knit Pants & Capris
#9	    29	Knit Leggings
#10	    30	Woven Shorts
#11	    31	Knit Shorts
#12	    32	Woven Skirts & Skorts

#Boys   Mens
#Cat#   Cat#
#13	    33	Knit Polos
#14	    34	Woven Shirts
#15	    35	Jeans
#16	    36	Woven Pants
#17	    37	Knit Pants
#18	    38	Woven Shorts
#19	    39	Knit Shorts

#tup will be always of form --
# (seg_index,pattern_index,color_index,stripes_index,check_index,melange_index,grainy_index)

def returncombo(single_segment,minor_segment,minor_segment_seg,category,s_index,patterns,stripes,checks,melange,grainy,colors):

    global vm_or_local

    # Some initialisations
    # --------------------
    patterns_index = list(range(patterns.shape[0]))
    stripes_index = list(range(stripes.shape[0]))
    checks_index = list(range(checks.shape[0]))
    melange_index = list(range(melange.shape[0]))
    grainy_index = list(range(grainy.shape[0]))
    colors_index = list(range(colors.shape[0]))

    # Legend
    # ------
    # 0 -- Use print
    # 1 -- Use stripes
    # 2 -- Use checks
    # 3 -- Use Melange
    # 4 -- Use grainy
    # 5 -- Use colors
    choice_choices = [0,1,2,3,4,5]

    # Choice making code
    # ------------------

    ### Woven tops, jumpsuits, dresses - GIRLS and WOMENS
    if category == 0 or category == 20 or category == 2 or category == 22 or category == 5 or category == 25:

        # Setting choice probabilities
        # ----------------------------
        ch_0 = 0.7 # print
        ch_1 = 0.1 # stripes
        ch_2 = 0.2 # checks
        ch_3 = 0.0 # melange
        ch_4 = 0.0 # grainy
        ch_5 = 0.0 # colors
        choice_probs = [ch_0,ch_1,ch_2,ch_3,ch_4,ch_5]

        # Single segment check
        # --------------------
        if single_segment == True:
            choice_single = int(np.random.choice(choice_choices, p = choice_probs))
            choice_g = choice_single
            choice_b = choice_single

        elif minor_segment == True:
            if minor_segment_seg == 'black':
                choice_g = int(np.random.choice(choice_choices, p = choice_probs))
                choice_b = 5
            else:
                choice_g = 5
                choice_b = int(np.random.choice(choice_choices, p = choice_probs))

        else:
            choice_g = int(np.random.choice(choice_choices, p = choice_probs))

            # Setting bblock based on gblock choice
            # -------------------------------------
            if choice_g == 0:
                choice_b = random.choice([4,5])
            elif choice_g == 1:
                choice_b = random.choice([4,5])
            elif choice_g == 2:
                choice_b = random.choice([4,5])
            else:
                choice_b = int(np.random.choice(choice_choices, p = choice_probs))

    ### Knit tops, jumpsuits, dresses - GIRLS and WOMENS
    elif category == 1 or category == 21 or category == 3 or category == 23 or category == 6 or category == 26:

        # Setting choice probabilities
        # ----------------------------
        ch_0 = 0.5 # print
        ch_1 = 0.3 # stripes
        ch_2 = 0.0 # checks
        ch_3 = 0.1 # melange
        ch_4 = 0.0 # grainy
        ch_5 = 0.1 # colors
        choice_probs = [ch_0,ch_1,ch_2,ch_3,ch_4,ch_5]

        # Single segment check
        # --------------------
        if single_segment == True:
            choice_single = int(np.random.choice(choice_choices, p = choice_probs))
            choice_g = choice_single
            choice_b = choice_single

        elif minor_segment == True:
            if minor_segment_seg == 'black':
                choice_g = int(np.random.choice(choice_choices, p = choice_probs))
                choice_b = random.choice([3,5])
            else:
                choice_g = random.choice([3,5])
                choice_b = int(np.random.choice(choice_choices, p = choice_probs))

        else:
            choice_g = int(np.random.choice(choice_choices, p = choice_probs))

            # Setting bblock based on gblock choice
            # -------------------------------------
            if choice_g == 0 or choice_g == 1:
                choice_b = random.choice([3,5])
            else:
                choice_b = int(np.random.choice(choice_choices, p = choice_probs))


    ### Knit shorts, pants, leggings - GIRLS and WOMENS | Knit shorts and pant - BOYS and MENS
    elif category == 8 or category == 28 or category == 9 or category == 29 or category == 11 or category == 31 or category == 17 or category == 37 or category == 19 or category == 39:

        # Setting choice probabilities
        # ----------------------------
        ch_0 = 0.3 # print
        ch_1 = 0.3 # stripes
        ch_2 = 0.0 # checks
        ch_3 = 0.3 # melange
        ch_4 = 0.0 # grainy
        ch_5 = 0.1 # colors
        choice_probs = [ch_0,ch_1,ch_2,ch_3,ch_4,ch_5]

        # Single segment check
        # --------------------
        if single_segment == True:
            choice_single = int(np.random.choice(choice_choices, p = choice_probs))
            choice_g = choice_single
            choice_b = choice_single

        elif minor_segment == True:
            if minor_segment_seg == 'black':
                choice_g = int(np.random.choice(choice_choices, p = choice_probs))
                choice_b = random.choice([3,5])
            else:
                choice_g = random.choice([3,5])
                choice_b = int(np.random.choice(choice_choices, p = choice_probs))

        else:
            choice_g = int(np.random.choice(choice_choices, p = choice_probs))

            # Setting bblock based on gblock choice
            # -------------------------------------
            if choice_g == 0 or choice_g == 1:
                choice_b = random.choice([3,5])
            else:
                choice_b = int(np.random.choice(choice_choices, p = choice_probs))

    ### Knit Polo -- Boys, Girls, Mens, Women
    elif category == 4 or category == 24 or category == 13 or category == 33:

        # Setting choice probabilities
        # ----------------------------
        ch_0 = 0.2 # print
        ch_1 = 0.5 # stripes
        ch_2 = 0.0 # checks
        ch_3 = 0.2 # melange
        ch_4 = 0.0 # grainy
        ch_5 = 0.1 # colors
        choice_probs = [ch_0,ch_1,ch_2,ch_3,ch_4,ch_5]

        # Single segment check
        # --------------------
        if single_segment == True:
            choice_single = int(np.random.choice(choice_choices, p = choice_probs))
            choice_g = choice_single
            choice_b = choice_single

        elif minor_segment == True:
            if minor_segment_seg == 'black':
                choice_g = int(np.random.choice(choice_choices, p = choice_probs))
                choice_b = random.choice([3,5])
            else:
                choice_g = random.choice([3,5])
                choice_b = int(np.random.choice(choice_choices, p = choice_probs))

        else:
            choice_g = int(np.random.choice(choice_choices, p = choice_probs))

            # Setting bblock based on gblock choice
            # -------------------------------------
            if choice_g == 0 or choice_g == 1:
                choice_b = random.choice([3,5])
            else:
                choice_b = int(np.random.choice(choice_choices, p = choice_probs))

    ### Woven pants, shorts, skirts, shirts - across all depts
    elif category == 7 or category == 27 or category == 15 or category == 35 or category == 16 or category == 36 or category == 10 or category == 30 or category == 12 or category == 32 or category == 18 or category == 38 or category == 14 or category == 34:

        # Setting choice probabilities
        # ----------------------------
        ch_0 = 0.4 # print
        ch_1 = 0.1 # stripes
        ch_2 = 0.4 # checks
        ch_3 = 0.0 # melange
        ch_4 = 0.0 # grainy
        ch_5 = 0.1 # colors
        choice_probs = [ch_0,ch_1,ch_2,ch_3,ch_4,ch_5]

        # Single segment check
        # --------------------
        if single_segment == True:
            choice_single = int(np.random.choice(choice_choices, p = choice_probs))
            choice_g = choice_single
            choice_b = choice_single

        elif minor_segment == True:
            if minor_segment_seg == 'black':
                choice_g = int(np.random.choice(choice_choices, p = choice_probs))
                choice_b = 5
            else:
                choice_g = 5
                choice_b = int(np.random.choice(choice_choices, p = choice_probs))

        else:
            choice_g = int(np.random.choice(choice_choices, p = choice_probs))

            # Setting bblock based on gblock choice
            # -------------------------------------
            if choice_g == 0 or choice_g == 1 or choice_g == 2:
                choice_b = 5
            else:
                choice_b = int(np.random.choice(choice_choices, p = choice_probs))


    # Index & tup setting code - gindex
    # ---------------------------------
    choice = choice_g
    if choice == 0:
        g_index = random.choice(patterns_index)
        gblock = patterns[g_index]
        tup = (s_index,g_index,0,0,0,0,0)
    elif choice == 1:
        g_index = random.choice(stripes_index)
        gblock = stripes[g_index]
        tup = (s_index,0,0,g_index,0,0,0)
    elif choice == 2:
        g_index = random.choice(checks_index)
        gblock = checks[g_index]
        tup = (s_index,0,0,0,g_index,0,0)
    elif choice == 3:
        g_index = random.choice(melange_index)
        gblock = melange[g_index]
        tup = (s_index,0,0,0,0,g_index,0)
    elif choice == 4:
        g_index = random.choice(grainy_index)
        gblock = grainy[g_index]
        tup = (s_index,0,0,0,0,0,g_index)
    else:
        g_index = random.choice(colors_index)
        gblock = colors[g_index]
        tup = (s_index,0,g_index,0,0,0,0)

    # Index & tup setting code - bindex
    # ---------------------------------
    choice = choice_b
    if choice == 0:
        b_index = random.choice(patterns_index)
        bblock = patterns[b_index]
        tup = (s_index,b_index,0,0,0,0,0)
    elif choice == 1:
        b_index = random.choice(stripes_index)
        bblock = stripes[b_index]
        tup = (s_index,0,0,b_index,0,0,0)
    elif choice == 2:
        b_index = random.choice(checks_index)
        bblock = checks[b_index]
        tup = (s_index,0,0,0,b_index,0,0)
    elif choice == 3:
        b_index = random.choice(melange_index)
        bblock = melange[b_index]
        tup = (s_index,0,0,0,0,b_index,0)
    elif choice == 4:
        b_index = random.choice(grainy_index)
        bblock = grainy[b_index]
        tup = (s_index,0,0,0,0,0,b_index)
    else:
        b_index = random.choice(colors_index)
        bblock = colors[b_index]
        tup = (s_index,0,b_index,0,0,0,0)


    return gblock, bblock, tup


# In[26]:


# 8
# API read ##
# Function that returns board ranges
# ----------------------------------


def feed_to_build_range(x,cats,task_id,gen_id,board_name,styling_prefix,no_ideas_per_row=8,no_total_rows=4):

    global vm_or_local

    # Some preps
    # ----------
    all_counter = 0
    c_counter = 0
    cats_as_np = np.array(cats).reshape(len(cats), 1)
    all_unique_cats = list(set(cats))
    cat_dict = {}
    cat_dict[0] = 'GIRLS WOVEN DRESSES'
    cat_dict[1] = 'GIRLS KNIT DRESSES'
    cat_dict[2] = 'GIRLS WOVEN TOPS'
    cat_dict[3] = 'GIRLS KNIT TOPS'
    cat_dict[4] = 'GIRLS POLOS'
    cat_dict[5] = 'GIRLS WOVEN JUMPSUITS'
    cat_dict[6] = 'GIRLS KNIT JUMPSUITS'
    cat_dict[7] = 'GIRLS WOVEN PANTS & CAPRIS'
    cat_dict[8] = 'GIRLS KNIT PANTS & CAPRIS'
    cat_dict[9] = 'GIRLS LEGGINGS'
    cat_dict[10] = 'GIRLS WOVEN SHORTS'
    cat_dict[11] = 'GIRLS KNIT SHORTS'
    cat_dict[12] = 'GIRLS WOVEN SKIRTS & SKORTS'
    cat_dict[13] = 'BOYS POLOS'
    cat_dict[14] = 'BOYS WOVEN SHIRTS'
    cat_dict[15] = 'BOYS JEANS'
    cat_dict[16] = 'BOYS WOVEN PANTS'
    cat_dict[17] = 'BOYS KNIT PANTS'
    cat_dict[18] = 'BOYS WOVEN SHORTS'
    cat_dict[19] = 'BOYS KNIT SHORTS'
    cat_dict[20] = 'WOMENS WOVEN DRESSES'
    cat_dict[21] = 'WOMENS KNIT DRESSES'
    cat_dict[22] = 'WOMENS WOVEN TOPS'
    cat_dict[23] = 'WOMENS KNIT TOPS'
    cat_dict[24] = 'WOMENS POLOS'
    cat_dict[25] = 'WOMENS WOVEN JUMPSUITS'
    cat_dict[26] = 'WOMENS KNIT JUMPSUITS'
    cat_dict[27] = 'WOMENS WOVEN PANTS & CAPRIS'
    cat_dict[28] = 'WOMENS KNIT PANTS & CAPRIS'
    cat_dict[29] = 'WOMENS LEGGINGS'
    cat_dict[30] = 'WOMENS WOVEN SHORTS'
    cat_dict[31] = 'WOMENS KNIT SHORTS'
    cat_dict[32] = 'WOMENS WOVEN SKIRTS & SKORTS'
    cat_dict[33] = 'MENS POLOS'
    cat_dict[34] = 'MENS WOVEN SHIRTS'
    cat_dict[35] = 'MENS JEANS'
    cat_dict[36] = 'MENS WOVEN PANTS'
    cat_dict[37] = 'MENS KNIT PANTS'
    cat_dict[38] = 'MENS WOVEN SHORTS'
    cat_dict[39] = 'MENS KNIT SHORTS'

    sty_dict = {}
    sty_dict[0] = 'GWDR'
    sty_dict[1] = 'GKDR'
    sty_dict[2] = 'GWTP'
    sty_dict[3] = 'GKTP'
    sty_dict[4] = 'GPL'
    sty_dict[5] = 'GWJP'
    sty_dict[6] = 'GKJP'
    sty_dict[7] = 'GWPC'
    sty_dict[8] = 'GKPC'
    sty_dict[9] = 'GLG'
    sty_dict[10] = 'GWST'
    sty_dict[11] = 'GKST'
    sty_dict[12] = 'GWSK'
    sty_dict[13] = 'BPL'
    sty_dict[14] = 'BWSH'
    sty_dict[15] = 'BJ'
    sty_dict[16] = 'BWP'
    sty_dict[17] = 'BKP'
    sty_dict[18] = 'BWST'
    sty_dict[19] = 'BKST'
    sty_dict[20] = 'WWDR'
    sty_dict[21] = 'WKDR'
    sty_dict[22] = 'WWTP'
    sty_dict[23] = 'WKTP'
    sty_dict[24] = 'WPL'
    sty_dict[25] = 'WWJP'
    sty_dict[26] = 'WKJP'
    sty_dict[27] = 'WWPC'
    sty_dict[28] = 'WKPC'
    sty_dict[29] = 'WLG'
    sty_dict[30] = 'WWST'
    sty_dict[31] = 'WKST'
    sty_dict[32] = 'WWSK'
    sty_dict[33] = 'MPL'
    sty_dict[34] = 'MWSH'
    sty_dict[35] = 'MJ'
    sty_dict[36] = 'MWP'
    sty_dict[37] = 'MKP'
    sty_dict[38] = 'MWST'
    sty_dict[39] = 'MKST'




    # Itering through unique categories to build range boards
    # -------------------------------------------------------
    for c in all_unique_cats:

        c_counter += 1
        if vm_or_local == 'local': print('At category ' + str(c_counter) + ' of around ' + str(len(all_unique_cats)))

        # 1. Getting indices of ideas belonging to a particular cat
        # ---------------------------------------------------------
        ind_curr_c = list(np.argwhere(cats_as_np[:,0] == c)[:,0])

        # 2. Getting all ideas belonging to this cat
        # ------------------------------------------
        curr_cat_ideas = x[ind_curr_c]

        # 3. Some paging ops
        # ------------------
        no_pages = math.ceil(curr_cat_ideas.shape[0]/(no_ideas_per_row*no_total_rows))

        # 4. Calling single range build function iteratively
        # --------------------------------------------------
        for i in range(no_pages):

            # 4.1 Setting curr input
            # ----------------------
            try:
                curr_xin_for_single_range = curr_cat_ideas[i*no_ideas_per_row*no_total_rows:i*no_ideas_per_row*no_total_rows + no_ideas_per_row*no_total_rows]
            except:
                curr_xin_for_single_range = curr_cat_ideas[i*no_ideas_per_row*no_total_rows:]

            # 4.2 Calling function
            # --------------------
            all_counter += 1
            curr_styling_prefix = styling_prefix + ' ' + sty_dict[c]

            curr_board_out = build_single_range_board(curr_xin_for_single_range,task_id,gen_id,board_name,curr_styling_prefix,cat_dict[c],i+1,no_pages,no_ideas_per_row,no_total_rows)
            if all_counter == 1:
                boardout = curr_board_out
            else:
                boardout = np.concatenate((boardout,curr_board_out), axis = 0)

        if vm_or_local == 'local': print('Done')



    return boardout






# In[27]:


# 8.1
# API read ##
# Main function to build a single board
# -------------------------------------


def build_single_range_board(xin,task_id,gen_id,board_name,styling_prefix,board_header,curr_page_number,total_page_numbers,no_ideas_per_row,no_total_rows):

    global vm_or_local

    # Initialising a fixed board size based on input image dimensions
    # ---------------------------------------------------------------
    # xin will only have a max of no_ideas_per_row *no_total_rows images
    # Feeding function must make sure of this
    # ------------------------------------------------------------------
    #h,w = int(285/2),int(221/2) # Hardcoded for now
    h,w = 285,221 # Hardcoded for now

    # Local
    # -----
    global vm_or_local
    if vm_or_local == 'local':
        font_file_path_header = '/Users/venkateshmadhava/Documents/ml_projects/protomate_master/code/fonts/Kodchasan-Bold.ttf'
        font_file_path_labels = '/Users/venkateshmadhava/Documents/ml_projects/protomate_master/code/fonts/RobotoCondensed-Bold.ttf'
        font_file_path_footer = '/Users/venkateshmadhava/Documents/ml_projects/protomate_master/code/fonts/RobotoMono-Light.ttf'

    # VM
    # --
    else:
        font_file_path_header = '/home/venkateshmadhava/pmate-vm/Kodchasan-Bold.ttf'
        font_file_path_labels = '/home/venkateshmadhava/pmate-vm/RobotoCondensed-Bold.ttf'
        font_file_path_footer = '/home/venkateshmadhava/pmate-vm/RobotoMono-Light.ttf'


    # Header and footer initialisations
    # ---------------------------------
    main_header_height = 25
    font_header_main = ImageFont.truetype(font_file_path_header, size=main_header_height, encoding="unic")
    gap_between_header_ideas = 20

    right_header_height = 10
    font_header_right = ImageFont.truetype(font_file_path_header, size=right_header_height, encoding="unic")

    all_footer_height = 10
    font_footer = ImageFont.truetype(font_file_path_footer, size=all_footer_height, encoding="unic")
    gap_between_footer_ideas = 20

    # Labels
    # ------
    font_height = 12
    font = ImageFont.truetype(font_file_path_labels, size=font_height, encoding="unic")
    naming_prefix = styling_prefix + ' #'

    # 1. Some initialisations
    # -----------------------
    gap_between_ideas_label = 10
    gap_between_ideas_v = 60
    gap_between_ideas_h = 40
    outer_padding_v = 25
    outer_padding_h = 50



    # 2. Inside board dimensions Just to place images
    # -----------------------------------------------
    ideas_only_dim_h = (h + font_height + gap_between_ideas_label) * no_total_rows + (no_total_rows - 1) * gap_between_ideas_h + 2 * outer_padding_h + main_header_height + gap_between_header_ideas * 2 + all_footer_height + gap_between_footer_ideas * 2
    ideas_only_dim_w = w * no_ideas_per_row + (no_ideas_per_row - 1) * gap_between_ideas_v + 2 * outer_padding_v

    # 3. Filling ideas within the board
    # ---------------------------------
    ideas_board = (np.ones((ideas_only_dim_h,ideas_only_dim_w,3))*255).astype('uint8')
    ideas_cords = []
    label_cords = []

    # 4. Getting ideas co-ordinates dynamically
    # --------------------------------------
    for r in range(1,no_total_rows+1):
        for c in range(1,no_ideas_per_row+1):

            # Getting row co-ordinate
            # -----------------------
            if r == 1: # First row
                curr_rs = outer_padding_h + main_header_height + gap_between_header_ideas * 2
            else:
                curr_rs = outer_padding_h + (r-1)*(h + font_height + gap_between_ideas_label + gap_between_ideas_h) + main_header_height + gap_between_header_ideas * 2

            # Getting col co-ordinate
            # -----------------------
            if c == 1: # First idea
                curr_cs = outer_padding_v
            else:
                curr_cs = outer_padding_v + (c-1) * (w + gap_between_ideas_v)

            # Appending to list
            # -----------------
            ideas_cords.append((curr_rs,curr_cs))

    # 5. Attaching ideas
    # -----------------
    idea_at = {}
    for i in range(no_ideas_per_row * no_total_rows):
        try:
            # Attaching image
            # ---------------
            curr_np_img = cv2.resize(xin[i], dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            ideas_board[ideas_cords[i][0]:ideas_cords[i][0]+h,ideas_cords[i][1]:ideas_cords[i][1]+w,:] = curr_np_img
            idea_at[i] = True

        except:
            idea_at[i] = False
            'do nothing'

    # 6. Attaching labels
    # -------------------
    ideas_board_img = Image.fromarray(ideas_board)
    draw = ImageDraw.Draw(ideas_board_img)

    for i in range(no_ideas_per_row * no_total_rows):

        if idea_at[i] ==  True:

            # Attaching label
            # ---------------
            curr_label = naming_prefix + str((curr_page_number-1) * no_ideas_per_row * no_total_rows + i+1)
            curr_label_w = font.getsize(curr_label)[0]

            # Working out centers
            # -------------------
            center_c = ideas_cords[i][1] + int(w/2)
            start_c = center_c - int(curr_label_w/2)
            start_r = ideas_cords[i][0] + h + gap_between_ideas_label

            # Drawing text
            # ------------
            draw.text((start_c,start_r),curr_label, fill=(100,100,100), font=font)

    # 7. Attaching header left
    # -------------------------
    header_text = board_name + ' / ' + board_header
    header_start_c = outer_padding_v
    header_start_r = gap_between_header_ideas
    draw.text((header_start_c,header_start_r),header_text, fill=(50,50,50), font=font_header_main)

    # 8. Attaching header right
    # -------------------------
    header_r_text = str(curr_page_number) + '/' + str(total_page_numbers)
    header_r_w = font_header_right.getsize(header_r_text)[0]
    header_r_w += outer_padding_v
    header_r_start_c = ideas_only_dim_w - header_r_w - 5
    header_r_start_r = int((main_header_height + gap_between_header_ideas * 2)/2) - int(all_footer_height/2)
    draw.text((header_r_start_c,header_r_start_r),header_r_text, fill=(0,0,0), font=font_footer)

    # Footer texts initialisations
    # ----------------------------
    time_block = datetime.datetime.now().strftime("Created on %Y-%m-%d at %H:%M for task ")
    footer_left_text = time_block + str(task_id) + ', generation ' + str(gen_id) + '.'
    footer_right_text = 'Powered By Protomate'
    footer_start_row_in_image = ideas_only_dim_h - (all_footer_height + gap_between_footer_ideas * 2)
    footer_start_r =footer_start_row_in_image + gap_between_footer_ideas

    # 9. Attaching footer left
    # ---------------------
    footer_start_c = outer_padding_v
    draw.text((footer_start_c,footer_start_r),footer_left_text, fill=(0,0,0), font=font_footer)

    # 10. Attaching footer right
    # --------------------------
    footer_right_w = font_footer.getsize(footer_right_text)[0]
    footer_right_w += outer_padding_v
    footer_r_start_c = ideas_only_dim_w - footer_right_w
    draw.text((footer_r_start_c,footer_start_r),footer_right_text, fill=(0,0,0), font=font_footer)

    # 11. Returning Single range board in (1,h,w,3)
    # ---------------------------------------------
    boardout_np = np.array(ideas_board_img)
    boardout_np = boardout_np.reshape(1,boardout_np.shape[0],boardout_np.shape[1],3)

    return boardout_np



# # Ven_API functions

# In[28]:


# API 1 Function
# API -- create_new_patterns
# creates new patterns for the first time using uploaded theme images at
# /task_id/themes/ and saves to /task_id/all_patterns/
# Creates all_patterns, colors, all_maps numpy arrays and stores them at /task_id/numpy/
# Creates segments, linemarkings, categories from input selected styling names and saves as numpy
# at /task_id/numpy for easy generation
# Returns OK, NOT OK


def api_create_new_patterns(task_id,selected_style_names,progress):

    global vm_or_local

    # progress is a class object of class progress for that task_id
    ##

    #try:

    # Setting input folder name as per set format
    # -------------------------------------------
    inputfolder = task_id + '/themes'

    # Standard initialisations
    # ------------------------
    h,w,rp_size = 285,221,30

    try:
        # 1. Getting theme images from storage
        # ------------------------------------
        progress.set_status(0) # 0 'Loading theme images..'
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        theme_list = get_images_from_storage(inputfolder,'list')
        # 2. Stitching images together as list
        # ------------------------------------
        themes_stitched = protomatebeta_stitch_incoming_images_v1(theme_list)
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500


    try:
        # 3. Extracting blocks from stitched images
        # -----------------------------------------
        progress.set_status(1) # 1 'Initialising AI learning. This may take a while (around 15 seconds per image)..'
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        flblocks,keycolors = protomatebeta_extract_blocks_for_aop_v1(themes_stitched,progress,h,w)
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500


    try:
        # 4. Building patterns
        # --------------------
        progress.set_status(2) # 2 'Building patterns based on learnt objects from theme images..',
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        built_patterns = protomate_build_aop_patterns_v1(flblocks,h,w,rp_size)
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500


    try:
        # 7. Saving all patterns to storage for front end retrieval
        # ---------------------------------------------------------
        progress.set_status(3) # 3 'Saving built patterns for user selection..',
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        storage_dir = task_id + '/all_patterns'
        image_prefix = str(task_id) + '_all_patterns'
        save_to_storage_from_array_list(built_patterns,storage_dir,image_prefix,True,progress)
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500


    try:
        progress.set_status(4) # 4 'Saving internal files for generation..',
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        # 8. Saving numpy formats of all patterns, all maps, all colors
        # -------------------------------------------------------------
        # 8.1 Saving all patterns
        # -----------------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_all_patterns.npy'
        np.save(file_io.FileIO(storage_address, 'w'), built_patterns)
        # 8.2 Saving all colors
        # -----------------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_all_colors.npy'
        np.save(file_io.FileIO(storage_address, 'w'), keycolors)
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500


    try:
        progress.set_status(5) # 5 'Preparing selected stylings for generations..',
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        # 9. Picking selected stylings and saving them into temp arrays for correction
        # ----------------------------------------------------------------------------
        x_lines,x_segs,cats = get_stylings_from_storage(selected_style_names,True,progress)
        # 9.1. Correcting incoming lines and segments
        # -----------------------------------------
        xl_corr,xs_corr = protomatebeta_correct_segments_linemarkings(x_lines,x_segs)
        # 10. Saving lines, segs and categories under /task_id/numpy for generation
        # -------------------------------------------------------------------------
        # Saving Segments
        # ---------------
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
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500

    # Saving status update to 1
    # -------------------------
    np_status = np.array([1]).reshape(1,1)
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_status.npy'
    np.save(file_io.FileIO(storage_address, 'w'), np_status)

    progress.set_status(6) # Done
    progress.process_start_time = None
    progress.process_eta_end_time = None
    progress.process_percent = None
    progress.runnning_status = 'Finished.'

    # Delecting appropriate dicts to allow user to run threaded task again
    # --------------------------------------------------------------------
    global progress_api_dict
    global create_texture_threads
    global new_pattern_threads
    global generate_ideas_threads

    time.sleep(10) # for 10 secs, status will be available after task completes

    try:
        del new_pattern_threads[task_id]
        del progress_api_dict[task_id]
    except:
        'do nothing'


    return 'All good.', 200


# In[29]:


# API 2 function
# API -- create_textures
# Takes in task_id, selected_indices (as string) and creates textures
# saving them under /task_id/numpy/ And also saved picked indices as numpy array
# At same location
# Returns OK, NOT OK


def api_create_textures(task_id,picked_ind_string,progress):

    global vm_or_local

    # progress is a class object of class progress for that task_id
    ##

    #try:

    try:
        progress.set_status(0) # 0 'Loading learnt patterns..'
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        # 1. Converting input indices to string
        # -------------------------------------
        picked_ind = [int(s) for s in picked_ind_string.split(',')]
        # 2. Getting all patterns npy file
        # --------------------------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_all_patterns.npy'
        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
        all_pats = np.load(f)
        picked_patterns = all_pats[picked_ind]
        h,w = picked_patterns.shape[1],picked_patterns.shape[2]
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500


    try:
        progress.set_status(1) # 1 'Building textures..',
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        # 3. Creating stripes and checks
        # ------------------------------
        picked_stripes,picked_checks,picked_melange,picked_grainy = protomatebeta_build_textures_v1(picked_patterns,h,w,False,progress,task_id,False)
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500


    try:
        progress.set_status(2) # 2 'Saving textures..',
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        # 4. Saving textures under /task_id/numpy/
        # ----------------------------------------
        # Saving Stripes
        # --------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_stripes.npy'
        np.save(file_io.FileIO(storage_address, 'w'), picked_stripes)
        # Saving Checks
        # --------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_checks.npy'
        np.save(file_io.FileIO(storage_address, 'w'), picked_checks)
        # Saving Melange
        # --------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_melange.npy'
        np.save(file_io.FileIO(storage_address, 'w'), picked_melange)
        # Saving Grainy
        # --------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_grainy.npy'
        np.save(file_io.FileIO(storage_address, 'w'), picked_grainy)
        # Saving Picked_ind
        # -----------------
        picked_ind_np = np.array(picked_ind).reshape(len(picked_ind), 1)
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_picked_ind.npy'
        np.save(file_io.FileIO(storage_address, 'w'), picked_ind_np)
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500

    # Saving status update to 2
    # -------------------------
    np_status = np.array([2]).reshape(1,1)
    storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_status.npy'
    np.save(file_io.FileIO(storage_address, 'w'), np_status)

    progress.set_status(3) # Done
    progress.process_start_time = None
    progress.process_eta_end_time = None
    progress.process_percent = None
    progress.runnning_status = 'OK'

    # Delecting appropriate dicts to allow user to run threaded task again
    # --------------------------------------------------------------------
    global progress_api_dict
    global create_texture_threads
    global new_pattern_threads
    global generate_ideas_threads

    time.sleep(10) # for 10 secs, status will be available after task completes

    try:
        del create_texture_threads[task_id]
        del progress_api_dict[task_id]
    except:
        'do nothing'

    return 'All good.', 200

    #except Exception as ex:
    #    error_str = type(ex).__name__ + ': ' + ex.args[0]
    #    return error_str, 500


# In[30]:


# API 3 function
# API -- generate ideas
# Takes in task_id, gen_id as inputs
# Generates new ideas and saves them under /task_id/ideas/gen_id/
# Returns OK, NOT OK


def api_generate(task_id,gen_id,task_board_name,task_styling_name_prefix,progress,no_options):

    global vm_or_local

    #try:

    # progress is a class object of class progress for that task_id
    ##

    try:
        progress.set_status(0) # 0 'Loading stylings..',
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        # 1. Collect required numpy files and load them locally for generation
        # --------------------------------------------------------------------
        # Collecting linemarkings
        # -----------------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_linemarkings.npy'
        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
        lines = np.load(f)
        if vm_or_local == 'local': print('Got lines..')
        # Collecting segments
        # -------------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_segments.npy'
        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
        segs = np.load(f)
        if vm_or_local == 'local': print('Got segs..')
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500



    try:
        progress.set_status(1) # 1 'Loading patterns..',
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        # Collecting all pats
        # -------------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_all_patterns.npy'
        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
        all_patterns = np.load(f)
        if vm_or_local == 'local': print('Got all pats..')
        # Collecting picked_ind
        # ---------------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_picked_ind.npy'
        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
        picked_ind = np.load(f)
        if vm_or_local == 'local': print('Got picked indices..')
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500



    try:
        progress.set_status(2) # 2 'Loading colors..',
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        # Collecting colors
        # ---------------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_all_colors.npy'
        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
        colors = np.load(f)
        if vm_or_local == 'local': print('Got colors..')
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500



    try:
        progress.set_status(3) # 3 'Loading textures..',
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        # Colecting checks
        # ----------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_checks.npy'
        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
        checks = np.load(f)
        if vm_or_local == 'local': print('Got checks..')
        # Collecting Stripes
        # -----------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_stripes.npy'
        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
        stripes = np.load(f)
        if vm_or_local == 'local': print('Got stripes..')
        # Collecting Melange
        # -----------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_melange.npy'
        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
        melange = np.load(f)
        if vm_or_local == 'local': print('Got melange..')
        # Collecting Grainy
        # -----------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_grainy.npy'
        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
        grainy = np.load(f)
        if vm_or_local == 'local': print('Got grainy..')
        # Collecting Catagories
        # ----------------------
        storage_address = 'gs://ven-ml-project.appspot.com/' + str(task_id) + '/numpy/np_categories.npy'
        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
        categories = np.load(f)
        if vm_or_local == 'local': print('Got categories..')
        # 2. Preparing picked patterns for generation
        # -------------------------------------------
        picked_patterns = all_patterns[list(picked_ind[:,0])]
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500



    try:
        progress.set_status(4) # 4 'Generating ideas..',
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        # 3. Actual generation
        # --------------------
        ideas,cats = protomatebeta_create_ideas_v2(segs,lines,categories,picked_patterns,stripes,checks,melange,grainy,colors,progress,task_id,gen_id,False,no_options)
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500



    #try:
    #    progress.set_status(5) # 5 'Saving ideas..',
    #    progress.process_start_time = None
    #    progress.process_eta_end_time = None
    #    progress.process_percent = None
    #    # 4. Saving generated images under /task_id/ideas/gen_id/
    #    # -------------------------------------------------------
    #    storage_dir = task_id + '/ideas/' + str(gen_id)
    #    image_prefix = str(task_id) + '_' + str(gen_id) + '_ideas'
    #    save_to_storage_from_array_list(ideas,storage_dir,image_prefix,True,progress)
    #    progress.runnning_status = 'OK'
    #except Exception as ex:
    #    error_str = type(ex).__name__ + ': ' + ex.args[0]
    #    progress.runnning_status = 'ERROR! ' + error_str
    #    return error_str, 500



    try:
        progress.set_status(5) # 5 'Building and saving rangeboards..',
        progress.process_start_time = None
        progress.process_eta_end_time = None
        progress.process_percent = None
        # 5. Building and saving range boards under /task_id/rangeboards/gen_id/
        # ----------------------------------------------------------------------
        range_built = feed_to_build_range(ideas,cats,task_id,gen_id,task_board_name,task_styling_name_prefix)
        storage_dir = task_id + '/rangeboards/' + str(gen_id)
        image_prefix = str(task_id) + '_' + str(gen_id) + '_rangeboards'
        save_to_storage_from_array_list(range_built,storage_dir,image_prefix,True,progress)
        # 6. Building PDF for download
        # ----------------------------
        range_list = []
        for range_i in range(range_built.shape[0]):
            curr_im = Image.fromarray(range_built[range_i])
            range_list.append(curr_im)
        # 6.1
        # ---
        bucket_name = 'ven-ml-project.appspot.com'
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        with tempfile.NamedTemporaryFile() as temp:
            # Set name to the temp file
            # -------------------------
            pdf_name = ''.join([str(temp.name),'.pdf'])
            # Save PDF to temp file
            # -----------------------
            range_list[0].save(pdf_name, "PDF" ,resolution=100.0, save_all=True, append_images=range_list[1:])
            # Storing the image temp file inside the bucket
            # ---------------------------------------------
            destination_blob_name = storage_dir + '/downloadable_range_boards.pdf'
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(pdf_name,content_type='application/pdf')
        progress.runnning_status = 'OK'
    except Exception as ex:
        error_str = type(ex).__name__ + ': ' + ex.args[0]
        progress.runnning_status = 'ERROR! ' + error_str
        progress.curr_step = None
        return error_str, 500

    progress.set_status(6) # Done
    progress.process_start_time = None
    progress.process_eta_end_time = None
    progress.process_percent = None
    progress.runnning_status = 'OK'

    # Delecting appropriate dicts to allow user to run threaded task again
    # --------------------------------------------------------------------
    global progress_api_dict
    global create_texture_threads
    global new_pattern_threads
    global generate_ideas_threads

    time.sleep(10) # for 10 secs, status will be available after task completes

    try:
        del generate_ideas_threads[task_id]
        del progress_api_dict[task_id]
    except:
        'do nothing'



    return 'All good.', 200




# # Actual Ven API endpoints

# In[31]:


# Creating a global progress dict to help with simpler progress API
# -----------------------------------------------------------------

global progress_api_dict
progress_api_dict = {}


# In[32]:


# Temp code to create progress class
##

class progress_classobj():

    def __init__(self,task_id,at_step):
        super().__init__()

        # Initialisations
        # ---------------
        self.task_id = task_id
        self.at_step = at_step

        # A single mode progress call
        # ---------------------------
        if self.at_step == 1: # Create new patterns

            # Initialising progress updates
            # -----------------------------
            self.all_progress_updates = [
                'Loading theme images..',
                'Initialising AI learning. This may take a while (around 15 seconds per image)..',
                'Building patterns based on learnt objects from theme images..',
                'Saving built patterns for user selection..',
                'Saving internal files for generation..',
                'Preparing selected stylings for generations..',
                'Done.']

        elif self.at_step == 2: # Create textures

            # Initialising progress updates
            # -----------------------------
            self.all_progress_updates = [
                'Loading learnt patterns..',
                'Building textures..',
                'Saving textures..',
                'Done.']

        elif self.at_step == 3: # Generate ideas

            # Initialising progress updates
            # -----------------------------
            self.all_progress_updates = [
                'Loading stylings..',
                'Loading patterns..',
                'Loading colors..',
                'Loading textures..',
                'Generating ideas..',
                'Building and saving rangeboards..',
                'Done.']

        # External message params initialisations
        # ---------------------------------------
        self.curr_step = 0
        self.total_step = len(self.all_progress_updates) - 1
        self.curr_message = 'Yet to start.'
        self.runnning_status = 'Yet to start.'
        self.process_start_time = None
        self.process_eta_end_time = None
        self.process_percent = None



    def set_status(self, at_now):

        # Setting message status
        # ----------------------
        self.curr_step = at_now
        self.curr_message = self.all_progress_updates[at_now]

    def dict_out(self):

        # To return params for progress API
        # ---------------------------------
        d = {}
        d['mode_number'] = self.at_step

        if self.at_step == 1:
            d['curr_mode'] = 'Creating new patterns'
        elif self.at_step == 2:
            d['curr_mode'] = 'Creating textures'
        else:
            d['curr_mode'] = 'Generating ideas'

        if self.process_percent == None:
            d['curr_process_percent'] = 'Process Percent Not Available'
        else:
            d['curr_process_percent'] = self.process_percent

        d['curr_step'] = self.curr_step
        d['total_step'] = self.total_step
        d['curr_message'] = self.curr_message

        d['runnning_status'] = self.runnning_status

        return d


    def getallprogressstatuses(self):
        # To return all progress statuses for front end display
        # -----------------------------------------------------
        d = {}
        for i in range(len(self.all_progress_updates)):
            d[i] = self.all_progress_updates[i]

        return d


# ### 1. create new patterns external API

# In[33]:


##

class create_new_patterns_threaded_task(threading.Thread):
    def __init__(self,p_task_id,p_selected_style_names):
        super().__init__()

        # Initialisations
        # ---------------
        self.p_task_id = p_task_id
        self.p_selected_style_names = p_selected_style_names
        self.progress = progress_classobj(p_task_id,1)

        # For progress api
        # ----------------
        global progress_api_dict
        try:
            del progress_api_dict[p_task_id]
        except:
            'do nothing'

        progress_api_dict[p_task_id] = self.progress


    def run(self): # Run method probably overrides the inherited Threads run method

        # 1. Running the create new patterns function
        # -------------------------------------------
        api_create_new_patterns(self.p_task_id,self.p_selected_style_names,self.progress)


# In[34]:


# Creating a Main global dictionary to track progress of create new pattern task
# ------------------------------------------------------------------------------
global new_pattern_threads
new_pattern_threads = {}


# In[35]:


## MAIN function TO BE CALLED ON API
# ----------------------------------

class externalAPI_create_new_patterns(Resource):

    def post(self):

        ## Authenticating request
        ## ----------------------
        try:

            # Get stored key
            # --------------
            vm_api_key = get_api_key()

            try:

                api_key = request.args['api_key']

                if api_key == vm_api_key:

                    # Authorized request
                    # ------------------

                    # Setting up key values to accept
                    # -------------------------------
                    parser = reqparse.RequestParser()
                    parser.add_argument('task_id')
                    parser.add_argument('selected_style_names')
                    args = parser.parse_args()

                    p_task_id = args['task_id']
                    p_selected_style_names = args['selected_style_names']

                    # First trying to get np_status
                    # -----------------------------
                    try:
                        storage_address = 'gs://ven-ml-project.appspot.com/' + str(p_task_id) + '/numpy/np_status.npy'
                        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))

                        # This means patterns already built
                        # ---------------------------------
                        return 'Invalid operation. Patterns already built for this task.', 500 #### Checked

                    except:

                        # This means there is no such file and new pattern build can begin.
                        # This function really just starts the create new pattern thread class and assigns it to a global dict
                        # ----------------------------------------------------------------------------------------------------
                        global new_pattern_threads
                        try:
                            # Checking for parallel active thread
                            # -----------------------------------
                            if new_pattern_threads[p_task_id].progress.curr_step >= 0:
                                return 'Invalid operation. Parallel operation already running. Check via progress API.', 500 #### Checked
                            else:
                                return 'Something went wrong, check progress for error.', 500 #### Checked
                        except:
                            print('API Create new patterns firing: ' + str(p_task_id))
                            try:
                                del new_pattern_threads[p_task_id]
                            except:
                                'do nothing'

                            new_pattern_threads[p_task_id] = create_new_patterns_threaded_task(p_task_id,p_selected_style_names)
                            new_pattern_threads[p_task_id].start()
                            return 'Thread started', 200 #### Checked

                else:

                    # Incorrect credentials
                    # ---------------------
                    return 'Incorrect credentials', 401
            except:

                # Invalid headers
                # ---------------
                return 'Invalid credentails', 400

        except:

            # Secret key not set in storage
            # -----------------------------
            return 'API keys not initialsed', 401




# ### 2. create new texture external API

# In[36]:


##

class create_textures_threaded_task(threading.Thread):
    def __init__(self,p_task_id,p_picked_ind_string):
        super().__init__()

        # Initialisations
        # ---------------
        self.p_task_id = p_task_id
        self.p_picked_ind_string = p_picked_ind_string
        self.progress = progress_classobj(p_task_id,2)

        # For progress api
        # ----------------
        global progress_api_dict
        try:
            del progress_api_dict[p_task_id]
        except:
            'do nothing'

        progress_api_dict[p_task_id] = self.progress

    def run(self): # Run method probably overrides the inherited Threads run method

        # 1. Running the create textures function
        # -------------------------------------------
        api_create_textures(self.p_task_id,self.p_picked_ind_string,self.progress)



# In[37]:


# Creating a Main global dictionary to track progress of create textures task
# ---------------------------------------------------------------------------
global create_texture_threads
create_texture_threads = {}


# In[38]:


## MAIN function TO BE CALLED ON API for creating texture
# -------------------------------------------------------

class externalAPI_create_textures(Resource):

    def post(self):

        ## Authenticating request
        ## ----------------------
        try:

            # Get stored key
            # --------------
            vm_api_key = get_api_key()

            try:

                api_key = request.args['api_key']

                if api_key == vm_api_key:

                    # Authorized request
                    # ------------------

                    # Setting up key values to accept
                    # -------------------------------
                    parser = reqparse.RequestParser()
                    parser.add_argument('task_id')
                    parser.add_argument('picked_ind_string')
                    args = parser.parse_args()

                    p_task_id = args['task_id']
                    p_picked_ind_string = args['picked_ind_string']

                    # First trying to get np_status
                    # -----------------------------
                    global create_texture_threads

                    try:
                        storage_address = 'gs://ven-ml-project.appspot.com/' + str(p_task_id) + '/numpy/np_status.npy'
                        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
                        npstatus = np.load(f)[0,0]

                        if npstatus == 1: # Good to go

                            try:
                                # Checking for parallel active thread
                                # -----------------------------------
                                if create_texture_threads[p_task_id].progress.curr_step >= 0:
                                    return 'Invalid operation. Parallel operation already running. Check via progress API.', 500 ## Checked
                                else:
                                    return 'Something went wrong, check progress for error.', 500
                            except:
                                print('API Create Texture firing: ' + str(p_task_id))
                                try:
                                    del create_texture_threads[p_task_id]
                                except:
                                    'do nothing'
                                create_texture_threads[p_task_id] = create_textures_threaded_task(p_task_id,p_picked_ind_string)
                                create_texture_threads[p_task_id].start()
                                return 'Thread started', 200

                        elif npstatus == 2:
                            return 'Invalid operation. Textures already built for this task.', 500 #### Checked
                        else:
                            err_msg = 'Invalid operation. Something not right about the flow. Here is the npstatus: ' + str(npstatus)
                            return err_msg, 500

                    except:

                        return 'Invalid operation. Looks like patterns not built for this task.', 500 #### Checked
                else:

                    # Incorrect credentials
                    # ---------------------
                    return 'Incorrect credentials', 401
            except:

                # Invalid headers
                # ---------------
                return 'Invalid credentails', 400

        except:

            # Secret key not set in storage
            # -----------------------------
            return 'API keys not initialsed', 401



# ### 3. generate ideas external API

# In[39]:


##

class generate_ideas_threaded_task(threading.Thread):
    def __init__(self,p_task_id,p_gen_id,p_task_board_name,p_task_styling_prefix,p_no_options):
        super().__init__()

        # Initialisations
        # ---------------
        self.p_task_id = p_task_id
        self.p_gen_id = p_gen_id
        self.p_task_board_name = p_task_board_name
        self.p_task_styling_prefix = p_task_styling_prefix
        self.progress = progress_classobj(p_task_id,3)
        self.p_no_options = p_no_options

        # For progress api
        # ----------------
        global progress_api_dict
        try:
            del progress_api_dict[p_task_id]
        except:
            'do nothing'

        progress_api_dict[p_task_id] = self.progress

    def run(self): # Run method probably overrides the inherited Threads run method

        # 1. Running the generate ideas function
        # --------------------------------------
        api_generate(self.p_task_id,self.p_gen_id,self.p_task_board_name,self.p_task_styling_prefix,self.progress,self.p_no_options)




# In[40]:


# Creating a Main global dictionary to track progress of generate ideas
# ---------------------------------------------------------------------
global generate_ideas_threads
generate_ideas_threads = {}


# In[41]:


## MAIN function TO BE CALLED ON API
# ----------------------------------

class externalAPI_generate_ideas(Resource):

    def post(self):

        ## Authenticating request
        ## ----------------------
        try:

            # Get stored key
            # --------------
            vm_api_key = get_api_key()

            try:

                api_key = request.args['api_key']

                if api_key == vm_api_key:

                    # Authorized request
                    # ------------------

                    # Setting up key values to accept
                    # -------------------------------
                    parser = reqparse.RequestParser()
                    parser.add_argument('task_id')
                    parser.add_argument('gen_id')
                    parser.add_argument('task_board_name')
                    parser.add_argument('task_styling_name_prefix')
                    parser.add_argument('no_options')
                    args = parser.parse_args()

                    p_task_id = args['task_id']
                    p_gen_id = args['gen_id']
                    p_task_board_name = args['task_board_name']
                    p_task_styling_name_prefix = args['task_styling_name_prefix']
                    p_no_options = args['no_options']


                    # This function really just starts the create new pattern thread class and assigns it to a global dict
                    # ----------------------------------------------------------------------------------------------------
                    global generate_ideas_threads

                    try:
                        storage_address = 'gs://ven-ml-project.appspot.com/' + str(p_task_id) + '/numpy/np_status.npy'
                        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
                        npstatus = np.load(f)[0,0]

                        if npstatus == 1:
                            return 'Invalid operation. Textures not built for this task.', 500 ## Checked
                        elif npstatus == 2:
                            try:
                                # Checking for parallel thread
                                # ----------------------------
                                if generate_ideas_threads[p_task_id].progress.curr_step >= 0:
                                    return 'Invalid operation. Parallel operation already running. Check via progress API.', 500 #### Checked
                                else:
                                    return 'Something went wrong, check progress for error.', 500
                            except:
                                print('API Generate Ideas firing: ' + str(p_task_id))
                                try:
                                    del generate_ideas_threads[p_task_id]
                                except:
                                    'do nothing'
                                generate_ideas_threads[p_task_id] = generate_ideas_threaded_task(p_task_id,p_gen_id,p_task_board_name,p_task_styling_name_prefix,p_no_options)
                                generate_ideas_threads[p_task_id].start()
                                return 'Thread started', 200
                    except:
                        return 'Invalid operation. Looks like patterns are not built for this task yet.', 500 #### Checked
                else:

                    # Incorrect credentials
                    # ---------------------
                    return 'Incorrect credentials', 401
            except:

                # Invalid headers
                # ---------------
                return 'Invalid credentails', 400

        except:

            # Secret key not set in storage
            # -----------------------------
            return 'API keys not initialsed', 401



# ### 4. progress and status APIs

# In[42]:


## MAIN function TO BE CALLED for all progress status updates associated with progress of a task
# ----------------------------------------------------------------------------------------------

class externalAPI_get_all_progress_updates(Resource):

    def post(self):

        ## Authenticating request
        ## ----------------------
        try:

            # Get stored key
            # --------------
            vm_api_key = get_api_key()

            try:

                api_key = request.args['api_key']

                if api_key == vm_api_key:

                    # Authorized request
                    # ------------------

                    # For progress api
                    # ----------------
                    global progress_api_dict

                    # Setting up key values to accept
                    # -------------------------------
                    parser = reqparse.RequestParser()
                    parser.add_argument('task_id')
                    args = parser.parse_args()
                    p_task_id = args['task_id']

                    # Returning data
                    # --------------
                    print('API Get All Progress Status firing: ' + str(p_task_id))
                    try:
                        d = progress_api_dict[p_task_id].getallprogressstatuses()
                        return jsonify(d)

                    except KeyError:
                        return 'Invalid task id.', 500
                else:

                    # Incorrect credentials
                    # ---------------------
                    return 'Incorrect credentials', 401
            except:

                # Invalid headers
                # ---------------
                return 'Invalid credentails', 400
        except:

            # Secret key not set in storage
            # -----------------------------
            return 'API keys not initialsed', 401



# In[43]:


## MAIN function TO BE CALLED for progress
# ----------------------------------------

class externalAPI_get_progress(Resource):

    def post(self):

        ## Authenticating request
        ## ----------------------
        try:

            # Get stored key
            # --------------
            vm_api_key = get_api_key()

            try:

                api_key = request.args['api_key']

                if api_key == vm_api_key:

                    # Authorized request
                    # ------------------

                    # For progress api
                    # ----------------
                    global progress_api_dict
                    global create_texture_threads
                    global new_pattern_threads
                    global generate_ideas_threads

                    # Setting up key values to accept
                    # -------------------------------
                    parser = reqparse.RequestParser()
                    parser.add_argument('task_id')
                    args = parser.parse_args()
                    p_task_id = args['task_id']

                    print('API Get Progress firing: ' + str(p_task_id))
                    # Returning data
                    # --------------
                    try:
                        d = progress_api_dict[p_task_id].dict_out()

                        # Updating thread status
                        # ----------------------
                        if d['mode_number'] == 1:
                            d['thread_status'] = new_pattern_threads[p_task_id].isAlive()
                        elif d['mode_number'] == 2:
                            d['thread_status'] = create_texture_threads[p_task_id].isAlive()
                        else:
                            d['thread_status'] = generate_ideas_threads[p_task_id].isAlive()

                        # Updating time left
                        # ------------------
                        try:
                            curr_time = time.time()

                            if curr_time > progress_api_dict[p_task_id].process_eta_end_time:
                                d['curr_process_time_remaining'] = 'Finishing up anytime..'
                            else:
                                curr_time_left = progress_api_dict[p_task_id].process_eta_end_time - curr_time
                                d['curr_process_time_remaining'] = printeta(curr_time_left)

                        except:
                            d['curr_process_time_remaining'] = 'ETA Not Available'

                        # Returning d
                        # -----------
                        return jsonify(d)

                    except KeyError:
                        return 'Invalid task id.', 500
                else:

                    # Incorrect credentials
                    # ---------------------
                    return 'Incorrect credentials', 401
            except:

                # Invalid headers
                # ---------------
                return 'Invalid credentails', 400

        except:

            # Secret key not set in storage
            # -----------------------------
            return 'API keys not initialsed', 401



# In[44]:


## MAIN function TO BE CALLED for task status
# -------------------------------------------

class externalAPI_get_task_status(Resource):

    def post(self):

        ## Authenticating request
        ## ----------------------
        try:

            # Get stored key
            # --------------
            vm_api_key = get_api_key()

            try:

                api_key = request.args['api_key']

                if api_key == vm_api_key:

                    # Authorized request
                    # ------------------

                    # Setting up key values to accept
                    # -------------------------------
                    parser = reqparse.RequestParser()
                    parser.add_argument('task_id')
                    args = parser.parse_args()
                    p_task_id = args['task_id']

                    d = {}
                    print('API Get Task Status firing: ' + str(p_task_id))
                    try:
                        storage_address = 'gs://ven-ml-project.appspot.com/' + str(p_task_id) + '/numpy/np_status.npy'
                        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
                        npstatus = np.load(f)[0,0]

                        if npstatus == 1: # Patterns done, but textures NOT done.
                            d['task_status'] = 'ok for textures'
                            return jsonify(d) ## Checked

                        elif npstatus == 2: # Textures done and can generate ideas.
                            d['task_status'] = 'ok for generation'
                            return jsonify(d) ## Checked
                    except:
                        d['task_status'] = 'Error. Either no such task or no patterns built.'
                        return jsonify(d) #### Checked
                else:

                    # Incorrect credentials
                    # ---------------------
                    return 'Incorrect credentials', 401
            except:

                # Invalid headers
                # ---------------
                return 'Invalid credentails', 400

        except:

            # Secret key not set in storage
            # -----------------------------
            return 'API keys not initialsed', 401






# In[45]:


## MAIN function TO BE CALLED for download pdf
# --------------------------------------------

class externalAPI_send_range(Resource):

    def post(self):

        ## Authenticating request
        ## ----------------------
        try:

            # Get stored key
            # --------------
            vm_api_key = get_api_key()

            try:

                api_key = request.args['api_key']

                if api_key == vm_api_key:

                    # Authorized request
                    # ------------------

                    # Setting up key values to accept
                    # -------------------------------
                    parser = reqparse.RequestParser()
                    parser.add_argument('task_id')
                    parser.add_argument('gen_id')
                    args = parser.parse_args()

                    # Getting params
                    # --------------
                    p_task_id = args['task_id']
                    p_gen_id = args['gen_id']

                    print('API Download PDF firing: ' + str(p_task_id))
                    # Returning file
                    # --------------
                    try:
                        storage_address = 'gs://ven-ml-project.appspot.com/' + str(p_task_id) + '/rangeboards/' + str(p_gen_id) + '/downloadable_range_boards.pdf'
                        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
                        return send_file(f, mimetype='application/pdf')
                    except:
                        return 'Could not find range PDF. Invalid URL.', 500
                else:

                    # Incorrect credentials
                    # ---------------------
                    return 'Incorrect credentials', 401
            except:

                # Invalid headers
                # ---------------
                return 'Invalid credentails', 400

        except:

            # Secret key not set in storage
            # -----------------------------
            return 'API keys not initialsed', 401



# In[46]:


## MAIN function TO BE CALLED for download pdf
# --------------------------------------------

class externalAPI_get_all_patterns_url(Resource):

    def post(self):

        ## Authenticating request
        ## ----------------------
        try:

            # Get stored key
            # --------------
            vm_api_key = get_api_key()

            try:

                api_key = request.args['api_key']

                if api_key == vm_api_key:

                    # Authorized request
                    # ------------------

                    # Setting up key values to accept
                    # -------------------------------
                    parser = reqparse.RequestParser()
                    parser.add_argument('task_id')
                    args = parser.parse_args()

                    # Getting params
                    # --------------
                    p_task_id = args['task_id']

                    print('API get all patterns public URL firing: ' + str(p_task_id))

                    # Returning JSON
                    # --------------
                    try:

                        # Loading dict
                        # ------------
                        storage_address = 'gs://ven-ml-project.appspot.com/' + str(p_task_id) + '/numpy/np_all_patterns_urls.npy'
                        f = BytesIO(file_io.read_file_to_string(storage_address, binary_mode=True))
                        np_d = np.load(f)
                        d = np_d.item()

                        return jsonify(d)

                    except:

                        return 'Internal error occured. Sorry!', 500

                else:

                    # Incorrect credentials
                    # ---------------------
                    return 'Incorrect credentials', 401
            except:

                # Invalid headers
                # ---------------
                return 'Invalid credentails', 400

        except:

            # Secret key not set in storage
            # -----------------------------
            return 'API keys not initialsed', 401



# # running the external api functions

# In[47]:


app = Flask(__name__)
api = Api(app)

# Adding resource
# ---------------
api.add_resource(externalAPI_create_new_patterns, '/newpatterns') # Route
api.add_resource(externalAPI_create_textures, '/createtextures') # Route
api.add_resource(externalAPI_generate_ideas, '/generateideas') # Route
api.add_resource(externalAPI_get_progress, '/getprogress') # Route
api.add_resource(externalAPI_get_all_progress_updates, '/getallprogressstatuses') # Route
api.add_resource(externalAPI_get_task_status, '/gettaskstatus') # Route
api.add_resource(externalAPI_send_range, '/getrange') # Route
api.add_resource(externalAPI_get_all_patterns_url, '/getallpatternsurl') # Route


# In[48]:


global vm_or_local
if __name__ == '__main__':
    if vm_or_local == 'local':
        app.run(port='5002') # For local
    else:
        app.run(host='0.0.0.0', port=8000) # VM
