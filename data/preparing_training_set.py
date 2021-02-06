import os
import sys
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pandas import json_normalize

import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 10000000000
import tifffile as tiff

from utils import enc2mask, mask2enc


parser = argparse.ArgumentParser(description="This python script is used to check the maximum pixel size of glomerulus, and crop around the glomerulus with the desired pixel size.")
parser.add_argument("-n", "--numPixel", 
                    action="store",
                    dest="imgSize",
                    default=700, 
                    type=int, 
                    help="the crop size of square centered at the glomerulus")
parser.add_argument("-s", "--save",
                    action="store_true",
                    dest="saveImage", 
                    default=False,
                    help="crop and store the glomerulus images")
parser.add_argument("-p", "--path",
                    dest="root_dir", 
                    default="",
                    help="the (root) directory in which the dataset is stored",
                    required=True)

args = parser.parse_args()

if( not os.path.isdir(args.root_dir) ):
    sys.exit("Error: No path found!")
elif( not os.path.isfile(os.path.join(args.root_dir,"HuBMAP-20-dataset_information.csv")) ):
    sys.exit("Error: Check your path!")

### read the metadata file and check which images are for training
df_global_metadata_reader=pd.read_csv(os.path.join(args.root_dir,"HuBMAP-20-dataset_information.csv"), header=0)
df_global_metadata_reader["for_training"]=[ os.path.isfile( os.path.join(args.root_dir,"train",local_metadatafile) ) for local_metadatafile in df_global_metadata_reader['glomerulus_segmentation_file'] ]

### read the pixel-level mask of the training dataset
df_masks=pd.read_csv(os.path.join(args.root_dir,"train.csv"), header=0)

### By looping over each image,
### determine the max width and height of all glomerulus
### and save the crop images if requested
gmax_width=-1
gmax_height=-1

for idx in range(len(df_global_metadata_reader['glomerulus_segmentation_file'])): 
    # skipping the testing images
    if df_global_metadata_reader.loc[idx]['for_training']==False:
        continue

    # extracting the image filename
    image_id=df_global_metadata_reader.loc[idx]['image_file'].replace('.tiff','')
    
    # read the image and convert it to an array of H x W x C
    local_tifffile_i=df_global_metadata_reader.loc[idx]['image_file']
    img = tiff.imread(os.path.join(args.root_dir,"train",local_tifffile_i))
    if (len(img.shape) == 5): 
        img = np.transpose(img.squeeze(), (1,2,0))
    global_num_hpixel=img.shape[0]
    global_num_wpixel=img.shape[1]

    # read the metadata file (for the polygon of the glomerulus)
    local_metadatafile_i=df_global_metadata_reader.loc[idx]['glomerulus_segmentation_file']
    with open(os.path.join(args.root_dir,"train",local_metadatafile_i)) as f:
        json_data=json.load(f)
    df_metadata_glomerulus=json_normalize(json_data)
    list_coordinates=df_metadata_glomerulus["geometry.coordinates"].to_numpy()

    mask=enc2mask(df_masks.loc[df_masks['id']==image_id]['encoding'].item(),(global_num_wpixel,global_num_hpixel))

    cnt=0
    for j in range(len(list_coordinates)):
        # reading the max and min coordinates for glomerulus "j"
        min_x=-1
        max_x=-1
        min_y=-1
        max_y=-1
        for i in list_coordinates[j][0]:
            if(min_x==-1 or i[0]<min_x):
                min_x=i[0]
            if(max_x==-1 or i[0]>max_x):
                max_x=i[0]
            if(min_y==-1 or i[1]<min_y):
                min_y=i[1]
            if(max_y==-1 or i[1]>max_y):
                max_y=i[1]
        width=max_x-min_x
        height=max_y-min_y

        # update the maximum height and width
        if(gmax_width==-1 or gmax_width<(width)):
            gmax_width=width
        if(gmax_height==-1 or gmax_height<(height)):
            gmax_height=height

        if(args.saveImage):
            os.system("rm -rf {}; mkdir -p {}".format(os.path.join(args.root_dir,"processed/train"),os.path.join(args.root_dir,"processed/train")))
            os.system("rm -rf {}; mkdir -p {}".format(os.path.join(args.root_dir,"processed/mask"),os.path.join(args.root_dir,"processed/mask")))

            lower_h=max(0,min_y-((args.imgSize-height)//2-1))
            upper_h=min(global_num_hpixel,max_y+(args.imgSize-(height))//2)
            lower_w=max(0,min_x-((args.imgSize-width)//2-1))
            upper_w=min(global_num_wpixel,max_x+(args.imgSize-(width))//2)
            if(lower_h==0 and upper_h-lower_h+1<args.imgSize):
                lower_h=(args.imgSize-1)
            elif(upper_h==global_num_hpixel and upper_h-lower_h+1<args.imgSize):
                upper_h=global_num_hpixel-(args.imgSize-1)
            elif(upper_h-lower_h+1==(args.imgSize-2)):
                lower_h-=1
                upper_h+=1
            elif(upper_h-lower_h+1==(args.imgSize-1)):
                lower_h-=1
            elif(upper_h-lower_h+1<args.imgSize):
                sys.exit("ERROR: Some special case not considered for image cropping")
            elif(upper_h-lower_h+1>args.imgSize):
                sys.exit("ERROR: exceeding max image height dimension: H={}".format(upper_h-lower_h+1))   
            if(lower_w==0 and upper_w-lower_w+1<args.imgSize):
                lower_w=(args.imgSize-1)
            elif(upper_w==global_num_hpixel and upper_w-lower_w+1<args.imgSize):
                upper_w=global_num_hpixel-(args.imgSize-1)
            elif(upper_w-lower_w+1==(args.imgSize-2)):
                lower_w-=1
                upper_w+=1
            elif(upper_w-lower_w+1==(args.imgSize-1)):
                lower_w-=1
            elif(upper_w-lower_w+1<args.imgSize):
                sys.exit("ERROR: Some special case not considered for image cropping")
            elif(upper_w-lower_w+1>args.imgSize):
                sys.exit("ERROR: exceeding max image width dimension: W={}".format(upper_w-lower_w+1))
            
            img_crop=img[lower_h:upper_h+1,lower_w:upper_w+1]
            img_out = Image.fromarray(img_crop, 'RGB')
            img_out.save(os.path.join(args.root_dir,"processed/train","glomerulus_{}_{}.png".format(image_id,j)))
            mask_crop=mask[lower_h:upper_h+1,lower_w:upper_w+1]
            mask_out = Image.fromarray(mask_crop, 'L').convert('1')
            mask_out.save(os.path.join(args.root_dir,"processed/mask",'glomerulus_{}_{}.png'.format(image_id,j)))
        cnt+=1
        if(cnt<5 or cnt==10 or cnt==20 or cnt%50==0):
            print("processed {} glomeruli".format(cnt))
            
    if(args.saveImage):
        print("total saved image for {}.tiff is {}".format(image_id,cnt))
    print()
    
print("Max width is {}".format(gmax_width))
print("Max height is {}".format(gmax_height))

