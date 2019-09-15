# General System stuff
import sys
import gc
import glob
import os
from os.path import getsize
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
IOU_THRESHOLD = 0.6
OVERLAP_THRESHOLD = 0.8
MIN_DETECTIONS = 1

#--------------------------------------#

# Processing stuff
import multiprocessing
import numpy as np
import random
import itertools
import operator
from operator import itemgetter
from itertools import groupby
from tqdm import tqdm

#--------------------------------------#

# Deep learning and GPU stuff
from keras import backend as K
from numba import cuda
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import googleapiclient.discovery as discovery

sys.path.insert(0, '/home/ec2-user/SageMaker/Segment2P/mrcnn')
from mrcnn import model as modellib
from mrcnn import utils
from train import cellConfig
#--------------------------------------#

# Image Stuff
import cv2
from PIL import Image
from tifffile import imsave,imread
from skimage import exposure
from skimage.color import rgb2gray
from skimage import io as skio
from skimage import util
#--------------------------------------#

# File Handling
import datetime
import time
from glob import glob
import csv
import pandas as pd
import urllib.request
import base64
import json
import time
import re

#--------------------------------------#

# GUI
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

#--------------------------------------#

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import colorsys
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.offline as offline
plotly.offline.init_notebook_mode(connected=True)

import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to had
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name
imports = list(set(get_imports()))

# The only way I found to get the version of the root package
# from only the name of the package is to cross-check the names 
# of installed packages vs. imported packages
requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

print("Imported Packages:\n")
for r in requirements:
    print("{}=={}".format(*r))
    
def cleanliorimage(alignedpath):
        rescale = MinMaxScaler(feature_range=(0, 1))
        pt = QuantileTransformer(output_distribution='normal')
        img = skio.imread(alignedpath)
        img = pt.fit_transform(img)
        img = rescale.fit_transform(img)
        img = exposure.equalize_adapthist(img, clip_limit=0.05)
        imrescaled = cv2.resize(img, dsize=(1024, 1024),
                                interpolation=cv2.INTER_AREA)
        imrescaled = util.invert(imrescaled)
        imrescaled = imrescaled*(255**2)
        imrescaled = imrescaled.astype('u2')
#         skio.imsave(os.path.join('/'.join(alignedpath.split('/')[:-1]),"inference/")+alignedpath.split('/')[-1], imrescaled)
        return(imrescaled)

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
    
def merge_multiple_detections(masks):
        """

        :param masks:
        :return:
        """
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
    
def submitjob(inputpaths,outputpath):
        # Submit job to cloud
    # ------------------------------------------#
    # Define batch job submission vfunction
    def make_batch_job_body(project_name, input_paths, output_path,
                            model_name, region, data_format='JSON',
                            version_name=None, max_worker_count=None,
                            runtime_version=None):

        project_id = 'projects/{}'.format(project_name)
        model_id = '{}/models/{}'.format(project_id, model_name)
        if version_name:
            version_id = '{}/versions/{}'.format(model_id, version_name)

        # Make a jobName of the format "model_name_batch_predict_YYYYMMDD_HHMMSS"
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())

        # Make sure the project name is formatted correctly to work as the basis
        # of a valid job name.
        clean_project_name = re.sub(r'\W+', '_', project_name)

        job_id = '{}_{}_{}'.format(clean_project_name, model_name,
                               timestamp)

        # Start building the request dictionary with required information.
        body = {'jobId': job_id,
                'predictionInput': {
                    'dataFormat': data_format,
                    'inputPaths': input_paths,
                    'outputPath': output_path,
                    'region': region}}

        # Use the version if present, the model (its default version) if not.
        if version_name:
            body['predictionInput']['versionName'] = version_id
        else:
            body['predictionInput']['modelName'] = model_id

        # Only include a maximum number of workers or a runtime version if specified.
        # Otherwise let the service use its defaults.
        if max_worker_count:
            body['predictionInput']['maxWorkerCount'] = max_worker_count

        if runtime_version:
            body['predictionInput']['runtimeVersion'] = runtime_version

        return body

    # Project definitions
    project_name = "divine-builder-142611"
    credentials_path = "/home/mestalbet/bucket/PythonScripts/Segmentation Project-a5a157bd9401.json"
    project_id = 'projects/{}'.format(project_name)
#     input_paths = "gs://segproj/PythonScripts/submit_data/"
#     output_path = "gs://segproj/PythonScripts/cloud_output/"
    model_name = "segmentation"
    region = "us-central1"

    # Submit job
    version_names = ['Model0100_Ver1_0','Model0198_Ver1_0','Model0199_Ver1_0','Model0200_Ver1_0']
    for vn in version_names:
        op = os.path.join(output_path,vn)
        if not os.path.exists(op):
            os.makedirs(op)

        dirs = os.listdir(op) 
        vernum = [int(d[-1]) for d in dirs]
        if not vernum:
            vernum = 0
        else:
            vernum = np.max(vernum)
        op = os.path.join(op, "run_%d" % (vernum+1))
        if not os.path.exists(op):
            os.makedirs(op)

        batch_predict_body = make_batch_job_body(project_name, input_paths, op,
                                                 model_name, region, data_format='JSON',
                                                 version_name=vn, max_worker_count=None,
                                                 runtime_version=None)
        ml = discovery.build('ml', 'v1')
        request = ml.projects().jobs().create(parent=project_id,body=batch_predict_body)
        response = request.execute()

chunksize = 1
data_dir = "/home/mestalbet/bucket/PythonScripts/Results_LiorImages/inference/pngs/"
class CellInferenceConfig(cellConfig):
        # Set batch size to 1 to run one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = chunksize
        # Don't resize imager for inferencing
        IMAGE_RESIZE_MODE = "pad64"
        # Non-max suppression threshold to filter RPN proposals.
        # You can increase this during training to generate more propsals.
        RPN_NMS_THRESHOLD = 0.7
        # define the folder path to data for prediction
        global data_dir
        all_files = []
        sub_directory = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                relativePath = os.path.relpath(root, data_dir)
                if relativePath == ".":
                    relativePath = ""
                all_files.append(
                    (relativePath.count(os.path.sep), relativePath, file))
        all_files.sort(reverse=True)
        for (count, folder), files in groupby(all_files, itemgetter(0, 1)):
            sub_directory.append(folder)
            
class ImageDataset(utils.Dataset):
    def load_images(self, dataset_dir):
        """
        Loads dataset images.
        :param dataset_dir: string, path to dataset directory.
        :return: None
        """
        self.add_class("cell", 1, "cell")

        image_ids = [fn for fn in os.listdir(dataset_dir)
                     if any(fn.endswith(ext) for ext in ['tif', "png"])]

        for image_id in image_ids:
            self.add_image(
                'cell',
                image_id=os.path.splitext(image_id)[0],
                path=os.path.join(dataset_dir, image_id)
            )
            
def createjson(imgpath = "/home/mestalbet/PythonScripts/Results_LiorImages/inference/", 
               json_output_path = "gs://segproj/PythonScripts/submit_data/inputs.json",
               model_path = "/home/mestalbet/Segment2P/TrainWeights/mask_rcnn_cell_0100.h5",
               data_dir = "/home/mestalbet/PythonScripts/Results_LiorImages/inference/"):
    predict_instance_json = "/home/mestalbet/temp/inputs.json"
    os.system("rm %s" % (predict_instance_json))
    config=CellInferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=data_dir)
    model.load_weights(model_path, by_name=True)

    dataset = ImageDataset()
    dataset.load_images(imgpath)
    dataset.prepare()
    with open(predict_instance_json, "w+") as fp:
        for image_id in dataset.image_ids:
            image = dataset.load_image(image_id)
            active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
            source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
            active_class_ids[source_class_ids] = 1
            img, window, scale, padding, crop = utils.resize_image(
                    image,
                    min_dim=config.IMAGE_MIN_DIM,
                    min_scale=config.IMAGE_MIN_SCALE,
                    max_dim=config.IMAGE_MAX_DIM,
                    mode=config.IMAGE_RESIZE_MODE)
            meta = np.array(
                [image_id] +                  # size=1
                list(image.shape) +           # size=3
                list(img.shape) +             # size=3
                list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
                [scale] +                     # size=1
                list(active_class_ids)        # size=num_classes
                )
            anchors = model.get_anchors(image.shape)
            json_data = {'input_image':image.tolist(),
                         'input_image_meta':meta.tolist(),
                         'input_anchors':anchors.tolist()} #,'key':int(image_id)
            jline = json.dumps(json_data) + "\n"
            fp.write(jline)
        os.system('gsutil -m cp "/home/mestalbet/temp/inputs.json" %s' % json_output_path)

def segmentimage(data_dir, model_list):
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
#                 self.log = open(data_dir+"log.log", "a")

        def write(self, message):
            self.log = open(data_dir+"log.log", "a")
            self.terminal.write(message)
            self.log.write(message)
            self.log.close()

        def flush(self):
            pass

    class ImageDataset(utils.Dataset):
        def load_images(self, dataset_dir):
            """
            Loads dataset images.
            :param dataset_dir: string, path to dataset directory.
            :return: None
            """
            self.add_class("cell", 1, "cell")

            image_ids = [fn for fn in os.listdir(dataset_dir)
                         if any(fn.endswith(ext) for ext in ['tif', "png"])]

            for image_id in image_ids:
                self.add_image(
                    'cell',
                    image_id=os.path.splitext(image_id)[0],
                    path=os.path.join(dataset_dir, image_id)
                )

    class CellInferenceConfig(cellConfig):
        # Set batch size to 1 to run one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # Don't resize imager for inferencing
        IMAGE_RESIZE_MODE = "pad64"
        # Non-max suppression threshold to filter RPN proposals.
        # You can increase this during training to generate more propsals.
        RPN_NMS_THRESHOLD = 0.7
        # define the folder path to data for prediction
        global data_dir
        all_files = []
        sub_directory = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                relativePath = os.path.relpath(root, data_dir)
                if relativePath == ".":
                    relativePath = ""
                all_files.append(
                    (relativePath.count(os.path.sep), relativePath, file))
        all_files.sort(reverse=True)
        for (count, folder), files in groupby(all_files, itemgetter(0, 1)):
            sub_directory.append(folder)

    def detect(model, data_dir, out_dir):
        '''
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        detection_dir = "detections_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
        detection_dir = os.path.join(out_dir, detection_dir)
        os.makedirs(detection_dir)
        '''
        # Read dataset
        dataset = ImageDataset()
        dataset.load_images(data_dir)
        dataset.prepare()
        # Load over images
        for image_id in tqdm(dataset.image_ids):
            # Load image and run detection
            image = dataset.load_image(image_id)
            # Detect objects
            r = model.detect([image], verbose=0)[0]
            # Encode image to RLE. Returns a string of multiple lines
            source_id = dataset.image_info[image_id]["id"]

            #out_path = os.path.join(detection_dir, '%s.png' % str(source_id))
            out_path = os.path.join(out_dir, '%s.png' % str(source_id))

            mask = np.argmax(r['masks'], 2)
            cv2.imwrite(os.path.normpath(out_path), mask)

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

    def merge_multiple_detections(masks):
        """

        :param masks:
        :return:
        """
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

        cv2.imwrite(os.path.join(data_dir, 'results/final_bin.png'), bin_mask)
        cv2.imwrite(os.path.join(data_dir, 'results/final.png'), final_mask)
        return(final_mask)

    def process_sequence(masks):
        """

        :param masks:
        :return:
        """

    def postprocess(data_dir, out_dir):
        """

        :param data_dir:
        :param out_dir:
        :return:
        """
        #os.makedirs(out_dir, exist_ok=True)
        models_dir = [os.path.join(data_dir, filename)
                      for filename in os.listdir(data_dir)]
        print('Merging multiple models predictions.')
        filenames = os.listdir(models_dir[0])

        for filename in tqdm(filenames):
            masks = [cv2.imread(os.path.join(model_dir, filename), 0)
                     for model_dir in models_dir]

            result = merge_multiple_detections(masks)
            bin_result = np.zeros(result.shape)
            bin_result[np.where(result > 0)] = 255

            cv2.imwrite(os.path.join(out_dir, filename), result)
    
    all_files = []
    sub_directory = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            relativePath = os.path.relpath(root, data_dir)
            if relativePath == ".":
                relativePath = ""
            all_files.append(
                (relativePath.count(os.path.sep), relativePath, file))
    all_files.sort(reverse=True)
    for (count, folder), files in groupby(all_files, itemgetter(0, 1)):
        sub_directory.append(folder)

    config = CellInferenceConfig()
    model = modellib.MaskRCNN(
        mode="inference", config=config, model_dir=data_dir)
    sys.stdout = Logger()

    total_start = time.time()
    run_time_log = []
    for i in sub_directory:
        counter = 1
        start = time.time()
        mask_duplicate_dir = os.path.join("/home/mestalbet/TempMasks/", i+'_mask')
        for m in model_list:
            # print(m)
            print('>Loading model from: ', m)
            model.load_weights(m, by_name=True)
            try:
                predict_location = os.path.join(data_dir, i)
                print('prediction for: ', predict_location)
                print('model run '+str(counter)+' of '+str(len(model_list)))
                try:
                    out_dir = os.path.join(
                        mask_duplicate_dir, "_"+str(counter))
                    os.makedirs(out_dir)
#                     out_dir = "/home/mestalbet/TempMasks/"
                except:
                    print('failed to create mask folder')
                try:
                    detect(model, predict_location, out_dir)
                except:
                    print('failed to deploy the inference, skipping...')
            except:
                print('error, skipping...')
            counter += 1
        try:
            avg_prediction_dir = os.path.join(out_dir, i+"_mask_avg")
            os.makedirs(avg_prediction_dir)
        except:
            print('failed to create avg mask folder')
        postprocess(mask_duplicate_dir, avg_prediction_dir)
        end = time.time()
        time_diff = end-start
        run_time_log.append(time_diff)
        hour = time_diff // 3600
        time_diff %= 3600
        minutes = time_diff // 60
        time_diff %= 60
        seconds = time_diff
        print('prediction run time = %d hr: %d min: %d s' %
              (hour, minutes, seconds))
    print(run_time_log)
    for item in run_time_log:
        runtimelogfile = open('exptime.txt', 'w')
        runtimelogfile.write("%s\n" % item)
        runtimelogfile.close()
    total_end = time.time()
    total_time = total_end - total_start
    total_day = total_time // (3600*24)
    total_time %= (3600*24)
    total_hour = total_time // 3600
    total_time %= 3600
    total_minutes = total_time // 60
    total_time %= 60
    total_seconds = total_time
    print('Total prediction run time = %d day: %d hr: %d min: %d s' %
          (total_day, total_hour, total_minutes, total_seconds))
    
def getcell(mask, cellnum):
    cell = mask == cellnum
    return(cell.astype("float32"))


def random_colors(N):
    """
    Generate brewer colors.
    To get visually distinct colors
    """
    colors = sns.color_palette("deep", N)
    return(np.asarray(colors))


def apply_mask(image, mask, color, alpha=0.3):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return(image)


def createoverlay(meanimg, mask):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.bitwise_not(meanimg)
    img = np.dstack([img/(255**2), img/(255**2), img/(255**2)])
    numcells = len(np.unique(mask))
    colors = random_colors(numcells)
    for i in range(1, numcells-1):
        cellmask = getcell(mask, i)
        x = int(np.round(np.mean(np.where(cellmask > 0)[1])))
        y = int(np.round(np.mean(np.where(cellmask > 0)[0])))
        img = apply_mask(img, cellmask, colors[i])
        img = cv2.putText(img, str(
            i-1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255**2, 0), 2, cv2.LINE_AA)
    return(img)


def gettraces(frames, mask):
    numcells = len(np.unique(mask))
    traces = []
    for cellid in range(1, numcells-1):
        cellmask = getcell(mask, cellid)
        cellmask[cellmask == 0 ] = np.nan
        traces.append([np.nanmean(np.multiply(frame.astype("float32"), cv2.resize(
            cellmask, (256, 256)))) for frame in frames])
    sig = pd.DataFrame({"cells": range(1, numcells-1), "traces": traces})
    return(sig) #has a bug - it's averaging with many zeros


def plotTrace(signals, cellnum):
    offline.iplot([go.Scatter(x=np.arange(
        len(signals['traces'].iloc[cellnum])), y=signals['traces'].iloc[cellnum])])