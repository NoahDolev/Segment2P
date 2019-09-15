import random
import string
import sagemaker
import boto3
import shutil
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders
from bs4 import BeautifulSoup
import dash_html_components as html
import base64
from joblib import Parallel, delayed
import multiprocessing
from sagemaker import get_execution_role
import os
import cv2
import math
import numpy as np
import pandas  as  pd
import traceback
from skimage import exposure,color, img_as_int, img_as_ubyte
from skimage.io import imread as pngread
from skimage.io import imsave as pngsave
from skimage.morphology import disk
from skimage.filters.rank import autolevel,equalize
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import pathlib
from PIL import Image
import io
from zipfile import ZipFile

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://mail.google.com/']

creds = None
# The file token.pickle stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
if os.path.exists('/home/ec2-user/token.pickle'):
    with open('/home/ec2-user/token.pickle', 'rb') as token:
        creds = pickle.load(token)
# If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            '/home/ec2-user/credentials.json', SCOPES)
        flow.run_console()
        creds = flow.credentials
    # Save the credentials for the next run
    with open('/home/ec2-user/token.pickle', 'wb') as token:
        pickle.dump(creds, token)

service = build('gmail', 'v1', credentials=creds)

# Call the Gmail API
results = service.users().labels().list(userId='me').execute()
labels = results.get('labels', [])

def create_message(emailSubject, emailTo, emailFrom, message_body, emailCc, html_content=None, link = None):
#     try:
    message = MIMEMultipart()
    message['to'] = emailTo
    message['from'] = emailFrom
    message['subject'] = emailSubject
    message['Cc'] = emailCc
    body_mime = MIMEText(message_body, 'plain')
    message.attach(body_mime)

    if html_content:
        with io.open(html_content, 'r') as fb:
            txt = str(fb.readlines()).replace("http://www.replacemewithlink.com", link)
        soup = BeautifulSoup(txt.replace('\n', '<br />'), "html5lib")
        html_mime = MIMEText(
            soup.prettify(formatter="html5").replace('\\n\\', '').replace('\\n', '').replace('\n','') \
            .replace(',','').replace("'","").replace(']','').replace('[','')
            , 'html')
        message.attach(html_mime)

    with open('/home/ec2-user/dash-image-processing/images/0.jpg', 'rb') as f:
        # set attachment mime and file name, the image type is png
        mime = MIMEBase('image', 'jpg', filename='0.jpg')
        # add required header data:
        mime.add_header('Content-Disposition', 'attachment', filename='0.jpg')
        mime.add_header('X-Attachment-Id', '0')
        mime.add_header('Content-ID', '<0>')
        # read attachment file content into the MIMEBase object
        mime.set_payload(f.read())
        # encode with base64
        encoders.encode_base64(mime)
        # add MIMEBase object to MIMEMultipart object
        message.attach(mime)

    with open('/home/ec2-user/dash-image-processing/images/1.jpg', 'rb') as f:
        # set attachment mime and file name, the image type is png
        mime = MIMEBase('image', 'jpg', filename='1.jpg')
        # add required header data:
        mime.add_header('Content-Disposition', 'attachment', filename='1.jpg')
        mime.add_header('X-Attachment-Id', '1')
        mime.add_header('Content-ID', '<1>')
        # read attachment file content into the MIMEBase object
        mime.set_payload(f.read())
        # encode with base64
        encoders.encode_base64(mime)
        # add MIMEBase object to MIMEMultipart object
        message.attach(mime)

    with open('/home/ec2-user/dash-image-processing/images/2.jpg', 'rb') as f:
        # set attachment mime and file name, the image type is png
        mime = MIMEBase('image', 'jpg', filename='2.jpg')
        # add required header data:
        mime.add_header('Content-Disposition', 'attachment', filename='2.jpg')
        mime.add_header('X-Attachment-Id', '2')
        mime.add_header('Content-ID', '<2>')
        # read attachment file content into the MIMEBase object
        mime.set_payload(f.read())
        # encode with base64
        encoders.encode_base64(mime)
        # add MIMEBase object to MIMEMultipart object
        message.attach(mime)
    print('HTML Message Created')
    return({'raw': base64.urlsafe_b64encode(message.as_string().encode()).decode()})

def send_message(service, user_id, message):
  """Send an email message.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    message: Message to be sent.

  Returns:
    Sent Message.
  """
  try:
    print('About to send')
    message = (service.users().messages().send(userId=user_id, body=message).execute())
    return('Success')
  except Exception as error:
    print(error)
    return('An error occurred when sending you the results. \nPlease email mrivlinlab@gmail.com with error message \
            and a sample of the image you are trying to segment :\n%s' % error)
    
def sendemail(to,link):
    
    msg_body = """\
    Hi,
    This email is best viewed with a client which renders HTML. 
    Your results can be found here:
    {}
    """.format(link)
    msg = create_message(emailSubject = "Your Segmentation Results", emailTo = to, 
                         emailFrom = "mrivlinlab@gmail.com", message_body =  msg_body, emailCc = "",
                         html_content = "/home/ec2-user/dash-image-processing/email.html",link = link)
    with open('/home/ec2-user/token.pickle', 'rb') as token:
        creds = pickle.load(token)
    service = build('gmail', 'v1', credentials=creds)
    res = send_message(service = service, user_id = "me", message = msg)
    return(res)
    
def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def inputpipeline(files, batchid=randomString(10)):
    def preproc(img):
        selem = disk(60)
        try:
            img = autolevel(img, selem)
            img = exposure.adjust_gamma(img, 2)
            img = cv2.bilateralFilter(img, 9, 75, 75)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            pass
        return (img)

    def createmultipleinputs(inputpath):
        # pad to square
        im = cv2.imread(inputpath,0)
        if len(im.shape) == 3:
            print('Images should be grayscale but had dimensions {} - automatically converted'.format(im.shape))
            im = np.sum(im, 2)
        num = int(''.join(filter(str.isdigit, str(im.dtype)))) - 1
        im = exposure.rescale_intensity(im, out_range=(0, 2**num - 1))
        imshape = im.shape
        edgediff = np.max(imshape) - np.min(imshape)
        orig = im
        
        orig_distorted = cv2.resize(orig, (math.ceil(0.9*np.max(imshape)), math.ceil(0.9*np.max(imshape))), interpolation=cv2.INTER_AREA)
            
        if imshape[1] > imshape[0]:
            orig = cv2.copyMakeBorder(im, math.ceil(edgediff / 2), math.ceil(edgediff / 2), 0, 0, cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])
        if imshape[0] > imshape[1]:
            orig = cv2.copyMakeBorder(im, 0, 0, math.ceil(edgediff / 2), math.ceil(edgediff / 2), cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])
        
        #original size but square
        if any(orig.shape)<1024:
            orig = cv2.copyMakeBorder(orig,  1024-orig.shape[0], 0,1024-orig.shape[1], 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            orig = orig[0:1024, 0:1024]
        
        # distorted size    
        if any(orig_distorted.shape)<1024:
            orig_distorted = cv2.copyMakeBorder(orig_distorted, 1024-orig_distorted.shape[0], 0,1024-orig_distorted.shape[1], 0,cv2.BORDER_CONSTANT,value=[0, 0, 0])
        else:
            orig_distorted=orig_distorted[0:1024, 0:1024]
    
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(127,127))
        orig_clahe = clahe.apply(orig)
        orig_pp = preproc(orig)
        orig_distorted_pp = preproc(orig_distorted)
        imlabels = ['raw','orig', 'orig_pp', 'orig_clahe','orig_distorted','orig_distorted_pp']
        return ([im, orig, orig_pp, orig_clahe, orig_distorted, orig_distorted_pp],imlabels)

    def populate_inputs(localpaths, batchid=''):
        os.makedirs('/tmp/{}/'.format(batchid), exist_ok=True)

        def innerloop(filepath, batchid=batchid):
            resimages, imlabels = createmultipleinputs(filepath)
            for idx in range(0, len(resimages)):
                savepath = '/tmp/' + batchid + '/' + batchid + '_' + \
                filepath.split(pathlib.PurePosixPath(filepath).suffix)[0].split('/')[-1] + '__' + imlabels[idx] + '.jpg'
                cv2.imwrite(savepath, img_as_ubyte(resimages[idx]))

        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(innerloop)(filepath) for filepath in localpaths)
        os.system(
            "aws s3 sync '/tmp/{}/' 's3://sagemaker-eu-west-1-102554356212/submissions/{}/' ".format(batchid, batchid))
        shutil.rmtree('/tmp/{}/'.format(batchid))

    def runbatch(model_id, batchid='', waitflag = False):

        env = {'SAGEMAKER_MODEL_SERVER_TIMEOUT': '3600'}
        s3 = boto3.resource('s3')
        s3_resource = boto3.resource('s3')
        sess = sagemaker.Session()
        bucket = sess.default_bucket()
        s3results = s3_resource.Bucket(name=bucket)
        removesamples = [obj.key for obj in s3results.objects.all() if (
                "results_" + model_id in obj.key and batchid in obj.key and (
                "out" in obj.key or "masks" in obj.key))]
        for removeme in removesamples:
            boto3.client('s3').delete_object(Bucket=bucket, Key=removeme)

        transform_job = sagemaker.transformer.Transformer(
            model_name=model_id,
            instance_count=1,
            instance_type='ml.p3.2xlarge',
            strategy='SingleRecord',
            assemble_with='None',
            output_path="s3://{}/results_{}/".format(bucket,model_id),
            base_transform_job_name='inference-pipelines-batch',
            sagemaker_session=sess,
            accept='image/png',
            env=env)
        transform_job.transform(data='s3://{}/submissions/'.format(bucket),
                                content_type='image/jpeg',
                                split_type=None)
        if waitflag:
            transform_job.wait()

    def merge_multiple_detections(masks):
        """

        :param masks:
        :return:
        """
        IOU_THRESHOLD = 0.6
        OVERLAP_THRESHOLD = 0.7
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
        return (final_mask)

    def merge_two_masks(maskpaths):
        masks = []
        for mpath in maskpaths:
            binarymask = cv2.imread(mpath,0)
            distance = ndi.distance_transform_edt(binarymask)
            local_maxi = peak_local_max(distance, labels=binarymask, footprint=np.ones((3, 3)), indices=False)
            markers = ndi.label(local_maxi)[0]
            masks.append(watershed(-distance, markers, mask=binarymask))
        mask = merge_multiple_detections(masks)
        distance = ndi.distance_transform_edt(mask)
        local_maxi = peak_local_max(distance, labels=mask, footprint=np.ones((3, 3)), indices=False)
        markers = ndi.label(local_maxi)[0]
        mask = watershed(-distance, markers, mask=mask)
        return (mask)

    def merge_masks(modelres, modelids, batchid=''):
        outpath = '/home/ec2-user/tmp/results/{}/merged/'.format(batchid)
        os.makedirs(outpath, exist_ok=True)

        def savemerge(masklist, outpath=outpath, modelids=modelids):
#             mask = np.uint8(merge_two_masks(masklist)) > 0 #returns binary mask
            mask = merge_two_masks(masklist) # returns mask where pixel value = cell number
            savepath = os.path.join(outpath, 'merged_' + masklist[0].split('/')[-1].split(modelids[0])[-1])
            savepath = savepath.replace(pathlib.PurePosixPath(savepath).suffix,'.png')
#             pngsave(savepath, np.uint8(mask>0))
            cv2.imwrite(savepath,mask)

        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(
            delayed(savemerge)([modelres[0][idx], modelres[1][idx]]) for idx in range(0, len(modelres[0])))

    def batch2masks(model_id, batchid=''):
        s3 = boto3.resource('s3')
        s3_resource = boto3.resource('s3')
        sess = sagemaker.Session()
        bucket = sess.default_bucket()
        os.makedirs('/home/ec2-user/tmp/results/{}'.format(batchid), exist_ok=True)
        os.system('aws s3 sync s3://{}/results_{}/{}/ /home/ec2-user/tmp/results/{}//'.format(bucket, model_id, batchid,batchid))
        files = os.listdir('/home/ec2-user/tmp/results/{}/'.format(batchid))
        files = [f for f in files if '.out' in f and batchid in f]
        savepaths=[]
        for f in files:
            file = os.path.join('/home/ec2-user/tmp/results/{}/'.format(batchid), f)
            with open(file, 'rb') as image:
                img = image.read()
            img = bytearray(img)
            mask = np.array(Image.open(io.BytesIO(img)))
            savepath = '/home/ec2-user/tmp/results/' + batchid + '/' + model_id + '-'+ f.replace(pathlib.PurePosixPath(f).suffix,'')
            savepath = savepath.replace(pathlib.PurePosixPath(savepath).suffix,'.png')
            cv2.imwrite(savepath, img_as_ubyte(mask))
            savepaths.append(savepath)  
        return(savepaths)

    def merge_masks_diff_inputs(groupkeys, batchid=''):
        outpath = '/home/ec2-user/results/{}/inputmerged/'.format(batchid)
        os.makedirs(outpath, exist_ok=True)
        masks = []
        
        orig_shape = None
        orig_shape = [file for file in groupkeys if '__raw' in file]
        if orig_shape is not None:
            orig_shape = cv2.imread(orig_shape[0],0).shape
        for file in groupkeys:
            if '__raw' not in file:
                if ('_distorted' in file and orig_shape is not None): #warps mask back into original shape
                    distortby = math.ceil(0.9*np.max(orig_shape))
                    tempmask = cv2.imread(file,0)
                    tempmask = tempmask[1024-distortby:-1,1024-distortby:-1]
                    edgediff = np.max(orig_shape) - np.min(orig_shape)
                    tempmask = cv2.resize(tempmask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_AREA)
                    if orig_shape[1] > orig_shape[0]:
                        tempmask = cv2.copyMakeBorder(tempmask, math.ceil(edgediff / 2), math.ceil(edgediff / 2), 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    if orig_shape[0] > orig_shape[1]:
                        tempmask = cv2.copyMakeBorder(tempmask, 0, 0, math.ceil(edgediff / 2), math.ceil(edgediff / 2), cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    if any(tempmask.shape)<1024:
                        binarymask = cv2.copyMakeBorder(tempmask,  1024-tempmask.shape[0], 0,1024-tempmask.shape[1], 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    else:
                        binarymask = tempmask[0:1024, 0:1024]
                else:
                    binarymask = cv2.resize(cv2.imread(file,0), (1024, 1024), interpolation=cv2.INTER_AREA)
                    
                distance = ndi.distance_transform_edt(binarymask)
                local_maxi = peak_local_max(distance, labels=binarymask, footprint=np.ones((3, 3)), indices=False)
                markers = ndi.label(local_maxi)[0]
                masks.append(watershed(-distance, markers, mask=binarymask))
        try:
            binarymask = merge_multiple_detections(masks)
            distance = ndi.distance_transform_edt(binarymask)
            local_maxi = peak_local_max(distance, labels=binarymask, footprint=np.ones((3, 3)), indices=False)
            markers = ndi.label(local_maxi)[0]
            mask = watershed(-distance, markers, mask=binarymask)
        except:
            distance = ndi.distance_transform_edt(binarymask)
            local_maxi = peak_local_max(distance, labels=binarymask, footprint=np.ones((3, 3)), indices=False)
            markers = ndi.label(local_maxi)[0]
            mask = watershed(-distance, markers, mask=binarymask)
            pass
        savepath = os.path.join(outpath,
                                file.split('/')[-1].split('__')[0].replace('merged_', 'inputmerged_') + '.png')
        cv2.imwrite(savepath, img_as_ubyte(mask))
            
    def retrieve_file_paths(dirName):

        # setup file paths variable
        filePaths = []

        # Read all directory, subdirectories and file lists
        for root, directories, files in os.walk(dirName):
            for filename in files:
                if '.jpg' in filename:
                # Create the full filepath by using os module.
                    filePath = os.path.join(root, filename)
                    filePaths.append(filePath)
            # return all paths
        return(filePaths)
    
    try:
        modelids = ["fresh-train-trial-2019-07-28-08-49-49-994", "semantic-segmentatio-190726-1931-032-e7d26e04"]
        [os.rename(f,f.replace('_', '-')) for f in files]
        files = [f.replace('_','-') for f in files] #we use __ to split multiple preprocessings so must remove them from file name
        files = [f for f in files if '.jpg' in f or '.png' in f or '.tif' in f]
        populate_inputs(files, batchid=batchid)
        for idx in range(0,len(modelids)):
            model = modelids[idx]
            waitflag = (idx == len(modelids)-1)
            runbatch(model,batchid,waitflag)
        results = [batch2masks(mid, batchid=batchid) for mid in modelids]
        twomods = list(
            set([r.split(modelids[0])[-1] for r in results[0]]) & set([r.split(modelids[1])[-1] for r in results[1]]))
        firstfiles = [r for r in results[0] if r.split(modelids[0])[-1] in twomods]
        secondfiles = [r for r in results[1] if r.split(modelids[1])[-1] in twomods]
        merge_masks((firstfiles, secondfiles), modelids = modelids, batchid=batchid)
        keys = []
        for root, dirnames, filenames in os.walk('/home/ec2-user/tmp/results/{}/merged/'.format(batchid)):
            for files in filenames:
                if ('.jpg' in files or '.png' in files or '.tif' in files):
                    keys.append(os.path.join('/home/ec2-user/tmp/results/{}/merged/'.format(batchid), files))
        df = pd.DataFrame({'keys': keys, 'orig_name': [k.split('/')[-1].split('__')[0].split('.jpg')[0] for k in
                                                       keys]})  
        originals = np.unique(df['orig_name'].values)
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(
            delayed(merge_masks_diff_inputs)(df['keys'].loc[df['orig_name'] == org].values, batchid) for org in originals)
        
        with ZipFile('/home/ec2-user/results/{}/{}.zip'.format(batchid,batchid), 'a') as myzip:
            dirname = '/home/ec2-user/results/{}/inputmerged/'.format(batchid)
            files2zip = retrieve_file_paths(dirname)
            for file in files2zip:
                myzip.write(file, arcname = os.path.basename(file))      
        os.system(
            "aws s3 cp '/home/ec2-user/results/{}/{}.zip' 's3://sagemaker-eu-west-1-102554356212/results_merged/' --acl public-read".format(batchid, batchid,batchid))
        return(batchid)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return(batchid)
