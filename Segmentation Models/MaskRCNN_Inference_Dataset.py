import torch
# Some basic setup:
# Setup detectron2 logger
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = [30, 15]

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import fiftyone as fo
from PIL import Image

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("CottonImaging_train",)
cfg.DATASETS.TEST = ()
#cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 12
#cfg.SOLVER.BASE_LR = 0.00015  # Learning Rate
cfg.SOLVER.MAX_ITER = 10000    
cfg.SOLVER.STEPS = []        # do not decay learning rate
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   # Default = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 
cfg.OUTPUT_DIR = "/home/avl/Projects/Cotton Imaging Project/Data/Splitted Dataset 3/Ethan/Prediction Masks/"

cfg.MODEL.WEIGHTS = "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/Training/Base Model/model_final.pth"  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# Create a dataset from a directory of images
dataset = fo.Dataset.from_images_dir("/home/avl/Projects/Cotton Imaging Project/Data/Splitted Dataset 3/Ethan/Ethan")

#predictions_view = dataset.take(255, seed=51)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get class list
#classes = dataset.default_classes
classes = ["OpenBoll", "ClosedBoll", "Flower", "Square"]
PATH = '/home/avl/Projects/Cotton Imaging Project/Data/Splitted Dataset 3/Ethan/Prediction Masks/'

# Add predictions to samples
with fo.ProgressBar() as pb:
    for sample in pb(dataset):        
        i = 1      
        # Load image
        image = cv2.imread(sample.filepath)
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w , c = image.shape

        # Perform inference
        preds = predictor(image)
        labels = preds["instances"].pred_classes.cpu().detach().numpy()
        scores = preds["instances"].scores.cpu().detach().numpy()
        #boxes = preds["instances"].pred_boxes.tensor.cpu().detach().numpy()
        masks = preds["instances"].pred_masks.cpu().detach().numpy()
        
        # Convert detections to FiftyOne format
        detections = []
        segmentations = []
#        for label, score, box, seg in zip(labels, scores, boxes, masks):
        for label, score, seg in zip(labels, scores, masks):
        
            # Generate images for each mask
            cv2.imwrite(PATH+sample.filepath[sample.filepath.rfind('/')+1:-4]
                        + f"_{i}_" + classes[label] + ".png", seg * 255)
            i += 1
            
            segmentations.append(
                fo.Detection.from_mask(
                    mask=seg,
                    label=classes[label],
                    confidence=score
                )
            )

        # Save predictions to dataset
#         sample["predictions"] = fo.Detections(detections=detections)
        sample["predictions"] = fo.Detections(detections=segmentations)
        sample.save()

print("Finished adding predictions")