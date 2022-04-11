import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

# Some basic setup:
# Setup detectron2 logger
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [30, 15]

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import fiftyone as fo
from PIL import Image


def add_predictions(dataset, predictor, actualLabel):
    predictions_view = dataset.take(len(dataset), seed=51)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Add predictions to samples
    with fo.ProgressBar() as pb:
        for sample in pb(predictions_view):        
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
            for label, score, seg in zip(labels, scores, masks):
                
                segmentations.append(
                    fo.Detection.from_mask(
                        mask=seg,
                        label=actualLabel,
                        confidence=score
                    )
                )

            sample["predictions"] = fo.Detections(detections=segmentations)
            sample.save()

    print("Finished adding predictions")
    return dataset


# -------------------------------------------------------------------OpenBoll---------------------------------------------------------------------
register_coco_instances("CottonImaging_train_OpenBoll", {}, '/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Training7030/OpenBoll/OpenBollTrainingSet_70.json', "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/training_images")
metadata_train_OpenBoll = MetadataCatalog.get("CottonImaging_train_OpenBoll")

register_coco_instances("CottonImaging_test_OpenBoll", {}, "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Testing7030/OpenBoll/OpenBollTestingSet_30.json", "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/test_images")
metadata_test_OpenBoll = MetadataCatalog.get("CottonImaging_test_OpenBoll")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("CottonImaging_train_OpenBoll",)
cfg.DATASETS.TEST = ()
#cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 12
#cfg.SOLVER.BASE_LR = 0.00015  # Learning Rate
cfg.SOLVER.MAX_ITER = 10000    
cfg.SOLVER.STEPS = []        # do not decay learning rate
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   # Default = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
cfg.OUTPUT_DIR = "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Models/OpenBoll"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

dataset = fo.Dataset.from_dir(
    data_path="/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/test_images",
    labels_path='/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Testing7030/OpenBoll/OpenBollTestingSet_30.json',
    dataset_type=fo.types.COCODetectionDataset,
    label_types=["detections", "segmentations"],
    label_field = "ground_truth",
    #name="Model_2500_1024BatchSize_15LR"
)

dataset = add_predictions(dataset, predictor, "OpenBoll")

dataset.export(
    labels_path=os.path.join(cfg.OUTPUT_DIR, "OpenBoll_testing_groundTruth_fiftyone.json") ,
    dataset_type=fo.types.COCODetectionDataset,
    label_field = "ground_truth_segmentations",
)

dataset.export(
    labels_path=os.path.join(cfg.OUTPUT_DIR, "OpenBoll_testing_predictions_fiftyone.json"),
    dataset_type=fo.types.COCODetectionDataset,
    label_field = "predictions",
)

dataset.delete()
# ------------------------------------------------------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------ClosedBoll---------------------------------------------------------------------
register_coco_instances("CottonImaging_train_ClosedBoll", {}, '/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Training7030/ClosedBoll/ClosedBollTrainingSet_70.json', "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/training_images")
metadata_train_ClosedBoll = MetadataCatalog.get("CottonImaging_train_ClosedBoll")

register_coco_instances("CottonImaging_test_ClosedBoll", {}, "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Testing7030/ClosedBoll/ClosedBollTestingSet_30.json", "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/test_images")
metadata_test_ClosedBoll = MetadataCatalog.get("CottonImaging_test_ClosedBoll")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("CottonImaging_train_ClosedBoll",)
cfg.DATASETS.TEST = ()
#cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 12
#cfg.SOLVER.BASE_LR = 0.00015  # Learning Rate
cfg.SOLVER.MAX_ITER = 10000    
cfg.SOLVER.STEPS = []        # do not decay learning rate
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   # Default = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
cfg.OUTPUT_DIR = "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Models/ClosedBoll"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

dataset = fo.Dataset.from_dir(
    data_path="/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/test_images",
    labels_path='/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Testing7030/ClosedBoll/ClosedBollTestingSet_30.json',
    dataset_type=fo.types.COCODetectionDataset,
    label_types=["detections", "segmentations"],
    label_field = "ground_truth",
    #name="Model_2500_1024BatchSize_15LR"
)

dataset = add_predictions(dataset, predictor, "ClosedBoll")

dataset.export(
    labels_path=os.path.join(cfg.OUTPUT_DIR, "ClosedBoll_testing_groundTruth_fiftyone.json") ,
    dataset_type=fo.types.COCODetectionDataset,
    label_field = "ground_truth_segmentations",
)

dataset.export(
    labels_path=os.path.join(cfg.OUTPUT_DIR, "ClosedBoll_testing_predictions_fiftyone.json"),
    dataset_type=fo.types.COCODetectionDataset,
    label_field = "predictions",
)

dataset.delete()
# ------------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------Flower----------------------------------------------------------------------
register_coco_instances("CottonImaging_train_Flower", {}, '/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Training7030/Flower/FlowerTrainingSet_70.json', "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/training_images")
metadata_train_Flower = MetadataCatalog.get("CottonImaging_train_Flower")

register_coco_instances("CottonImaging_test_Flower", {}, "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Testing7030/Flower/FlowerTestingSet_30.json", "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/test_images")
metadata_test_Flower = MetadataCatalog.get("CottonImaging_test_Flower")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("CottonImaging_train_Flower",)
cfg.DATASETS.TEST = ()
#cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 12
#cfg.SOLVER.BASE_LR = 0.00015  # Learning Rate
cfg.SOLVER.MAX_ITER = 10000    
cfg.SOLVER.STEPS = []        # do not decay learning rate
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   # Default = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
cfg.OUTPUT_DIR = "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Models/Flower"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

dataset = fo.Dataset.from_dir(
    data_path="/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/test_images",
    labels_path='/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Testing7030/Flower/FlowerTestingSet_30.json',
    dataset_type=fo.types.COCODetectionDataset,
    label_types=["detections", "segmentations"],
    label_field = "ground_truth",
    #name="Model_2500_1024BatchSize_15LR"
)

dataset = add_predictions(dataset, predictor, "Flower")

dataset.export(
    labels_path=os.path.join(cfg.OUTPUT_DIR, "Flower_testing_groundTruth_fiftyone.json") ,
    dataset_type=fo.types.COCODetectionDataset,
    label_field = "ground_truth_segmentations",
)

dataset.export(
    labels_path=os.path.join(cfg.OUTPUT_DIR, "Flower_testing_predictions_fiftyone.json"),
    dataset_type=fo.types.COCODetectionDataset,
    label_field = "predictions",
)

dataset.delete()
# ------------------------------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------Square----------------------------------------------------------------------
register_coco_instances("CottonImaging_train_Square", {}, '/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Training7030/Square/SquareTrainingSet_70.json', "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/training_images")
metadata_train_Square = MetadataCatalog.get("CottonImaging_train_Square")

register_coco_instances("CottonImaging_test_Square", {}, "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Testing7030/Square/SquareTestingSet_30.json", "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/test_images")
metadata_test_Square = MetadataCatalog.get("CottonImaging_test_Square")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("CottonImaging_train_Square",)
cfg.DATASETS.TEST = ()
#cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 12
#cfg.SOLVER.BASE_LR = 0.00015  # Learning Rate
cfg.SOLVER.MAX_ITER = 10000    
cfg.SOLVER.STEPS = []        # do not decay learning rate
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   # Default = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
cfg.OUTPUT_DIR = "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Models/Square"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

dataset = fo.Dataset.from_dir(
    data_path="/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/test_images",
    labels_path='/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/perCategoryDataset/Testing7030/Square/SquareTestingSet_30.json',
    dataset_type=fo.types.COCODetectionDataset,
    label_types=["detections", "segmentations"],
    label_field = "ground_truth",
    #name="Model_2500_1024BatchSize_15LR"
)

dataset = add_predictions(dataset, predictor, "Square")

dataset.export(
    labels_path=os.path.join(cfg.OUTPUT_DIR, "Square_testing_groundTruth_fiftyone.json") ,
    dataset_type=fo.types.COCODetectionDataset,
    label_field = "ground_truth_segmentations",
)

dataset.export(
    labels_path=os.path.join(cfg.OUTPUT_DIR, "Square_testing_predictions_fiftyone.json"),
    dataset_type=fo.types.COCODetectionDataset,
    label_field = "predictions",
)
# ------------------------------------------------------------------------------------------------------------------------------------------------