import io

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2 import structures
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# Register custom COCO dataset
register_coco_instances("clothes", {}, "./dataset/_annotations.coco.json",
                        "./dataset")
clothesnet = MetadataCatalog.get("clothes")
dataset_dicts = DatasetCatalog.get("clothes")

# Config for model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = ("./model/model.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 32
cfg.MODEL.DEVICE = "gpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set threshold for this model
predictor = DefaultPredictor(cfg)


def drop_duplicates(outputs):
    instances = [i for i in range(len(outputs["instances"]))]
    intersect_box = []

    for i in range(len(outputs["instances"].pred_boxes)):
        bboxes_1 = outputs["instances"].pred_boxes[i]
        for j in range(len(outputs["instances"].pred_boxes)):
            bboxes_2 = outputs["instances"].pred_boxes[j]
            if i != j:
                iou = structures.pairwise_iou(bboxes_1, bboxes_2)
                if iou > 0.3:
                    if (outputs["instances"].scores[i] <
                            outputs["instances"].scores[j]):
                        if i not in intersect_box:
                            intersect_box.append(i)

    for intersect in intersect_box:
        instances.remove(intersect)

    return instances


def predict(photo_name):
    im = cv2.imread(photo_name)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=clothesnet,
                   scale=0.8,
                   )

    instances = drop_duplicates(outputs)
    v = v.draw_instance_predictions(outputs["instances"][instances].to("cpu"))
    is_success, buffer = cv2.imencode(".jpg", v.get_image()[:, :, ::-1])
    bio = io.BytesIO(buffer)
    bio.name = 'image.jpeg'
    bio.seek(0)
    text = sorted(set(np.array(outputs["instances"][instances].pred_classes)),
                  reverse=True)
    return bio, text
