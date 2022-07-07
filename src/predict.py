import io

import cv2
import numpy as np
from detectron2 import structures
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from src.static_text import NON_LABELS_TEXT

class FasterRCNN:

    def __init__(self, config_path: str):

        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.predictor = DefaultPredictor(self.cfg)

        register_coco_instances("clothes", {},
                                "src/dataset/_annotations.coco.json",
                                "./dataset")
        self.clothesnet = MetadataCatalog.get("clothes")
        self.dataset_dicts = DatasetCatalog.get("clothes")

    def _drop_duplicates(self, outputs):

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

    def _predict(self, image):

        im = cv2.imread(image)
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=self.clothesnet,
                       scale=0.8,
                       )

        instances = self._drop_duplicates(outputs)
        v = v.draw_instance_predictions(
            outputs["instances"][instances].to("cpu"))
        is_success, buffer = cv2.imencode(".jpg", v.get_image()[:, :, ::-1])
        bio = io.BytesIO(buffer)
        bio.name = "image.jpeg"
        bio.seek(0)
        text = sorted(
            set(np.array(outputs["instances"][instances].pred_classes)),
            reverse=True)
        return bio, text

    def __call__(self, image: str):
        answer = self._predict(image)
        return answer
