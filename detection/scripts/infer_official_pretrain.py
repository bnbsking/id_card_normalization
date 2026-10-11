import cv2
import supervision as sv
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES


model = RFDETRMedium()


detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)


labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]
annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)


cv2.imwrite("outputs/infer_example.jpg", annotated_image)  # save numpy
"""
detections
Detections(xyxy=array([[6.9542793e+01, 2.4810391e+02, 6.2046466e+02, 9.2655975e+02],
       [6.2692426e+02, 7.3331775e+02, 6.9698950e+02, 7.8809888e+02],
       [9.5561743e-01, 3.5660303e+02, 6.4902881e+02, 1.2664476e+03],
       [0.0000000e+00, 6.6193787e+02, 4.4367255e+02, 1.2709741e+03]],
      dtype=float32), mask=None, confidence=array([0.9329489 , 0.6150977 , 0.57471305, 0.5271065 ], dtype=float32), class_id=array([18,  3,  1, 27]), tracker_id=None, data={'source_shape': array([[1280,  720],
       [1280,  720],
       [1280,  720],
       [1280,  720]]), 'class_name': array(['dog', 'car', 'person', 'backpack'], dtype=object)}, metadata={'source_image': array([[[120, 157, 202],
        [120, 157, 202],
        [120, 157, 202],
        ...,
        [129, 161, 199],
        [128, 160, 198],
        [128, 160, 198]],

       [[119, 156, 201],
        [119, 156, 201],
        [119, 156, 201],
        ...,
        [129, 161, 199],
        [129, 161, 199],
        [128, 160, 198]],

       [[119, 156, 201],
        [119, 156, 201],
        [119, 156, 201],
        ...,
        [130, 162, 200],
        [129, 161, 199],
        [129, 161, 199]],

       ...,

       [[205, 204, 202],
        [223, 222, 220],
        [236, 235, 233],
        ...,
        [150, 124, 109],
        [154, 126, 112],
        [150, 123, 106]],

       [[200, 199, 197],
        [228, 227, 225],
        [223, 222, 220],
        ...,
        [147, 121, 106],
        [147, 119, 105],
        [150, 121, 105]],

       [[211, 210, 208],
        [197, 196, 194],
        [224, 223, 221],
        ...,
        [144, 118, 103],
        [148, 120, 106],
        [163, 134, 118]]], shape=(1280, 720, 3), dtype=uint8)})
"""