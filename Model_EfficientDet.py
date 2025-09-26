import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.python.keras.utils.data_utils import get_file


def get_model(model_URL):
    model_name = os.path.basename(model_URL).split('.')[0]
    get_path = get_file(fname=model_name, untar=True, origin=model_URL)
    model = tf.saved_model.load(os.path.join(get_path, 'Model'))
    return model


url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz'


def Model_EfficientDet(image_path, max_output_size=50, iou_threshold=0.4, score_threshold=0.7,
                     soft_nms_sigma=0.4):
    image = image_path
    image2tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    image2tensor = image2tensor[tf.newaxis, ...]
    classes_name = (image.shape[0], 101)
    model = get_model(url)
    detection = model(image2tensor)
    bboxes = detection['detection_boxes'].numpy()[0]
    class_indexes = detection['detection_classes'].numpy().astype(np.int32)[0]
    class_scores = detection['detection_scores'].numpy()[0]

    selected_indices, _ = tf.image.non_max_suppression_with_scores(bboxes, scores=class_scores,
                                                                   max_output_size=max_output_size,
                                                                   iou_threshold=iou_threshold,
                                                                   score_threshold=score_threshold,
                                                                   soft_nms_sigma=soft_nms_sigma)

    img_h, img_w, img_c = image.shape

    # Ensure classes_name is large enough to avoid index out of range
    classes_color = np.random.uniform(low=0, high=255, size=(len(classes_name), 3))

    for i in selected_indices:
        class_score = class_scores[i]
        class_index = class_indexes[i] - 1  # Adjust index to be zero-based

        # Check if the class index is within range
        if class_index >= len(classes_name) or class_index < 0:
            continue
        class_color = (255, 255, 255)

        bbox = bboxes[i].tolist()
        ymin, xmin, ymax, xmax = bbox
        ymin, xmin, ymax, xmax = int(ymin * img_h), int(xmin * img_w), int(ymax * img_h), int(xmax * img_w)
        cv2.rectangle(img=image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=class_color, thickness=2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
