import numpy as np
import tensorflow as tf
import cv2
import os
from keras.utils.data_utils import get_file
import cv2
import os


def get_model(model_URL):
    model_name = os.path.basename(model_URL).split('.')[0]
    get_path = get_file(fname=model_name, untar=True, origin=model_URL)
    model = tf.saved_model.load(os.path.join(get_path, 'saved_model'))
    return model


url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz'


def object_detection(image_path, model, classes_name, max_output_size=50, iou_threshold=0.4, score_threshold=0.7,
                     soft_nms_sigma=0.4):
    image = image_path  # You may want to read the image with cv2: cv2.imread(image_path)
    image2tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    image2tensor = image2tensor[tf.newaxis, ...]

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

        class_label = classes_name[class_index]
        class_color = (255, 255, 255)  # classes_color[class_index]

        bbox = bboxes[i].tolist()
        ymin, xmin, ymax, xmax = bbox
        ymin, xmin, ymax, xmax = int(ymin * img_h), int(xmin * img_w), int(ymax * img_h), int(xmax * img_w)
        cv2.rectangle(img=image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=class_color, thickness=2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

path = './Dataset/train'
out_dir = os.listdir(path)
c = 0
Frames = []
for i in range(len(out_dir)):
    files = path + '/' + out_dir[i]
    in_dir = os.listdir(files)
    for j in range(2):
        print(i, j)
        FileName = files + '/' + in_dir[j]
        cap = cv2.VideoCapture(FileName)  # Or use 0 for webcam

        # Initialize the tracker
        tracker = cv2.TrackerCSRT_create()

        # Initialize variables
        initialized = False
        frame_height, frame_width = 0, 0

        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # If not initialized, use YOLO for detection
            if not initialized:
                # Create a blob from the frame and perform a forward pass
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                detections = net.forward(output_layers)

                # Loop through detections and find the most confident ones
                class_ids = []
                confidences = []
                boxes = []

                for detection in detections:
                    for obj in detection:
                        scores = obj[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > 0.5:  # Confidence threshold
                            center_x = int(obj[0] * frame_width)
                            center_y = int(obj[1] * frame_height)
                            w = int(obj[2] * frame_width)
                            h = int(obj[3] * frame_height)

                            # Bounding box coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                # Apply Non-Maximum Suppression to remove overlapping boxes
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                # Initialize tracker with the first detected object (if any)
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        tracker.init(frame, (x, y, w, h))
                        initialized = True
                        break  # Track the first detected object

            # If initialized, update the tracker and draw the bounding box
            if initialized:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

            Frames.append(frame)
np.save('Detected_Image.npy', Frames)
