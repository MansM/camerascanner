import cv2
import os
import pathlib
import logging
import tensorflow as tf
import numpy as np
import multiprocessing
from multiprocessing import Queue, Pool

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# MODE can be LABELS OR BOX
MODE = "LABELS"
# VIDEO_SOURCE="rtsp://UNIFIIP:PORT/TOKEN"
VIDEO_SOURCE=0

def worker(input_q, output_q):
    utils_ops.tf = tf.compat.v1
    tf.gfile = tf.io.gfile

    PATH_TO_LABELS = os.getcwd() + '/models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
    detection_model = load_model(model_name)

    while (True):
        print("worker spam")
        frame = input_q.get()
        img = show_inference(category_index, detection_model, frame)
        output_q.put(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2RGBA))


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir) / "saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(category_index, model, image):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(image)
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


if __name__ == '__main__':

    # print(detection_model.inputs)
    if (MODE == "LABELS"):
        logger = multiprocessing.log_to_stderr()
        logger.setLevel(multiprocessing.SUBDEBUG)

        input_q = Queue(2)  # fps is better if queue is higher but then more lags
        output_q = Queue()
        pool = Pool(1, worker, (input_q, output_q))

    if (MODE == "BOX"):
        logger = logging.getLogger()
        tensorflowNet = cv2.dnn.readNetFromTensorflow('ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb',
                                                      'ssd_mobilenet_v1_coco_2017_11_17/graph.pbtxt')

    stream = cv2.VideoCapture(VIDEO_SOURCE)

    while (True):
        grabbed, frame = stream.read()
        logger.debug("got a frame")

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rows, cols, channels = frame.shape
        # Display the resulting frame

        if (MODE == "LABELS"):
            # img = show_inference(detection_model, frame)
            input_q.put(frame)
            # print(img)
            frame = cv2.cvtColor(np.asarray(output_q.get()), cv2.COLOR_RGB2RGBA)
            # cv2.imread('frame')

        if (MODE == "BOX"):
            # # Use the given image as input, which needs to be blob(s).
            tensorflowNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))

            # Runs a forward pass to compute the net output
            networkOutput = tensorflowNet.forward()

            # Loop on the outputs
            for detection in networkOutput[0, 0]:

                score = float(detection[2])
                if score > 0.2:
                    left = detection[3] * cols
                    top = detection[4] * rows
                    right = detection[5] * cols
                    bottom = detection[6] * rows

                    # draw a red rectangle around detected objects
                    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

        # Show the image with a rectagle surrounding the detected objects
        cv2.imshow('Image', frame)
        # cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    stream.release()
    cv2.destroyAllWindows()
    if MODE == "LABELS": pool.terminate()
