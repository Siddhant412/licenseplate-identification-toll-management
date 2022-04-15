import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from datetime import date
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from dotenv import load_dotenv
from pathlib import Path


# Flags
fframework = 'tf'
fsize = 416
ftiny = True
fmodel = 'yolov4'
foutput_format = 'XVID'
fiou = 0.45
fscore = 0.50
fdont_show = False
finfo = True
fcount = False

# path of tf weights file
fweights = './checkpoints/yolov4-tiny-416'

# path of input video
fvideo = './data/video/demo1.mp4'

# path of output video location
foutput = './outputs/demotrack4.avi'

# path of cropped LP saving directory
# lp_saving_dir = './egLocationID/' 
load_dotenv()
lp_saving_dir = os.getenv('LOCATION_ID')
print("LOC ID: ", lp_saving_dir)


def save_LP_captures(cropped_lp, original_vehicle, current_tracking_id, lp_saving_dir='./egLocationID/', vehicle_image_count=1):

    context ={}

    today = date.today()
    d1 = today.strftime("%d%m%Y")

    if not os.path.isdir(lp_saving_dir):
        os.mkdir(lp_saving_dir)
    context['location_id'] = lp_saving_dir

    date_dir = os.path.join(lp_saving_dir, d1)
    if not os.path.isdir(date_dir):
        os.mkdir(date_dir)
    
    vehicle_id_dir = os.path.join(date_dir, str(current_tracking_id))
    if not os.path.isdir(vehicle_id_dir):
        os.mkdir(vehicle_id_dir)

    context['vehicle_id'] = current_tracking_id

    # lp_path = os.path.join(vehicle_id_dir, str(current_tracking_id)+"_crop.jpg")
    # vehicle_path = os.path.join(vehicle_id_dir, str(current_tracking_id)+"_original.jpg")

    lp_dir_path = os.path.join(vehicle_id_dir, 'cropped_image')
    if not os.path.isdir(lp_dir_path):
        os.mkdir(lp_dir_path)

    vehicle_dir_path = os.path.join(vehicle_id_dir, 'manual_images')
    if not os.path.isdir(vehicle_dir_path):
        os.mkdir(vehicle_dir_path)

    lp_path = os.path.join(lp_dir_path, str(current_tracking_id)+"_crop.jpg")
    vehicle_path = os.path.join(vehicle_dir_path, str(current_tracking_id) + str(vehicle_image_count) + "_original.jpg")
    
    context['status'] = 'detected'

    try:
        if cropped_lp is not None:
            cv2.imwrite(lp_path, cropped_lp)
        cv2.imwrite(vehicle_path, original_vehicle)

        context['crop_location'] = os.path.join(vehicle_id_dir,"crop")
        context['manual_image_location'] = os.path.join(vehicle_id_dir,"manual_images")
    finally:
        return context


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(ftiny, fmodel)
    input_size = fsize
    video_path = fvideo

    # load tflite model if flag is set
    if fframework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=fweights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(fweights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if foutput:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*foutput_format)
        out = cv2.VideoWriter(foutput, codec, fps, (width, height))

    frame_num = 0

    tracking_id_set = set()
    vehicle_image_count = 0
    prev_tracking_id = -1

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed')
            break
        frame_num += 1
        vehicle_image_count += 1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if fframework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if fmodel == 'yolov3' and ftiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=fiou,
            score_threshold=fscore
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        # allowed_classes = ['Vehicle']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if fcount:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        current_tracking_id = -1

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            lpX1 = int(bbox[0])
            lpY1 = int(bbox[1])
            lpX2 = int(bbox[2])
            lpY2 = int(bbox[3])

            lpcentX = int(lpX2 - (lpX2 - lpX1) / 2)
            lpcentY = int(lpY2 - (lpY2 - lpY1) / 2)

            # logic to save lp, vehicle imgs
            if lpcentY > 800 and lpcentY < 900:
                current_tracking_id = track.track_id
                if prev_tracking_id is not current_tracking_id:
                    vehicle_image_count = 1
                prev_tracking_id = current_tracking_id
                # cv2.putText(frame, str(current_tracking_id), (lpcentX, lpcentY), 0, 0.75, (255,255,255), 2)
                original_vehicle = frame[lpcentY-420 : lpcentY+100, lpcentX-280 : lpcentX+280]

                cropped_lp = None

                if current_tracking_id not in tracking_id_set:
                    tracking_id_set.add(current_tracking_id)
                    cropped_lp = frame[lpY1 : lpY2, lpX1 : lpX2]
                    # original_vehicle = frame[lpcentY-420 : lpcentY+100, lpcentX-280 : lpcentX+280]

                context = save_LP_captures(cropped_lp, original_vehicle, current_tracking_id, lp_saving_dir, vehicle_image_count)
                print("======JSON======")
                print(context)

            # for LP
            cv2.rectangle(frame, (lpX1, lpY1), (lpX2, lpY2), color, 2)

            # for vehicle 
            cv2.rectangle(frame, (lpcentX-280, lpcentY-420), (lpcentX+280, lpcentY+100), color, 2)

            # to find the vertical distances of LP centroid
            # cv2.putText(frame, str(lpcentY), (lpcentX, lpcentY), 0, 0.75, (255,255,255), 2)

            cv2.rectangle(frame, (lpX1, lpY1-30), (lpX1 + (len(class_name) + len(str(track.track_id))) * 17, lpY1), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(lpX1, lpY1-10),0, 0.75, (255,255,255),2)

            # if enable info flag then print details about each track
            if finfo:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (lpX1, lpY1, lpX2, lpY2) ))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not fdont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if foutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

    # dotenv_path = Path('D:\github\LP_Capture_Tracking\core\.env')
    # load_dotenv(dotenv_path=dotenv_path)
    # load_dotenv()
    # LOCATION_ID = os.getenv('LOCATION_ID')
    # print("Environment variable:", LOCATION_ID)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
