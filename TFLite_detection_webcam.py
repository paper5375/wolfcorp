import os
import argparse
import cv2
import numpy as np
from threading import Thread, Lock
import importlib.util
import time
import requests
import csv

class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        if not self.stream.isOpened():
            print("Error: Could not open webcam!")
            exit(1)
        print("Webcam initialized successfully")
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, framerate)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame
            if not grabbed:
                print("Webcam feed stopped unexpectedly")
                self.stop()

    def read(self):
        with self.lock:
            return self.frame.copy() if self.grabbed else None

    def stop(self):
        self.stopped = True
        self.stream.release()

def process_frame(frame, interpreter, input_details, output_details, labels, min_conf_threshold, imW, imH, floating_model, input_mean, input_std):
    print("Processing frame...")
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    detections = []
    current_count = 0

    for i in range(len(scores)):
        if scores[i] > min_conf_threshold and scores[i] <= 1.0 and labels[int(classes[i])] == 'person':
            ymin = int(max(1, boxes[i][0] * imH))
            xmin = int(max(1, boxes[i][1] * imW))
            ymax = int(min(imH, boxes[i][2] * imH))
            xmax = int(min(imW, boxes[i][3] * imW))
            if (xmax - xmin) > 50 and (ymax - ymin) > 50:
                confidence = int(scores[i] * 100)
                detections.append((xmin, ymin, xmax, ymax, confidence))
                current_count += 1

    if detections:
        boxes_list = [(x, y, w - x, h - y) for x, y, w, h, _ in detections]
        scores_list = [c / 100.0 for _, _, _, _, c in detections]
        indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, min_conf_threshold, 0.4)
        if len(indices) > 0:
            indices = indices.flatten()
            detections = [detections[i] for i in indices if i < len(detections)]
            current_count = len(detections)
        else:
            detections = []
            current_count = 0

    print(f"Detected {current_count} people")
    return detections, current_count

def send_count_to_wordpress(count, timestamp):
    url = 'https://wolfcorp.ajauniewhite.com/wp-json/custom/v1/people_count'
    data = {'count': count, 'timestamp': timestamp}
    print(f"Sending count to WordPress: {count} at {timestamp}")
    try:
        response = requests.post(url, json=data, auth=('wolfcorp.ajauniewhite.com', '9bsn HQaD M9Bs rHNz wUvA QHeI'))
        response.raise_for_status()
        print("Count sent successfully")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send count to WordPress: {e}")

def log_to_csv(count, timestamp):
    filename = 'people_count_log.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['timestamp', 'count'])
        writer.writerow([timestamp, count])

def main():
    print("Script started!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', required=True)
    parser.add_argument('--graph', default='detect.tflite')
    parser.add_argument('--labels', default='labelmap.txt')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--resolution', default='1280x720')
    parser.add_argument('--edgetpu', action='store_true')
    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = 'edgetpu.tflite' if args.edgetpu and args.graph == 'detect.tflite' else args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = args.threshold
    imW, imH = map(int, args.resolution.split('x'))
    use_TPU = args.edgetpu

    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    if labels and labels[0] == '???':
        del labels[0]

    interpreter = (Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')] if use_TPU else None)
                   if use_TPU else Interpreter(model_path=PATH_TO_CKPT))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model = input_details[0]['dtype'] == np.float32
    input_mean, input_std = 127.5, 127.5

    videostream = VideoStream(resolution=(imW, imH), framerate=60).start()
    time.sleep(0.1)

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    font = cv2.FONT_HERSHEY_SIMPLEX
    last_update_time = 0
    update_interval = 5

    while True:
        t1 = cv2.getTickCount()
        frame = videostream.read()
        if frame is None:
            print("No frame received, exiting loop")
            break

        detections, current_count = process_frame(frame, interpreter, input_details, output_details, labels, min_conf_threshold, imW, imH, floating_model, input_mean, input_std)

        for xmin, ymin, xmax, ymax, confidence in detections:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            label = f'person: {confidence}%'
            labelSize, baseLine = cv2.getTextSize(label, font, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), font, 0.7, (0, 0, 0), 2)

        cv2.putText(frame, f'FPS: {frame_rate_calc:.2f}', (30, 50), font, 1, (98, 189, 184), 2, cv2.LINE_AA)
        cv2.putText(frame, f'People: {current_count}', (30, 75), font, 1, (98, 189, 184), 2, cv2.LINE_AA)

        cv2.imshow('Person detector', frame)

        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
            send_count_to_wordpress(current_count, timestamp)
            log_to_csv(current_count, timestamp)
            last_update_time = current_time

        t2 = cv2.getTickCount()
        frame_rate_calc = freq / (t2 - t1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    videostream.stop()

if __name__ == "__main__":
    main()