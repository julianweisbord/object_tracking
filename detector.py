import cv2
import os
import io
import sys
import time
import random
import uuid
import base64
import argparse
import json
import numpy as np
import tensorflow as tf

# Some classes are hidden for ip reasons
CLASS_NUM_MAP = {1: 'person',
                 2: '**',
                 3: '***',
                 4: '****',
                 5: '*****',
                 6: '******'}

def load_net():
    with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def

def predict(img, graph_def):
    # Runs inference on an image

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv2.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        start_time = time.time()
        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        print("time: ", time.time() - start_time)
        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])

        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.05:
                # print("classId: ", classId)
                if CLASS_NUM_MAP[classId] != 'person':
                    continue
                #print("score: ", score)
                #print("bbox: ", bbox)
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                cv2.putText(img, str(CLASS_NUM_MAP[classId]), (int(x), int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], thickness=2)
        return img


def main():
    net = load_net()

    cap = cv2.VideoCapture('chaplin.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = predict(frame, net)
            cv2.imshow('frame: ', frame)
        else:
            sys.exit()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
