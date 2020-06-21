#!/home/pi/tf/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from picamera import PiCamera
from time import sleep
from datetime import datetime
from pathlib import Path
from PIL import Image
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

TF_PATH='/home/pi/model-v3'
LABEL_FILE=TF_PATH+'/dict.txt'
MODEL_FILE=TF_PATH+'/model.tflite'

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]
labels = load_labels(LABEL_FILE)

print('loading interpreter')
interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

def analyze(filename):
  print('analyze %s' % filename)
  start = timer()
  with Image.open(filename).resize((width, height)) as img:
    input_data = np.expand_dims(img, axis=0)
    if floating_model:
      default_mean = 127.5
      default_std = 127.5
      input_data = (np.float32(input_data) - default_mean) / default_std
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    end = timer()
    print('that took %0.1f seconds' % (end - start))
    top_k = results.argsort()[-5:][::-1]
    for i in top_k:
      if floating_model:
        print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
      else:
        print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

print('setting up camera')
camera = PiCamera()
# if it is upside down
# camera.rotation = 180
#camera.resolution = (1440, 960)
camera.resolution = (960, 640)

p = Path('/var/www/html/capture.jpg')

while True:
  date_str = datetime.now().strftime('%Y%m%d-%H%M%S')
  filename = '/var/www/html/%s.jpg' % date_str
  print('capture %s' % filename)
  camera.capture(filename)

  analyze(filename)

  # make a link
  p.unlink()
  p.symlink_to(filename)
  # sleep(5)


