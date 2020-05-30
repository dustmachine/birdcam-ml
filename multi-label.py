
#
#  about 6 minutes to process and move 1000 images
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, shutil
import argparse

import numpy as np
from PIL import Image
import tensorflow as tf # TF2

TF_PATH='/Users/bradparks/Projects/birdfeeder/tf/tflite-birdcam-v1'
LABEL_FILE=TF_PATH+'/dict.txt'
MODEL_FILE=TF_PATH+'/model.tflite'

SORTED_DIR='/Users/bradparks/Projects/birdfeeder/sorted/'

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def sort_file_into_dir(path, dest, confidence):
  bucket = int(round(confidence)/10)*10
  #print('confidence={}, bucket={}'.format(confidence,bucket))
  dir = 'unknown' if confidence < 50 else '{}-{}'.format(dest, bucket)
  print('should move {} to {}'.format(path, dir))
  shutil.move(path, '{}/{}/'.format(SORTED_DIR,dir))

def mkdirs(labels):
  for dir in labels:
    for num in range(50,100,10):
      dirname = '%s-%s' % (dir,num)
      os.makedirs(SORTED_DIR+dirname,exist_ok=True)
  os.makedirs(SORTED_DIR+'unknown',exist_ok=True)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-d',
      '--dir',
      help='directory to scan',
      required=True)
  args = parser.parse_args()


  labels = load_labels(LABEL_FILE)
  mkdirs(labels)

  interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  count = 0
  directory = args.dir
  with os.scandir(directory) as it:
    for entry in it:
      # print(count, entry.path)
      count+=1

      image = entry.path
      try:
        img = Image.open(image).resize((width, height))
      except:
        print('This file does not seem like an image: {}'.format(image))
        continue

      # add N dim
      input_data = np.expand_dims(img, axis=0)

      if floating_model:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std

      interpreter.set_tensor(input_details[0]['index'], input_data)

      interpreter.invoke()

      output_data = interpreter.get_tensor(output_details[0]['index'])
      results = np.squeeze(output_data)

      top_k = results.argsort()[-1:][::-1]
      for i in top_k:
        if floating_model:
          print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
          print('Unexpected!')
          exit()
        else:
          print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
          confidence = float(results[i] / 255.0 * 100)
          best_guess = labels[i]


      sort_file_into_dir(entry.path, best_guess, confidence)

exit()

