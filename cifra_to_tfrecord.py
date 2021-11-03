import glob
import os
from datetime import datetime
from typing import List
import io
import numpy as np
import PIL
from PIL import Image

import tensorflow.compat.v1 as tf


def unpickle(file_path: str):
    import pickle
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, synset, human,
                        height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = b'RGB'
  channels = 3
  image_format = b'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/synset': _bytes_feature(bytes(synset,'utf-8')),
      'image/class/text': _bytes_feature(bytes(human,'utf-8')),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(bytes(os.path.basename(filename),'utf-8')),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example



def convert_files(inpit_path: str, output_path: str):
    writer = tf.python_io.TFRecordWriter(output_path)
       
    matching_files = glob.glob(inpit_path)
    matching_files = sorted(matching_files)
    image_counter = 0

    for file_path in matching_files:
        dict = unpickle(file_path)

        labels: List[int] = dict[b"labels"]
        filenames: List[str] = dict[b"filenames"]
        data: List[np.array] = dict[b"data"]
        # synsets: List[str] = []

        assert len(labels) == len(filenames)
        assert len(labels) == len(data)
        for i in range(len(labels)):
            label = labels[i]
            filename = filenames[i].decode()
            rgb = np.array(data[i], dtype=np.uint8)
            rgb = rgb.reshape((3,32,32))
            rgb = rgb.T
            rgb = np.swapaxes(rgb, 0, 1)
            image = PIL.Image.fromarray(rgb, mode="RGB")
            # image.show()

            height = image.height
            width = image.width

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="BMP")
            image_buffer = img_byte_arr.getvalue()


            example = _convert_to_example(filename, image_buffer, label, "", "", height, width)
            writer.write(example.SerializeToString())
            image_counter += 1


            if not image_counter % 1000:
                print(f"{datetime.now()}: Processed {image_counter} images.")


input_path = "/mnt/Bolide/cifra/cifar-10-batches-py/"
output_path = "/mnt/Bolide/cifra/cifar-10-tfrecord"

os.makedirs(output_path, exist_ok=True)

convert_files(os.path.join(input_path, "data_batch*"), os.path.join(output_path, "train"))
convert_files(os.path.join(input_path, "test_batch"), os.path.join(output_path, "val"))
