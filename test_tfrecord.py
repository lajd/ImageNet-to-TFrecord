import tensorflow as tf
import os

tfrecord_path = "/_"

files = [os.path.join(tfrecord_path, f) for f in os.listdir(tfrecord_path) if os.path.isfile(os.path.join(tfrecord_path, f))]
files = sorted(files)
dataset = tf.data.TFRecordDataset(files)

for raw_record in dataset:
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  height = example.features.feature["image/height"].int64_list.value[0]
  width = example.features.feature["image/width"].int64_list.value[0]
  filename = example.features.feature["image/filename"].bytes_list.value[0].decode("utf-8")
  image = example.features.feature["image/encoded"].bytes_list.value[0]
#   image = np.frombuffer(image)
  image = tf.reshape(image, shape=[])
  image = tf.image.decode_image(image, channels=3) 
  image = image.numpy()
  img = tf.keras.preprocessing.image.array_to_img(image)
  img.show()
