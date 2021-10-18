import tensorflow as tf
import os

tfrecord_path = "/mnt/Data/imagenet-object-localization-challenge/tfrecord"

files = [os.path.join(tfrecord_path, f) for f in os.listdir(tfrecord_path) if os.path.isfile(os.path.join(tfrecord_path, f))]

dataset = tf.data.TFRecordDataset(files)

for raw_record in dataset.take(10):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  filename = example.features.feature["image/filename"].bytes_list.value[0].decode("utf-8")
  image = example.features.feature["image/encoded"].bytes_list.value[0]
#   image = np.frombuffer(image)
  image = tf.reshape(image, shape=[])
  image = tf.image.decode_image(image, channels=3) 
  image = image.numpy()
  img = tf.keras.preprocessing.image.array_to_img(image)
  img.show()
