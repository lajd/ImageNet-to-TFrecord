import sys
import os
import csv
from shutil import copyfile

images_dir = sys.argv[1]

label_file = sys.argv[2]

outputh_path = sys.argv[3]

image_class = {}
with open(label_file) as f:
    f.readline()
    reader = csv.reader(f, delimiter=",")
    for line in reader:
        imagename = line[0]
        label = line[1]
        cl = label.split(" ")[0]
        image_class[imagename] = cl


for filename in os.listdir(images_dir):
    filepath = os.path.join(images_dir, filename)
    imagename = os.path.splitext(filename)[0]
    if os.path.isdir(filepath):
        continue
    cl = image_class[imagename]

    dir = os.path.join(outputh_path, cl)
    if not os.path.exists(dir):
        os.makedirs(dir)

    output_file_path = os.path.join(dir, filename)
    if os.path.exists(output_file_path):
        continue
    copyfile(filepath, output_file_path)



