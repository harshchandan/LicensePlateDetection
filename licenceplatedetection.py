#Created in Google Colab
#Detects Licence plates from images of cars using TFOD

from google.colab import drive
drive.mount('/content/drive/')



#Setting up the environment
!pip install tensorflow-gpu
!pip install opencv-python==4.5.4.60

import tensorflow as tf
import cv2
import os
import shutil



#TensorFlow Object Detection API Installation
cd /content/drive/MyDrive/carPlate

!git clone https://github.com/tensorflow/models.git

cd /content/drive/MyDrive/carPlate/models/research

!protoc object_detection/protos/*.proto --python_out=.

!git clone https://github.com/cocodataset/cocoapi.git

cd /content/drive/MyDrive/carPlate/models/research/cocoapi/PythonAPI

!make

cp -r pycocotools /content/drive/MyDrive/carPlate/models/research

cd /content/drive/MyDrive/carPlate/models/research

!cp object_detection/packages/tf2/setup.py .

!python -m pip install --use-feature=2020-resolver .

!python object_detection/builders/model_builder_tf2_test.py



#Creating separating training and test images and xml files
f = open('/content/drive/MyDrive/carPlate/VOC2007/ImageSets/Main/trainval.txt', 'rt')
f_read = f.read()
train_list = f_read.splitlines()
f.close()

f = open('/content/drive/MyDrive/carPlate/VOC2007/ImageSets/Main/test.txt', 'rt')
f_read = f.read()
test_list = f_read.splitlines()
f.close()

def copy_files(folderPath, destination, list_file):
  for filename in list_file:
    if filename in os.listdir(folderPath):
      filename = os.path.join(folderPath, filename)
      shutil.copy(filename, destination)
    else:
      print("file does not exist:", filename)

train_list_jpeg = [s + ".jpeg" for s in train_list]
train_list_xml = [s + ".xml" for s in train_list]

copy_files('/content/drive/MyDrive/carPlate/VOC2007/Annotations','/content/drive/MyDrive/carPlate/VOC2007/images/train',train_list_xml)
copy_files('/content/drive/MyDrive/carPlate/VOC2007/JPEGImages','/content/drive/MyDrive/carPlate/VOC2007/images/train',train_list_jpeg)

test_list_jpeg = [s + ".jpeg" for s in test_list]
test_list_xml = [s + ".xml" for s in test_list]

copy_files('/content/drive/MyDrive/carPlate/VOC2007/Annotations','/content/drive/MyDrive/carPlate/VOC2007/images/test',test_list_xml)
copy_files('/content/drive/MyDrive/carPlate/VOC2007/JPEGImages','/content/drive/MyDrive/carPlate/VOC2007/images/test',test_list_jpeg)



#Importing pre-trained model from TensorFlow Object Detection Model Zoo for Transfer Learning
cd /content/drive/MyDrive/carPlate/VOC2007/pre-trained-models

!wget http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz

!tar -xvf centernet_hg104_512x512_coco17_tpu-8.tar.gz




# Creating train and test tfrecord data for training the model:
cd /content/drive/MyDrive/carPlate/VOC2007

!python generate_tfrecord.py -x /content/drive/MyDrive/carPlate/VOC2007/images/train -l /content/drive/MyDrive/carPlate/VOC2007/label_map.pbtxt -o /content/drive/MyDrive/carPlate/VOC2007/train.record

!python generate_tfrecord.py -x /content/drive/MyDrive/carPlate/VOC2007/images/test -l /content/drive/MyDrive/carPlate/VOC2007/label_map.pbtxt -o /content/drive/MyDrive/carPlate/VOC2007/test.record



#Training the model
!python model_main_tf2.py --model_dir=/content/drive/MyDrive/carPlate/VOC2007/my_centernet_hg104 --pipeline_config_path=/content/drive/MyDrive/carPlate/VOC2007/my_centernet_hg104/pipeline.config



#Saving and exporting the trained model
!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path /content/drive/MyDrive/carPlate/VOC2007/my_centernet_hg104/pipeline.config --trained_checkpoint_dir /content/drive/MyDrive/carPlate/VOC2007/my_centernet_hg104 --output_directory /content/drive/MyDrive/carPlate/VOC2007/exported-models/my_models/model2
