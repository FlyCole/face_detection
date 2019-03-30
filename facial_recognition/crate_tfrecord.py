import tensorflow as tf
import cv2
import numpy as np
import random


# 从之前生成的txt中读取图片及标签
def get_tfrecord(source_file, save_file):
    file = open(source_file, 'r')
    file_list = file.readlines()
    random.shuffle(file_list)
    # 获取所以txt中图片及标签
    image_name = [name.split()[0] for name in file_list]
    label_local = [name.split()[1] for name in file_list]
    image = np.zeros((len(image_name), 96, 96, 3), dtype=np.uint8)
    label = np.zeros((len(image_name), 69), dtype=np.float32)
    for i in range(len(image_name)):
        # print(image_name[i])
        try:
            image_i = cv2.imread(image_name[i])
            image[i] = cv2.resize(image_i, (96, 96))
            label[i][int(label_local[i])] = 1.0
        except cv2.error:
            print(image_name[i])
    print('Writting.......', save_file)
    writer = tf.python_io.TFRecordWriter(save_file)
    for i in range(len(image)):
        # 将图像矩阵转化为一个字符串
        img_raw = image[i].tostring()
        lab = label[i].tostring()
        # 将一个样例转化为Example Protocol Buffer，并将所有需要的信息写入数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lab])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
        # 将example写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()
    print('Writting End')


get_tfrecord('train.txt', 'train.tfrecords')
get_tfrecord('val.txt', 'val.tfrecords')
get_tfrecord('test.txt', 'test.tfrecords')