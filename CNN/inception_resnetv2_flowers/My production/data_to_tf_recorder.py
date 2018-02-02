import tensorflow as tf
import os
from PIL import Image
import numpy as np
import json

slim = tf.contrib.slim
# path
data_dir = '/Users/xiongfei/flower_photos'
pic_num_list={}

tf_record_dir='/Users/xiongfei/PycharmProjects/inception_resnetv2_flowers/My production'
#制作二进制数据
def create_training_or_testing_record(data_dir):
    global pic_num_list
    sum_traing=0
    sum_testing=0
    classes = {'daisy', 'dandelion','roses', 'sunflowers', 'tulips'}
    for index, name in enumerate(classes):
        class_path = data_dir +"/"+ name+"/"
        print("The class %s is loading"%(name))
        class_list=os.listdir(class_path)
        class_sample_num=len(class_list)
        pic_num_training=0
        pic_num_testing=0
        for img_name in class_list[:int(class_sample_num*0.8)]:
            img_path = class_path + img_name
            #tensorflow的图片读取函数
            image_data = tf.gfile.FastGFile(img_path, 'rb').read()
            image_format = b'JPEG'
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
                        'image/encoded':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data]))

                    }
                )
            )
            pic_num_training+=1
            tf_filename=name+'_train.tfrecords'
            writer = tf.python_io.TFRecordWriter(tf_filename)
            writer.write(example.SerializeToString())
            writer.close()

        for img_name in class_list[int(class_sample_num*0.8):]:
            img_path = class_path + img_name
            #tensorflow的图片读取函数
            image_data = tf.gfile.FastGFile(img_path, 'rb').read()
            image_format = b'JPEG'
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
                        'image/encoded':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data]))

                    }
                )
            )
            pic_num_testing+=1
            tf_filename=name+'_test.tfrecords'
            writer = tf.python_io.TFRecordWriter(tf_filename)
            writer.write(example.SerializeToString())
            writer.close()
        pic_num_list[name]=[pic_num_training,pic_num_testing]
        sum_traing+=pic_num_training
        sum_testing+=pic_num_testing
        #json保存
    pic_num_list['total']=[sum_traing,sum_testing]
    with open('pic_num_list.txt', 'w') as fp:
        json.dump(pic_num_list, fp)
    print("traing_sample",sum_traing)
    print("testing_sample",sum_testing)
   #读取样本的时候顺便统计样本数并且返回
if __name__ == '__main__':
    create_training_or_testing_record(data_dir)



