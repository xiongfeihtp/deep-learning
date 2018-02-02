import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import time

slim = tf.contrib.slim

# 目录，数据，log，checkpoint，标注文件（并且将标注文件转化为字典）
dataset_dir = '.'
log_dir = './log'

# 载入的文件格式,方便处理
file_pattern = './*_train.tfrecords' #文件名格式
# 注释
# Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.
items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either tulips, sunflowers, roses, dandelion, or daisy.',
    'label': 'A label that is as such -- 0:daisy, 1:dandelion, 2:roses, 3:sunflowers, 4:tulips'
}
num_epochs = 1
batch_size = 8
image_size = 299
num_samples=2934
num_classes = 5

# 学习率设置
inil_lr = 0.0002
lr_decay_factor = 0.7
num_epochs_before_decay = 2
def get_split(split_name, file_pattern=file_pattern,num_samples=num_samples):
    reader = tf.TFRecordReader
    # 恢复成原始数据
    # 映射方式
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),  # Image()
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),  # Image()
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))  # Tensor
    }
    # 转化为高级的数据格式
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    dataset = slim.dataset.Dataset(
        data_sources=file_pattern,
        decoder=decoder,
        reader=reader,
        num_readers=4,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions
    )
    return dataset


# 数据预处理和batch
def load_batch(dataset, batch_size, height=image_size, width=image_size, is_training=True):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity=24 + 3 * batch_size,
        common_queue_min=24
    )
    raw_image, label = data_provider.get(['image', 'label'])
    image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    raw_image, labels = tf.train.batch(
        [raw_image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=4 * batch_size,
        allow_smaller_final_batch=True
    )
    return raw_image, labels

def run():
    # 创建对应的log目录
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # 设置log等级
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        # 准备数据，按照队列载入batch
        dataset = get_split('train',file_pattern)
        images,labels = load_batch(dataset, batch_size=batch_size)
        # 训练次数计算
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        # 网络搭建
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(tf.to_float(images), num_classes=dataset.num_classes, is_training=True)


        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        # labels转化为one-hot形式
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

        # 损失函数设置
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        total_loss = tf.losses.get_total_loss()  # 加入正则项之后的loss

        global_step = get_or_create_global_step()

        # 设置学习率和学习方法
        lr = tf.train.exponential_decay(
            learning_rate=inil_lr,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=lr_decay_factor,
            staircase=True
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # 指标
        predictions = tf.argmax(end_points['Predictions'], 1)
        prob = end_points['Predictions']  #置信度
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update, prob)

        # summary，准备添加到supervisor
        tf.summary.scalar('loss/Total_loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('lr', lr)
        my_summary_op = tf.summary.merge_all()

        def train_step(sess, train_op, global_step):
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)
            return total_loss, global_step_count

        saver = tf.train.Saver(variables_to_restore)
        sv = tf.train.Supervisor(logdir=log_dir, summary_op=None,save_model_secs=60) #1min save the model for test
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * num_epochs):
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)

                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                        [logits, prob, predictions, labels])
                    print('logits: \n', logits_value)
                    print('Probabilities: \n', probabilities_value)
                    print('predictions: \n', predictions_value)
                    print('Labels:\n:', labels_value)
                # Log the summaries every 10 step.
                if step % 10 == 0:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                # If not, simply run the training step
                else:
                    loss, _ = train_step(sess, train_op, sv.global_step)

            # We log the final training loss and accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))
            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            # saver.save(sess, "./flowers_model.ckpt")
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
if __name__ == '__main__':
    run()
