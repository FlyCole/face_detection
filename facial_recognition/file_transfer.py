import tensorflow as tf

students_num = 69


def data_loader(filename, num_epochs):
    # 装载filename(.tfrecords)文件中的数据. num_epochs表示装载的轮数(与训练轮数有关),shuffle=True表示打乱顺序,否则则不打乱.
    filename_queue = tf.train.string_input_producer([filename], num_epochs, shuffle=True)
    reader = tf.data.TFRecordDataset(filename)
    _, serialized_example = reader.read(filename_queue)
    # 将一个样例转化为Example Protocol Buffer，并按所写数据结构读取数据
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.string),
        'image_raw': tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [96*96*3])
    # 将图片转化为float类型.同时进行全局归一化预处理
    img = tf.cast(img, tf.float32)
    ave = tf.reduce_mean(img)
    img = (img - ave) / tf.sqrt(tf.reduce_mean(tf.square(img-ave)+1e-8))
    img = tf.reshape(img, [96, 96, 3])
    label = tf.decode_raw(features['label'], tf.float32)
    label = tf.reshape(label, [69])
    return img, label


# 装载模型
def load_net(sess, load_path):
    with tf.name_scope("load_model"):
        saver = tf.train.Saver()
        load = saver.restore(sess, load_path)


# 保存模型
def save_net(sess, save_path):
    with tf.name_scope("save_model"):
        saver = tf.train.Saver()
        save = saver.save(sess, save_path)
