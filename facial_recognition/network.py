from file_transfer import *


def network(input, keep_pre):
    with tf.name_scope('save_image'):
        tf.summary.image("input_image", input, 10)
    with tf.name_scope("layer1"):
        with tf.name_scope("weigth"):
            w1 = tf.Variable(tf.truncated_normal([3, 3, 3, 6], stddev=0.1))
            variable_summaries(w1)
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.truncated_normal([6], stddev=0.1))
            variable_summaries(b1)
        with tf.name_scope("output"):
            conv2d1 = tf.nn.conv2d(input, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
            output1 = tf.nn.max_pool(tf.nn.relu(conv2d1), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            variable_summaries(output1)
    with tf.name_scope("layer2"):
        with tf.name_scope("weigth"):
            w2 = tf.Variable(tf.truncated_normal([3, 3, 6, 6], stddev=0.1))
            variable_summaries(w2)
        with tf.name_scope("biases"):
            b2 = tf.Variable(tf.truncated_normal([6], stddev=0.1))
            variable_summaries(b2)
        with tf.name_scope("output"):
            conv2d2 = tf.nn.conv2d(output1, w2, [1, 1, 1, 1], padding='SAME') + b2
            output2 = tf.nn.relu(conv2d2)
            variable_summaries(output2)
    with tf.name_scope('layer3'):
        with tf.name_scope('weigth'):
            w3 = tf.Variable(tf.truncated_normal([3, 3, 6, 6], stddev=0.1))
            variable_summaries(w3)
        with tf.name_scope('biases'):
            b3 = tf.Variable(tf.truncated_normal([6], stddev=0.1))
            variable_summaries(b3)
        with tf.name_scope('output'):
            conv2d3 = tf.nn.conv2d(output2, w3, [1, 1, 1, 1], padding='SAME') + b3
            output3 = tf.nn.relu(conv2d3)
    with tf.name_scope('layer4'):
        with tf.name_scope('weigth'):
            w4 = tf.Variable(tf.truncated_normal([3, 3, 6, 12], stddev=0.1))
            variable_summaries(w4)
        with tf.name_scope('biases'):
            b4 = tf.Variable(tf.truncated_normal([12], stddev=0.1))
            variable_summaries(b4)
        with tf.name_scope('output'):
            conv2d4 = tf.nn.conv2d(output3, w4, [1, 1, 1, 1], padding='SAME') + b4
            output4 = tf.nn.max_pool(conv2d4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    with tf.name_scope('layer5'):
        with tf.name_scope('weigth'):
            w5 = tf.Variable(tf.truncated_normal([3, 3, 12, 12], stddev=0.1))
            variable_summaries(w5)
        with tf.name_scope('biases'):
            b5 = tf.Variable(tf.truncated_normal([12], stddev=0.1))
            variable_summaries(b5)
        with tf.name_scope('output'):
            conv2d5 = tf.nn.conv2d(output4, w5, [1, 1, 1, 1], padding='SAME') + b5
            output5 = tf.nn.relu(conv2d5)
    with tf.name_scope('layer6'):
        with tf.name_scope('weight'):
            w6 = tf.Variable(tf.truncated_normal([3, 3, 12, 12], stddev=0.1))
            variable_summaries(w6)
        with tf.name_scope('biases'):
            b6 = tf.Variable(tf.truncated_normal([12], stddev=0.1))
            variable_summaries(b6)
        with tf.name_scope('output'):
            conv2d6 = tf.nn.conv2d(output5, w6, [1, 1, 1, 1], padding='SAME') + b6
            output6 = tf.nn.relu(conv2d6)
    with tf.name_scope('layer7'):
        with tf.name_scope('weigth'):
            w7 = tf.Variable(tf.truncated_normal([3, 3, 12, 24], stddev=0.1))
            variable_summaries(w7)
        with tf.name_scope('biases'):
            b7 = tf.Variable(tf.truncated_normal([24], stddev=0.1))
            variable_summaries(b7)
        with tf.name_scope('output'):
            conv2d7 = tf.nn.conv2d(output6, w7, [1, 1, 1, 1], padding='SAME') + b7
            output7 = tf.nn.max_pool(tf.nn.relu(conv2d7), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    with tf.name_scope('layer8'):
        with tf.name_scope('weigth'):
            w8 = tf.Variable(tf.truncated_normal([3, 3, 24, 24], stddev=0.1))
            variable_summaries(w8)
        with tf.name_scope('biases'):
            b8 = tf.Variable(tf.truncated_normal([24], stddev=0.1))
            variable_summaries(b8)
        with tf.name_scope('output'):
            conv2d8 = tf.nn.conv2d(output7, w8, [1, 1, 1, 1], padding='SAME') + b8
            output8 = tf.nn.relu(conv2d8)
    with tf.name_scope('layer9'):
        with tf.name_scope('weigth'):
            w9 = tf.Variable(tf.truncated_normal([3, 3, 24, 48], stddev=0.1))
            variable_summaries(w9)
        with tf.name_scope('biases'):
            b9 = tf.Variable(tf.truncated_normal([48], stddev=0.1))
            variable_summaries(b9)
        with tf.name_scope('output'):
            conv2d9 = tf.nn.conv2d(output8, w9, [1, 1, 1, 1], padding='SAME') + b9
            output9 = tf.nn.relu(conv2d9)
    with tf.name_scope('layer10'):
        with tf.name_scope('weigth'):
            w10 = tf.Variable(tf.truncated_normal([3, 3, 48, 48], stddev=0.1))
            variable_summaries(w10)
        with tf.name_scope('biases'):
            b10 = tf.Variable(tf.truncated_normal([48], stddev=0.1))
            variable_summaries(b10)
        with tf.name_scope('output'):
            conv2d10 = tf.nn.conv2d(output9, w10, [1, 1, 1, 1], padding='SAME') + b10
            output10 = tf.nn.max_pool(tf.nn.relu(conv2d10), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    # 将最后一层卷积与第七层卷积的输出联合作为全连接层输入
    with tf.name_scope('reshape'):
        fully_input1 = tf.reshape(output7, [-1, 12*12*24])
        fully_input2 = tf.reshape(output10, [-1, 6*6*48])
        full_input = tf.concat([fully_input1, fully_input2], 1)
    with tf.name_scope('fully_layer'):
        with tf.name_scope('weigth'):
            w11 = tf.Variable(tf.truncated_normal([12 * 12 * 24 + 6 * 6 * 48, 3072], stddev=0.1))
            variable_summaries(w11)
        with tf.name_scope('biases'):
            b11 = tf.Variable(tf.truncated_normal([3072], stddev=0.1))
            variable_summaries(b11)
        with tf.name_scope('output'):
            matmut = tf.matmul(full_input, w11) + b11
            fully_out = tf.nn.relu(matmut)
            variable_summaries(fully_out)
    with tf.name_scope("dropout_out_layer"):
        drop_out_layer = tf.nn.dropout(matmut, keep_pre)
    with tf.name_scope('output'):
        with tf.name_scope('weigth'):
            w12 = tf.Variable(tf.truncated_normal([3072, students_num], stddev=0.1))
            variable_summaries(w12)
        with tf.name_scope('biases'):
            b12 = tf.Variable(tf.truncated_normal([students_num], stddev=0.1))
            variable_summaries(b12)
        # 输出使用softmax
        with tf.name_scope('output'):
            output12 = tf.matmul(drop_out_layer, w12) + b12
            predict = tf.nn.softmax(output12)
    # 正则项
    with tf.name_scope('regular'):
        regular = 0.001*(tf.nn.l2_loss(w12) + tf.nn.l2_loss(w11)) +\
                  0.002*(tf.nn.l2_loss(w10) + tf.nn.l2_loss(w9) + tf.nn.l2_loss(w8)) + \
                  0.003*(tf.nn.l2_loss(w7) + tf.nn.l2_loss(w6) + tf.nn.l2_loss(w5)) + \
                  0.004*(tf.nn.l2_loss(w4) + tf.nn.l2_loss(w3) + (tf.nn.l2_loss(w2))) + \
                  0.005*tf.nn.l2_loss(w1)
    return predict, 1e-3*regular


# 记录日志
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(var, mean))))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
