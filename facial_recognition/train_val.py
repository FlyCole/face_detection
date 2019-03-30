from network import *

train_file = 'train.tfrecords'
val_file = 'val.tfrecords'
train_num = 600*69
val_num = 200*69
EPOCH = 20
BATCH = 200
Learn_rate = [0.001, 0.0005, 0.0001]
change_step = [20, 50]

model_save_dir = "save"
summaries_save_dir = 'tmp/data'


def main():
    with tf.name_scope("load_data"):
        train_img, train_lab = data_loader(train_file, EPOCH + 5)
        train_image, train_label = tf.train.shuffle_batch([train_img, train_lab], batch_size=BATCH,
                                                          capacity=1005, min_after_dequeue=1000)
        val_img, val_lab = data_loader(val_file, EPOCH + 5)
        val_image, val_label = tf.train.shuffle_batch([val_img, val_lab], batch_size=BATCH, capacity=1005,
                                                      min_after_dequeue=1000)
    with tf.name_scope('input'):
        input_image = tf.placeholder(dtype=tf.float32, shape=[None, 96, 96, 3])
        correct_label = tf.placeholder(dtype=tf.float32, shape=[None, students_num])
        learn_rate = tf.placeholder(dtype=tf.float32)
        drop_input = tf.placeholder(dtype=tf.float32)
    predict, regular = network(input_image, drop_input)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(correct_label, 1), tf.argmax(predict, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy', accuracy)
    with tf.name_scope('loss'):
        loss = -tf.reduce_mean(correct_label * tf.log(tf.clip_by_value(predict, 1e-10, 1.0))) + regular
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
        tf.summary.scalar('learn_rate', learn_rate)
    merged = tf.summary.merge_all()
    sess = tf.InteractiveSession()
    summary_writer = tf.summary.FileWriter(summaries_save_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)
    count = 0
    max_accuracy = 0
    relate_loss = 0
    relate_epoch = 0
    for i in range(EPOCH):
        print(i, "th is train...")
        if 0 <= i <= change_step[0]:
            rate = Learn_rate[0]
        else:
            if change_step[0] < i <= change_step[1]:
                rate = Learn_rate[1]
            else:
                rate = learn_rate[2]
        for j in range(int(train_num / BATCH)):
            count += 1
            image, label = sess.run([train_image, train_label])
            # print(image[0], label[0])
            if count % 100 == 0:
                summary, train_1 = sess.run([merged, train_step],
                                            feed_dict={input_image: image, correct_label: label,
                                                       learn_rate: rate, drop_input:0.5})
                summary_writer.add_summary(summary, count)
            else:
                train_step.run(feed_dict={input_image: image, correct_label: label,
                                          learn_rate: rate, drop_input: 0.5})
        mean_loss = 0
        mean_acc = 0
        mean_regu = 0
        for j in range(int(val_num / BATCH)):
            image, label = sess.run([val_image, val_label])
            accuracy_i, regu, loss_i, = sess.run([accuracy, regular, loss],
                                                 feed_dict={input_image: image, correct_label: label,
                                                            drop_input: 1.0})
            mean_acc += accuracy_i
            mean_loss += loss_i
            mean_regu += regu
        mean_loss = mean_loss * BATCH / val_num
        mean_acc = mean_acc * BATCH / val_num
        mean_regu = mean_regu * BATCH / val_num
        if max_accuracy < mean_acc:
            max_accuracy = mean_acc
            relate_loss = mean_loss
            relate_epoch = i
            save_net(sess, model_save_dir+'/best.ckpt')
        print('Epoch: ', i, "  ", "Accuracy: ", mean_acc, "  ", "Loss: ", mean_loss, "regulater: ", mean_regu)
        print('Best Epoch: ', relate_epoch, "  ", 'Accuracy: ', max_accuracy, '  Loss: ', relate_loss)
        print('\n')
        if i % 100 == 0:
            save_net(sess, model_save_dir + '/' + str(i) + '.ckpt')
    save_net(sess, model_save_dir+'/lastmodel.ckpt')
    summary_writer.close()


if __name__ == '__main__':
    main()
