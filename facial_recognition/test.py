from network import *
from file_transfer import *
import cv2

test_file = 'test.tfrecords'
test_num = 200*69
BATCH = 100
EPOCH = 1
model_save_dir = 'save/lastmodel.ckpt'
student = {'1611490': 0, '1611466': 1, '1613376': 2, '1611461': 3,
           '1611488': 4, '1611408': 5, '1610731': 6, '1611446': 7,
           '1611476': 8, '1611494': 9, '1611468': 10, 'empty': 11,
           '1511346': 12, '1611260': 13, '1611491': 14, '1611480': 15,
           '1611438': 16, '1611424': 17, '1611427': 18, '1611418': 19,
           '1711459': 20, '1611482': 21, '1611492': 22, '1611419': 23,
           '1611409': 24, '1611458': 25, '1611471': 26, '1611449': 27,
           '1611459': 28, '1611433': 29, '1611407': 30, '1610763': 31,
           '1611472': 32, '1611420': 33, '1611415': 34, '1611483': 35,
           '1611460': 36, '1611470': 37, '1611412': 38, '1611473': 39,
           '1611413': 40, '1611417': 41, '1611455': 42, '1611436': 43,
           '1611450': 44, '1611465': 45, '1611467': 46, '1611462': 47,
           '1611425': 48, '1611464': 49, '1611493': 50, '1611444': 51,
           '1611451': 52, '1611486': 53, '1613378': 54, '1611487': 55,
           '1611478': 56, '1611431': 57, '1611430': 58, '1611426': 59,
           '1611434': 60, '1611421': 61, '1613550': 62, '1611447': 63,
           '1613371': 64, '1611437': 65, '1611463': 66, '1611440': 67,
           '1611453': 68}


new_students = {k: v for v, k in student.items()}


def main():
    with tf.name_scope("load_data"):
        test_img, test_lab = data_loader(test_file, EPOCH + 5)
        test_image, test_label = tf.train.shuffle_batch([test_img, test_lab], batch_size=BATCH,
                                                        capacity=1005, min_after_dequeue=1000)
    with tf.name_scope('input'):
        input_image = tf.placeholder(dtype=tf.float32, shape=[None, 96, 96, 3])
        correct_label = tf.placeholder(dtype=tf.float32, shape=[None, students_num])
        drop_input = tf.placeholder(dtype=tf.float32)
    predict, regular = network(input_image, drop_input)
    with tf.name_scope('accuracy'):
        predict_1 = tf.argmax(predict, 1)
        correct_1 = tf.argmax(correct_label, 1)
        correct_prediction = tf.equal(correct_1, predict_1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy', accuracy)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)
    load_net(sess, model_save_dir)
    mean_acc = 0
    for j in range(int(test_num / BATCH)):
        image, label = sess.run([test_image, test_label])
        correct, predict, accuracy_i = sess.run([predict_1, correct_1, accuracy],
                                                feed_dict={input_image: image, correct_label: label,
                                                           drop_input: 1.0})
        print("真实:", new_students[correct[0]], "预测: ", new_students[int(predict[0])])
        cv2.imshow('a', image[0])
        cv2.waitKey(5000)
        mean_acc += accuracy_i
    mean_acc = mean_acc * BATCH / test_num
    print("accuracy: ", mean_acc)


if __name__ == '__main__':
    main()

