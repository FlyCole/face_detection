from network import *
import cv2
import dlib
import tensorflow as tf
import numpy as np

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
    sess = tf.InteractiveSession()
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('output.avi', fourcc, 45, (640, 480))
    input = tf.placeholder(tf.float32, shape=[None, 96,  96, 3])
    drop_input = tf.placeholder(dtype=tf.float32)
    pre, _ = network(input, drop_input)
    name = tf.argmax(pre, 1)
    sess.run(tf.initialize_all_variables())
    load_net(sess, 'save/best.ckpt')
    while True:
        ret, frame = cap.read()
        local = get_face(frame)
        print(local)
        data = np.zeros((len(local), 96, 96, 3), np.uint8)
        for i in range(len(local)):
            try:
                # img2 = frame[local[i][0]:local[i][1], local[i][2]:local[i][3]]
                cv2.rectangle(frame, (local[i][2], local[i][0]), (local[i][3], local[i][1]), (255, 0, 0), 5)
                img = cv2.resize(frame[local[i][0]:local[i][1], local[i][2]:local[i][3]], (96, 96))
                img = np.array(img, np.float32)
                ave = np.mean(img)
                GCN = np.sqrt(np.mean(np.square(img - ave)+1e-8))
                data[i] = (img - ave) / GCN
            except cv2.error:
                print("Flase")
        predict, w = sess.run([pre, name], feed_dict={input: data, drop_input: 1.0})
        for i in range(len(w)):
            print(new_students[w[i]])
            print(predict[i][w[i]])
            frame = cv2.putText(frame, new_students[w[i]], (local[i][3], local[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow('img', frame)
        videoWriter.write(frame)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def get_face(img):
    detector = dlib.get_frontal_face_detector()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    local = []
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        local.append([x1, y1, x2, y2])
    return local


if __name__ == '__main__':
    main()
