import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

faces_path='/home/ryf/Documents/data'
size=128
total_num=57

imgs=[]
num=[]#存储对应学号
labs=[]

def getPaddingSize(img):
    h,w,_=img.shape
    top,bottom,left,right=(0,0,0,0)
    longest=max(h,w)

    if w<longest:
        tmp=longest-w
        left=tmp//2
        right=tmp-left
    elif h<longest:
        tmp=longest-h
        top=tmp//2
        bottom=tmp-top
    else:
        pass
    return top,bottom,left,right

def readData(label,path,h=size,w=size):
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            filename=path+'/'+filename

            img=cv2.imread(filename)

            top,bottom,left,right=getPaddingSize(img)
            # cv2.imshow("i", img)
            # cv2.waitKey(2)
            #将图片放大，扩充边缘
            img=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
            img=cv2.resize(img,(h,w))

            imgs.append(img)
            labs.append(label)  # 将label加入标签组

            #print(filename)

#读取图片
k=0
for filename in os.listdir(faces_path):
    pathdir=faces_path+'/'+filename
    (name, extension) = os.path.splitext(pathdir)
    num.append(name)
    print(num)
    label = np.zeros([total_num])  # 构造总文件数大小的数组
    label[len(num) - 1] = 1  # 存入一个文件name，Label对应位修改为1
    np.asarray(label,np.int32)
    readData(label,pathdir)
    print(len(imgs))
    print(label)
    k+=1

#将图片数据与标签转换成数组
imgs=np.array(imgs)
labs=np.array(labs)

#打乱数据并重新分配训练集和测试集比例为7:3
train_input,valid_input,train_output,valid_output=train_test_split(imgs,labs,test_size=0.3,random_state=0)
# np.asarray(train_input, np.float32)
# np.asarray(valid_input, np.float32)
np.asarray(train_output,np.float32)
np.asarray(valid_output,np.float32)
print(train_output.shape)
#参数：图片数据总数，图片高，宽，通道
#train_input=train_input.reshape(train_input.shape[0],size,size,3)
#valid_input=valid_input.reshape(valid_input.shape[0],size,size,3)

#数据归一化
train_input=train_input.astype(np.float16)/255.0-0.5
valid_input=valid_input.astype(np.float16)/255.0-0.5

print('train size:%s,valid size:%s'%(len(train_input),len(valid_input)))
#图片块，每次取200张图片
batch_size=200
num_batch=len(train_input)//batch_size
print(num_batch)
x = tf.placeholder(tf.float32, [None, size, size, 3])
y = tf.placeholder(tf.float32, [None, total_num])

keep_prob_5=tf.placeholder(tf.float32)
keep_prob_75=tf.placeholder(tf.float32)
#权重初始化
def weightVariable(shape):
    init=tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(init)
#偏置值初始化
def biasVariable(shape):
    init = tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(init)
#卷积操作
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#池化操作
def maxPool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# 随机选取权重不更新操作
def dropout(x,keep):
    return tf.nn.dropout(x,keep)

#CNN网络构建
def cnnLayer():
    # 第一层
    with tf.name_scope('C1_Conv'):
        W1 = weightVariable([5, 5, 3, 16])  # 卷积核大小(3,3)，输入通道(3)，输出通道或卷积核个数(3)
        b1 = biasVariable([16])
        # 卷积
        conv1 = tf.nn.relu(conv2d(x, W1) + b1)

    # 第二层
    with tf.name_scope('C1_Pool'):
        pool1=maxPool(conv1)

    # 第三层
    with tf.name_scope('C3_Conv'):
        W3 = weightVariable([5, 5, 16, 36])
        b3 = biasVariable([36])
        conv3 = tf.nn.relu(conv2d(pool1, W3) + b3)

    # 第四层
    with tf.name_scope('C4_Conv'):
        pool4 = maxPool(conv3)

    # 第五层
    with tf.name_scope('C5_Conv'):
        W5 = weightVariable([5, 5, 36, 36])
        b5 = biasVariable([36])
        conv5= tf.nn.relu(conv2d(pool4, W5) + b5)

    # 第六层
    with tf.name_scope('C6_Conv'):
        pool6 = maxPool(conv5)

    # 第七层
    with tf.name_scope('D_Flat'):
        D_Flat=tf.reshape(pool6,[-1,36*16*16])

    with tf.name_scope('Hidden_Layer'):
        W6 = weightVariable([16*16*36,128])
        b6 = biasVariable([128])
        Hidden_Layer = tf.nn.relu(tf.matmul(D_Flat,W6)+b6)
        D_dropout=dropout(Hidden_Layer,keep_prob_5)

    with tf.name_scope('out'):
        W7=weightVariable([128,total_num])
        b7=biasVariable([total_num])
        out = tf.add(tf.matmul(D_dropout, W7), b7)
        out = tf.nn.softmax(out)
    return out

#CNN网络训练
def cnnTrain():
    out=cnnLayer()
    # 计算loss
    cross_entropy=-tf.reduce_mean(y*tf.log(tf.clip_by_value(out,1e-8,1.0)))
    #cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=out)
    # 全局最优化算法
    train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    # 比较标签是否相等，再求所有数平均值，tf.cast（强制转换）
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out,1),tf.argmax(y,1)),tf.float32))
    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss',cross_entropy)
    tf.summary.scalar('accuracy',accuracy)
    merged_summary_op=tf.summary.merge_all()
    # 数据保存器的初始化
    saver=tf.train.Saver()
    #sess = tf.InteractiveSession()
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter('./png', graph=tf.get_default_graph())
        for n in range(20):
            # 每次取200（batch_size）张图片
            for i in range(num_batch):
                batch_x = train_input[i * batch_size:(i + 1) * batch_size]
                batch_y = train_output[i * batch_size:(i + 1) * batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                _,loss, summary, acc = sess.run([train_step, cross_entropy, merged_summary_op, accuracy],
                                                 feed_dict={x: batch_x, y: batch_y, keep_prob_5: 0.5})
                summary_writer.add_summary(summary, n * num_batch + i)
                # 打印loss
                print(n * num_batch + i, "loss:",loss, "train_acc:", acc)

                # 获取测试数据准确率
                # train_acc = accuracy.eval(feed_dict={x: batch_x, y: batch_y, keep_prob_5: 1.0})
                # print(n * num_batch + i, train_acc)
                # 准确率大于0.99时保存并退出
                #if(n*num_batch+i)%100==0:
                #    acc = sess.run(train_step,feed_dict={x: valid_input, y: valid_output, keep_prob_5: 1.0})
                #    print(n * num_batch + i, "test_acc", acc)
                if acc > 0.99 and n > 2:
                    saver.save(sess, './train_faces.model', global_step=n * num_batch + i)
                    sys.exit(0)
        print('accuracy less 0.99,perfect!')

cnnTrain()
