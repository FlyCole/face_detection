import os
import random

# 数据集位置
data_path = '/home/ryf/Documents/data'
# 需要创建的三个文件
save_train_file = 'train.txt'
save_val_file = 'val.txt'
save_test_file = 'test.txt'
# 学生标签
student = {}
student_num = 0


def crate_data(file_path):
    global student_num
    # 获取目录下每一个文件夹至student_file
    for (_, student_file, _) in os.walk(file_path):
        train_file = open(save_train_file, 'w')
        test_file = open(save_test_file, 'w')
        val_file = open(save_val_file, 'w')
        for i in range(len(student_file)):
            # 查看第i个学生是否已在学生列添加进去表中,如果不在则添加学生及其标签
            if student_file[i] not in student.keys():
                student[student_file[i]] = student_num
                # 获取每个学生文件夹下的每一张照片名字,保持至image_file
                image_file = os.listdir(file_path + '/' + student_file[i])
                # 将图片顺序打乱.
                random.shuffle(image_file)
                # 将每张照片路径及标签存储进对应的集合
                for j in range(len(image_file)):
                    if j < 600:
                        train_file.write(file_path + '/' + student_file[i] + image_file[j] + " " + str(student_num) + '\n')
                    if 600 <= j < 800:
                        val_file.write(file_path + '/' + student_file[i]  + image_file[j] + " " + str(student_num) + '\n')
                    if j >= 800:
                        test_file.write(file_path + '/' + student_file[i]  + image_file[j] + " " + str(student_num) + '\n')
                student_num = student_num + 1
        test_file.close()
        train_file.close()
        val_file.close()
        break


def main():
    crate_data(data_path)
    print(student)
    print(student_num)


if __name__ == '__main__':
    main()


