import cv2
import os

# image_path 图片所在路径, save_path切割出来的图片存储路径
def get_negative(image_path, save_path):
    # 获取文件夹下所有图片
    file = os.listdir(image_path)
    # 切割每张图片的左上角128*128的区域并存储下来
    for i in range(len(file)):
        image = cv2.imread(image_path+"/"+file[i])
        img = image[:128, :128]
        cv2.imshow("a", img)
        cv2.waitKey(1)
        cv2.imwrite(save_path+"/"+str(i)+'.png', img)

get_negative("/home/chst/Documents/161144/body-bak", "/home/chst/Documents/data/empty/01")