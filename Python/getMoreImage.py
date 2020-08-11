import cv2
import os

IMG_PATH = 'C:\\Users\\Administrator\\Desktop\\jpg\\database\\train'

def read_img(source_imgpath):
    img = cv2.imread(source_imgpath, 1)
    return img


'''缩放'''
def crop_img(img, new_x, new_y):
    res = cv2.resize(img, (new_x, new_y), interpolation=cv2.INTER_AREA) #见下
    cv2.imwrite("./" + str(new_x) + '_' + str(new_y) +
                '.jpg', res)


'''旋转'''
def rotate_img(img, lable, number, rotate_angle, outputdir):

    if not os.path.exists(outputdir) and not os.path.isdir(outputdir):  #a判断当前路径是否为绝对路径或者是否为路径
        os.mkdir(outputdir)  #生成单级路径

    rows, cols = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).shape    #cvtcolor 是颜色转换参数

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imwrite(outputdir + os.path.sep + str(lable) + '_pic' + str(number) + '.jpg', dst)


if __name__ == '__main__':
    number = 0
    for f in os.listdir(IMG_PATH):
        abs_file = os.path.join(IMG_PATH,f)
        labels = f
        if os.path.isfile(abs_file):
            img = read_img(abs_file)
            crop_img(img, 100, 300)
            labels = labels.split('.')
            labels = labels[0].split('_')[0]
            curr_angle = 0
            while curr_angle < 50:
                curr_angle += 9
                rotate_img(img, labels, number, curr_angle, 'C:\\Users\\Administrator\\Desktop\\jpg\\database')
                rotate_img(img, labels, number + 1, -curr_angle, 'C:\\Users\\Administrator\\Desktop\\jpg\\database')
                number += 2