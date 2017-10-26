import cv2
import scipy.misc
import numpy as np
import os

def OneHot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh


def crop_resize(image_path, resize_shape=(64,64)):
    image = cv2.imread(image_path)
    height, width, channel = image.shape

    if width == height:
        resized_image = cv2.resize(image, resize_shape)
    elif width > height:
        resized_image = cv2.resize(image, (int(width * float(resize_shape[0])//height), resize_shape[1]))
        cropping_length = int( (resized_image.shape[1] - resize_shape[0]) // 2)
        resized_image = resized_image[:,cropping_length:cropping_length+resize_shape[1]]
    else:
        resized_image = cv2.resize(image, (resize_shape[0], int(height * float(resize_shape[1])/width)))
        cropping_length = int( (resized_image.shape[0] - resize_shape[1]) // 2)
        resized_image = resized_image[cropping_length:cropping_length+resize_shape[0], :]

    return resized_image/127.5 - 1

def save_visualization(X_origin, X, nh_nw, save_path='./vis/sample.jpg'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * 2 * nh_nw[1], 3))

    for n,x in enumerate(X):
        # if iterate to more than the required amount of images, end drawing
        if n >= nh_nw[0] * nh_nw[1] * 2:
            break;
        x_origin=X_origin[n]
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, (2*i)*w:(2*i)*w+w, :] = x_origin
        img[j*h:j*h+h, (2*i+1)*w:(2*i+1)*w+w, :] = x

    scipy.misc.imsave(save_path, img)

def check_create_dir(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)
    return dir


def randomPickRight(start, end, trX, trY, indexTable):
    randomList = []
    for i in range(start, end):
        while True:
            randomPick = np.random.choice(indexTable[trY[i]], 1)[0]
            if randomPick == i:
                continue
            else:
                randomList.append(randomPick)
            break
    return trX[randomList]