from scipy import misc
import glob
import os.path
import numpy as np


def web_face_load(directory_file, batchSize, feature="F_I_F_V"):
    '''
    Load a batch size of images from the file directory:
        input:
            file_path: The image directory
            batch_size: The batch size
        return:
            left_images: np.array
                The left image (shape = (batch_size, 96, 96, 3))
            right_images: np.array
                The right image (shape = (batch_size, 96, 96, 3))
            labels_left: list
                labels: The labels (shape = (batch_size))
            labels_right: same format as labels_left
    '''

    left_images = np.zeros((batchSize, 96, 96, 3), dtype=np.float32)
    right_images = np.zeros_like(left_images, dtype=np.float32)
    labels_left = []
    labels_right = []
    if feature == "F_I_F_V":
        for count in range(batchSize):
            randPerson_left = np.random.randint(0, len(directory_file))
            [left_image_path, right_image_path] = np.random.choice(glob.glob(glob.glob(directory_file)[randPerson_left] + '/*.jpg'), 2)
            if os.path.isfile(left_image_path) and os.path.isfile(right_image_path):
                left_images[count] = normalization(crop2Target(misc.imread(left_image_path)))
                right_images[count] = normalization(crop2Target(misc.imread(right_image_path)))
            labels_left.append(randPerson_left)
            labels_right.append(randPerson_left)
    else:
        for count in range(batchSize):
            randPerson_left = np.random.randint(0, len(directory_file))
            while True:
                randPerson_right = np.random.randint(0, len(directory_file))
                if randPerson_right != randPerson_left:
                    break
            left_image_path = np.random.choice(glob.glob(glob.glob(directory_file)[randPerson_left] + '/*.jpg'), 1)
            right_image_path = np.random.choice(glob.glob(glob.glob(directory_file)[randPerson_right] + '/*.jpg'), 1)
            if os.path.isfile(left_image_path) and os.path.isfile(right_image_path):
                left_images[count] = normalization(crop2Target(misc.imread(left_image_path)))
                right_images[count] = normalization(crop2Target(misc.imread(right_image_path)))
            labels_left.append(randPerson_left)
            labels_right.append(randPerson_right)
    return left_images, right_images, labels_left, labels_right

'''
The normalization of the input image
'''
def normalization(image):
    image = (image - np.mean(image)) / np.std(image)
    return image


'''
Randomly crop 125*125 image to 96*96
Achieve the data augmentation meanwhile
'''
def crop2Target(image):
    randomNumberX = np.random.randint(0, 28)
    randomNumberY = np.random.randint(0, 28)
    return image[randomNumberX:randomNumberX+96, randomNumberY:randomNumberY+96, :]


# '''
# Only for Unitest
# '''
# if __name__ == "__main__":
#     web_face_load("../data/image_sample", 1)