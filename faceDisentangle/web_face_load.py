from scipy import misc
import glob
import os.path
import numpy as np

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
        labels: list
            labels: The labels (shape = (batch_size))
'''
def web_face_load(directory_file, batchSize):
        left_images = np.zeros((batchSize, 96, 96, 3), dtype=np.float32)
        right_images = np.zeros_like(left_images, dtype=np.float32)
        labels = []
        for count in range(batchSize):
            randPerson = np.random.randint(0, len(directory_file))
            [left_image_path, right_image_path] = np.random.choice(glob.glob(glob.glob(directory_file)[randPerson] + '/*.jpg'), 2)
            if os.path.isfile(left_image_path) and os.path.isfile(right_image_path):
                left_images[count] = normalization(crop2Target(misc.imread(left_image_path)))
                right_images[count] = normalization(crop2Target(misc.imread(right_image_path)))
            labels.append(randPerson)
        return left_images, right_images, labels

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
#     web_face_load("../data/image_sample", 200)