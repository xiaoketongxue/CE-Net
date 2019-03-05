"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import scipy.misc as misc
import os
from PIL import Image

SHAPE = 512

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def check_mask_dim(mask):
    if len(np.shape(mask)) == 2:
        new_mask = np.expand_dims(mask, axis=2)

        return new_mask
    else:
        return mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask


def default_oct_DME_loader(img_path, mask_path, mode):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = misc.imresize(img, (SHAPE, SHAPE))

    # print(np.shape(img))


    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = misc.imresize(mask, (SHAPE, SHAPE))

    if len(np.shape(mask)) == 2:
        mask = np.expand_dims(mask, axis=2)

    mask = mask * 40.

    if mode == 'acc':

        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

        img, mask = randomShiftScaleRotate(img, mask,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))

        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomRotate90(img, mask)

    if mode == 'train':
        img, mask = randomHorizontalFlip(img, mask)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = check_mask_dim(mask)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)
    new_mask = np.zeros(np.shape(mask))

    new_mask[mask < 255] = 5
    new_mask[mask < 180] = 4
    new_mask[mask < 140] = 3
    new_mask[mask < 100] = 2
    new_mask[mask < 60] = 1
    new_mask[mask < 20] = 0
    # mask = abs(mask-1)
    # print(np.shape(new_mask))
    # print(np.shape(img))

    # print(np.max(new_mask))

    return img, new_mask

def default_DRIVE_loader(img_path, mask_path, mode):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = misc.imresize(img, (SHAPE, SHAPE))

    # print(np.shape(img))

    ground_truth = np.array(Image.open(mask_path))

    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = misc.imresize(ground_truth, (SHAPE, SHAPE))

    #print(np.max(mask))

    if len(np.shape(mask)) == 2:
        mask = np.expand_dims(mask, axis=2)



    if mode == 'acc':

        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

        img, mask = randomShiftScaleRotate(img, mask,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))

        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomRotate90(img, mask)

    if mode == 'train':
        img, mask = randomHorizontalFlip(img, mask)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = check_mask_dim(mask)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)
    new_mask = np.zeros(np.shape(mask))

    new_mask[mask > 128] = 1
    #print(np.max(new_mask), np.min(new_mask))

    # mask = abs(mask-1)
    # print(np.shape(new_mask))
    # print(np.shape(img))

    # print(np.max(new_mask))

    return img, new_mask



def default_oct_ER_loader(img_path, mask_path, mode):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = misc.imresize(img, (SHAPE, SHAPE))

    # print(np.shape(img))


    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = misc.imresize(mask, (SHAPE, SHAPE))

    if len(np.shape(mask)) == 2:
        mask = np.expand_dims(mask, axis=2)

    mask = mask * 22.

    if mode == 'acc':

        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

        img, mask = randomShiftScaleRotate(img, mask,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))

        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomRotate90(img, mask)

    if mode == 'train':
        img, mask = randomHorizontalFlip(img, mask)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = check_mask_dim(mask)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)
    new_mask = np.zeros(np.shape(mask))
    new_mask[mask <= 255] = 11
    new_mask[mask < 231] = 10
    new_mask[mask < 209] = 9
    new_mask[mask < 187] = 8
    new_mask[mask < 165] = 7
    new_mask[mask < 143] = 6
    new_mask[mask < 121] = 5
    new_mask[mask < 99] = 4
    new_mask[mask < 77] = 3
    new_mask[mask < 55] = 2
    new_mask[mask < 33] = 1
    new_mask[mask < 11] = 0
    # mask = abs(mask-1)
    # print(np.shape(new_mask))
    # print(np.shape(img))

    # print(np.max(new_mask))

    return img, new_mask


def default_oct_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (SHAPE, SHAPE))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (SHAPE, SHAPE))



    if len(np.shape(mask)) == 2:
        mask = np.expand_dims(mask, axis=2)


    mask = mask * 22.


    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))
    
    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))

    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    img = np.array(img, np.float32).transpose(2,0,1)/255.0
    mask = check_mask_dim(mask)
    mask = np.array(mask, np.float32).transpose(2,0,1)
    new_mask = np.zeros(np.shape(mask))
    new_mask[mask<255] = 10
    new_mask[mask<210] = 9
    new_mask[mask<185] = 8
    new_mask[mask<165] = 7
    new_mask[mask<141] = 6
    new_mask[mask<121] = 5
    new_mask[mask<99] = 4
    new_mask[mask<77] = 3
    new_mask[mask<55] = 2
    new_mask[mask<33] = 1
    new_mask[mask<11] = 0
    #mask = abs(mask-1)
    # print(np.shape(new_mask))
    # print(np.shape(img))

    return img, new_mask

def read_ORIGA_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        read_files = os.path.join(root_path, 'Set_A.txt')
    else:
        read_files = os.path.join(root_path, 'Set_B.txt')

    image_root = os.path.join(root_path, 'images')
    gt_root = os.path.join(root_path, 'masks')

    for image_name in open(read_files):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '.jpg')

        print(image_path, label_path)

        images.append(image_path)
        masks.append(label_path)

    return images, masks

def read_DRIVE_datasets(mode='train'):
    images = []
    masks = []

    if mode == 'train':
        source = 'dataset/DRIVE/training/'
    else:
        source = 'dataset/DRIVE/test/'

    image_root = os.path.join(source, 'images')
    gt_root = os.path.join(source, '1st_manual')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.tif')
        label_path = os.path.join(gt_root, image_name.split('_')[0] + '_manual1.gif')

        print(image_path, label_path)

        images.append(image_path)
        masks.append(label_path)

    return images, masks

def read_Messidor_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        read_files = os.path.join(root_path, 'train.txt')
    else:
        read_files = os.path.join(root_path, 'test.txt')

    image_root = os.path.join(root_path, 'save_image')
    gt_root = os.path.join(root_path, 'save_mask')

    for image_name in open(read_files):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.png')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '.png')

        images.append(image_path)
        masks.append(label_path)

    return images, masks

def read_RIM_ONE_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        read_files = os.path.join(root_path, 'train_files.txt')
    else:
        read_files = os.path.join(root_path, 'test_files.txt')

    image_root = os.path.join(root_path, 'RIM-ONE-images')
    gt_root = os.path.join(root_path, 'RIM-ONE-exp1')

    for image_name in open(read_files):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.png')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '-exp1.png')

        images.append(image_path)
        masks.append(label_path)

    return images, masks

def read_OCT_DME(root_path, mode='train'):

    gt_paths = []
    image_paths = []
    if mode == 'train':
        for image_index in range(45):
            image_root = os.path.join(root_path, 'volumedata')
            label_root = os.path.join(root_path, 'Label_01')

            image_path = os.path.join(image_root, str(image_index)+'.png')
            label_path = os.path.join(label_root, str(image_index)+'.png')

            image_paths.append(image_path)
            gt_paths.append(label_path)

    else:
        for image_index in range(45, 50):
            image_root = os.path.join(root_path, 'volumedata')
            label_root = os.path.join(root_path, 'Label_01')

            image_path = os.path.join(image_root, str(image_index) + '.png')
            label_path = os.path.join(label_root, str(image_index) + '.png')

            image_paths.append(image_path)
            gt_paths.append(label_path)

    assert len(gt_paths) == len(image_paths)

    return image_paths, gt_paths


def read_OCT_origin(root_path, mode='train'):

    gt_paths = []
    image_paths = []

    for root_folder in os.listdir(root_path):
        root_path_name = os.path.join(root_path, root_folder)

        gt_root = os.path.join(root_path_name, 'crop-gt')
        image_root = os.path.join(root_path_name, 'crop-images')
        # print(gt_root)
        for image_name in os.listdir(gt_root):
            if image_name.split('.')[-1] == 'png':
                gt_path = os.path.join(gt_root, image_name)

                image_path = os.path.join(image_root, image_name)
                if os.path.exists(image_path) and os.path.exists(gt_path):
                    gt_paths.append(gt_path)
                    image_paths.append(image_path)
    assert len(gt_paths) == len(image_paths)

    # for i in range(len(image_paths)):
    #     print(image_paths[i], gt_paths[i])

    # print(image_paths)
    # print(gt_paths)
    return image_paths, gt_paths

class ImageFolder(data.Dataset):

    def __init__(self,root_path, datasets='Messidor',  mode='train'):
        self.mode = mode
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        assert self.dataset in ['RIM-ONE', 'Messidor', 'ORIGA', 'OCT-origin', 'OCT-DME', 'DRIVE'], \
            "the dataset should be in 'Messidor', 'ORIGA', 'RIM-ONE', 'OCT-origin', 'DRIVE'"
        if self.dataset == 'RIM-ONE':
            self.images, self.labels = read_RIM_ONE_datasets(self.root, self.mode)
        elif self.dataset == 'Messidor':
            self.images, self.labels = read_Messidor_datasets(self.root, self.mode)
        elif self.dataset == 'ORIGA':
            self.images, self.labels = read_ORIGA_datasets(self.root, self.mode)
        elif self.dataset == 'OCT-origin':
            self.images, self.labels = read_OCT_origin(self.root, self.mode)
        elif self.dataset == 'OCT-DME':
            self.images, self.labels = read_OCT_DME(self.root, self.mode)

        elif self.dataset == 'DRIVE':
            self.images, self.labels = read_DRIVE_datasets(self.mode)

        else:
            print('Default dataset is Messidor')
            self.images, self.labels = read_Messidor_datasets(self.root, self.mode)



    def __getitem__(self, index):

        if self.dataset == 'OCT-DME':
            img, mask = default_oct_DME_loader(self.images[index], self.labels[index], mode=self.mode)
        elif self.dataset == 'DRIVE':
            img, mask = default_DRIVE_loader(self.images[index], self.labels[index], mode=self.mode)


        else:
            img, mask = default_oct_ER_loader(self.images[index], self.labels[index], mode=self.mode)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)

if __name__ == '__main__':
    ROOT = 'dataset/train_SR'

    mode = 'train'
    images, gts = read_OCT_origin(ROOT, mode)

    for i in range(len(images)):
        print(i, images[i], gts[i])
        image = misc.imread(images[i])
        img = misc.imresize(image, (SHAPE, SHAPE))
        gt = misc.imread(gts[i])

    #
    # dataset = ImageFolder(root_path=ROOT, datasets='OCT-origin')

    # image_path = 'dataset/train_SR/635.fds/oct14.png'
    # gt_path = 'dataset/train_SR/635.fds/gt_10/oct14.png'
    # image = cv2.imread(image_path)
    # gt = cv2.imread(gt_path)
    # print(image, gt)