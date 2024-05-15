import os
import cv2
import random
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset

NUM_DATALOADER_WORKERS = 4 #per GPU

def create_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                        num_workers=NUM_DATALOADER_WORKERS, drop_last=True,
                                        pin_memory=False)
    
def create_dataset(settings, image_path_list):
    return Vimeo90KDataset(settings, image_path_list)

def read_image_paths_from_file(file_path):
    image_paths = []
    with open(file_path) as f:
        for line in f:
            image_paths.append(line.strip())

    assert image_paths, 'Error: HR path is empty.'
    return image_paths

def save_downscaled_image(image_path, downscaled_image_path, scale):
    img = cv2.imread(image_path)
    scaled_img = cv2.resize(img, None, fx=1./scale, fy=1./scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(downscaled_image_path, scaled_img)

def prepare_data(scale):
    hr_sequences_folder = os.path.join('vimeo_septuplet', 'sequences')
    lr_sequences_folder = os.path.join('vimeo_septuplet', f'sequences_scale_{scale}')

    # Create an lr_sequences folder
    os.makedirs(lr_sequences_folder)

    print("Generating LR Images...")
    for sequence_folder in os.listdir(hr_sequences_folder):
        hr_sequence_folder_path = os.path.join(hr_sequences_folder, sequence_folder)
        lr_sequence_folder_path = os.path.join(lr_sequences_folder, sequence_folder)

        print(f"Generating images for {lr_sequence_folder_path}...")
        for sub_sequence_folder in os.listdir(hr_sequence_folder_path):
            hr_sub_sequence_folder_path = os.path.join(hr_sequence_folder_path, sub_sequence_folder)
            lr_sub_sequence_folder_path = os.path.join(lr_sequence_folder_path, sub_sequence_folder)
            os.makedirs(lr_sub_sequence_folder_path, exist_ok=True)
            
            for image_file in os.listdir(hr_sub_sequence_folder_path):
                hr_image_path = os.path.join(hr_sub_sequence_folder_path, image_file)
                lr_image_path = os.path.join(lr_sub_sequence_folder_path, image_file)
                save_downscaled_image(hr_image_path, lr_image_path, scale)


# Modified from RSTT
'''
Vimeo dataset
'''
class Vimeo90KDataset(Dataset):
    '''
    Reading the training Vimeo dataset
    key example: train/00001/0001/im1.png
    '''

    def __init__(self, settings, image_path_list):
        super(Vimeo90KDataset, self).__init__()
        self.settings = settings
        self.image_path_list = image_path_list
        self.HR_image_shape = (3, 256, 448)
        self.LR_image_shape = (3, self.HR_image_shape[1] // self.settings["scale"], self.HR_image_shape[2] // self.settings["scale"])
        self.HR_crop_image_shape = (3, 256, 384) # both dims must be divisible by 128. This is due to input scale being 4x smaller, 3x convolution layers, and 4-pixel sliding window, meaning X/4/2/2/2/4 must be a whole number
        self.LR_crop_image_shape = (3, self.HR_crop_image_shape[1] // self.settings["scale"], self.HR_crop_image_shape[2] // self.settings["scale"])

        self.LR_num_frames = 1 + self.settings["num_frames"] // 2
        assert self.LR_num_frames > 1, 'Error: Not enough LR frames to interpolate'

        self.LR_index_list = [i * 2 for i in range(self.LR_num_frames)]

        self.HR_root = os.path.join("vimeo_septuplet", "sequences")
        self.LR_root = os.path.join("vimeo_septuplet", f"sequences_scale_{self.settings['scale']}")

    def read_img(self, path):
        """Read image using cv2.

        Args:
            path (str): path to the image.

        Returns:
            array: (H, W, C) BGR image. 
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 255.
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        return img

    def augment(self, img_list, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5

        def _augment(img):
            if hflip:
                img = img[:, ::-1, :]
            if vflip:
                img = img[::-1, :, :]
            
            return img

        return [_augment(img) for img in img_list]

    def __getitem__(self, index):
        key = self.image_path_list[index]
        name_a, name_b = key.split('/')

        # Get frame list
        HR_frames_list = list(range(1, self.settings["num_frames"] + 1))
        if random.random() < 0.5:
            HR_frames_list.reverse()
        LR_frames_list = [HR_frames_list[i] for i in self.LR_index_list]

        # Get HR images
        img_HR_list = []
        for v in HR_frames_list:           
            img_HR = self.read_img(os.path.join(self.HR_root, name_a, name_b, 'im{}.png'.format(v)))
            img_HR_list.append(img_HR)
                
        # Get LR images
        img_LR_list = []
        for v in LR_frames_list:
            img_LR = self.read_img(os.path.join(self.LR_root, name_a, name_b, 'im{}.png'.format(v)))
            img_LR_list.append(img_LR)

        # Crop to size
        _, LH, LW = self.LR_image_shape
        rnd_h_LR = random.randint(0, max(0, LH - self.HR_crop_image_shape[1]))
        rnd_w_LR = random.randint(0, max(0, LW - self.HR_crop_image_shape[2]))
        rnd_h_HR = int(rnd_h_LR * self.settings["scale"])
        rnd_w_HR = int(rnd_w_LR * self.settings["scale"])
        img_LR_list = [v[rnd_h_LR:rnd_h_LR + self.LR_crop_image_shape[1], rnd_w_LR:rnd_w_LR + self.LR_crop_image_shape[2], :] for v in img_LR_list]
        img_HR_list = [v[rnd_h_HR:rnd_h_HR + self.HR_crop_image_shape[1], rnd_w_HR:rnd_w_HR + self.HR_crop_image_shape[2], :] for v in img_HR_list]

        # Augmentation - flip, rotate
        img_list = img_LR_list + img_HR_list
        img_list = self.augment(img_list, True, True)
        img_LR_list = img_list[0:-self.settings["num_frames"]]
        img_HR_list = img_list[-self.settings["num_frames"]:]

        # Stack LR images to NHWC, N is the frame number
        img_LRs = np.stack(img_LR_list, axis=0)
        img_HRs = np.stack(img_HR_list, axis=0)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_HRs = img_HRs[:, :, :, [2, 1, 0]]
        img_LRs = img_LRs[:, :, :, [2, 1, 0]]
        img_HRs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HRs, (0, 3, 1, 2)))).float()
        img_LRs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LRs, (0, 3, 1, 2)))).float()

        return {'LRs': img_LRs, 'HRs': img_HRs, 'key': key}

    def __len__(self):
        return len(self.image_path_list)
