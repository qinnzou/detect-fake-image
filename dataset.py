import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os
import json
import random
import cv2
from PIL import Image
from imageio import imread
from tqdm import tqdm
import random

def pil_loader(path):
    # print(path)
    im = Image.open(path)
    width, height = im.size   # Get dimensions
    # assert(width < height)
    left = 0
    top = (height - width)/2
    right = width
    bottom = (height + width)/2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

trans = {
    "train": transforms.Compose([  
        transforms.Resize((299,299)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    "val": transforms.Compose([ 
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "test": transforms.Compose([ 
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

class PairedDeepFakeDataset(Dataset):
    def __init__( 
        self, 
        real_dataset_path = "/media/user/deepfake/data/Retinaface/real/raw",
        fake_dataset_path = "/media/user/deepfake/data/Retinaface/Deepfakes/raw",
        json_dir = "/media/user/deepfake/detect-fake-image/jsons",
        phase = "train",
        pairs = ("/c23/", "/c40/"),
    ):    
        self.compression_level = ["raw", "c23", "c40"]
        self.phase = phase
        self.real_dataset_path = real_dataset_path
        self.fake_dataset_path = fake_dataset_path

        assert(os.path.exists(self.real_dataset_path))
        assert(os.path.exists(self.fake_dataset_path))

        self.json_path = os.path.join(json_dir, self.phase+".json")
        assert(os.path.exists(self.json_path))

        self.real_imgs = []
        self.fake_imgs = []

        self.imgs = []
        self.cls_labels = []

        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])    

        self.transform_mask = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((19, 19)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        self.pairs = pairs
        self.hq, self.lq = self.pairs
        self.create_file_list()

    def load_mask(self, path):
        return self.transform_mask(pil_loader(path))
    
    def load_img(self, path):
        return self.transform(pil_loader(path))

    def create_file_list(self):    
        real_video_names = []
        fake_video_names = []

        with open(self.json_path, 'r') as f:
            data_split = json.load(f)
        for name1, name2 in data_split:
            real_video_names.append(name1)
            real_video_names.append(name2)
            fake_video_names.append(name1+"_"+name2)
            fake_video_names.append(name2+"_"+name1)

        for video_name in tqdm(real_video_names):
            # print(video_name)
            video_path = os.path.join( 
                self.real_dataset_path, video_name)
            img_names = os.listdir(video_path)
            # random.shuffle(img_names)
            if self.phase == "train" and len(img_names) > 270:
                img_names = img_names[:270]
            if self.phase == "test":
                img_names = img_names[:140]
            for img_name in img_names:
                img_path = os.path.join(video_path, img_name)
                hq_path = img_path.replace("/c23/", self.hq)
                lq_path = img_path.replace("/c23/", self.lq)
                if os.path.exists(img_path) and os.path.exists(hq_path) and os.path.exists(lq_path):
                    self.real_imgs.append(img_path)
        
        for video_name in tqdm(fake_video_names):
            video_path = os.path.join( 
                self.fake_dataset_path, video_name)
            img_names = os.listdir(video_path)
            # random.shuffle(img_names)
            if self.phase == "train" and len(img_names) > 270:
                img_names = img_names[:270]
            if self.phase == "test":
                img_names = img_names[:140]
            for img_name in img_names :
                img_path = os.path.join(video_path, img_name)
                hq_path = img_path.replace("/c23/", self.hq)
                lq_path = img_path.replace("/c23/", self.lq)
                mask_path = img_path.replace("/c23/", "/masks/")
                if os.path.exists(img_path) and os.path.exists(hq_path) and os.path.exists(lq_path) and os.path.exists(mask_path):
                    self.fake_imgs.append(img_path)
        
        print(len(self.real_imgs), len(self.fake_imgs))
        self.imgs = self.real_imgs + self.fake_imgs 
        self.cls_labels = len(self.real_imgs) * [0] + len(self.fake_imgs) * [1]

    def __getitem__(self, index):
        cls_label = self.cls_labels[index]
        # print(self.imgs[index])
        if cls_label == 1: # false
            mask_path = self.imgs[index].replace("/c23/", "/masks/")
            msk = self.load_mask(mask_path)
            msk[msk > 0.2] = 1.0
            msk[msk < 0.2] = 0.0
        else:
            msk = torch.zeros(1,19,19)
        
        hq_path = self.imgs[index].replace("/c23/", self.hq)
        hq_img = self.load_img(hq_path)
        
        lq_path = self.imgs[index].replace("/c23/", self.lq)
        lq_img = self.load_img(lq_path)
        return hq_img, lq_img, msk, cls_label

    def __len__(self):
        return len(self.imgs)

class AllPairedDeepFakeDataset(Dataset):
    def __init__( 
        self, 
        real_dataset_path = "/media/user/deepfake/data/Retinaface/real/raw",
        fake_dataset_paths = "/media/user/deepfake/data/Retinaface/Deepfakes/raw",
        json_dir = "/media/user/deepfake/detect-fake-image/jsons",
        phase = "train",
        compression_level=("c40", "c23"),
    ):
        
        self.compression_level = ["raw", "c23", "c40"]
        self.phase = phase
        self.high_compression_level, self.low_compression_level = compression_level

        self.real_dataset_path = real_dataset_path
        self.fake_dataset_paths = fake_dataset_paths

        assert(os.path.exists(self.real_dataset_path))

        self.json_path = os.path.join(json_dir, self.phase+".json")
        assert(os.path.exists(self.json_path))

        self.real_imgs = []
        self.fake_imgs = []

        self.imgs = []
        self.cls_labels = []

        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    

        self.transform_mask = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((19, 19)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        
        self.pairs_list = [ 
            ("/raw/", "/c23/"),
            ("/raw/", "/c40/"),
            ("/c23/", "/c40/")
        ]

        self.pairs = ("/c23/", "/c40/")
        self.hq, self.lq = self.pairs
        self.db = []
        # with open(real_video_num_json, 'r') as f:

        self.create_file_list()
    
    # def load_mask(self, path):
    #     mask = cv2.imread(path, 0)
    #     mask = cv2.resize( mask.astype('float'),( 19,19 ),interpolation = cv2.INTER_CUBIC )
    #     mask /= 255
    #     mask[mask>0.1] = 1.0
    #     mask[mask<0.1] = 0.0
    #     return self.transform_mask(mask)

    def load_mask(self, path):
        return self.transform_mask(pil_loader(path))
    
    def load_img(self, path):
        return self.transform(pil_loader(path))

    def create_file_list(self):
    
        real_video_names = []
        fake_video_names = []


        with open(self.json_path, 'r') as f:
            data_split = json.load(f)
        for name1, name2 in data_split:
            real_video_names.append(name1)
            real_video_names.append(name2)
            fake_video_names.append(name1+"_"+name2)
            fake_video_names.append(name2+"_"+name1)

        for video_name in tqdm(real_video_names):
            # print(video_name)
            video_path = os.path.join( 
                self.real_dataset_path, video_name)
            img_names = os.listdir(video_path)
            random.shuffle(img_names)
            if self.phase == "train" and len(img_names) > 270:
                img_names = img_names[:270]
            if self.phase == "test":
                img_names = img_names[:140]
            for img_name in img_names :
                img_path = os.path.join(video_path, img_name)
                hq_path = img_path.replace("/c23/", self.hq)
                lq_path = img_path.replace("/c23/", self.lq)
                if os.path.exists(img_path) and os.path.exists(hq_path) and os.path.exists(lq_path):
                    self.real_imgs.append(img_path)
            # imgs = [ 
            #     os.path.join(video_path, img_name) for img_name in os.listdir(video_path)
            # ]
            # self.real_imgs += imgs
        
        for video_name in tqdm(fake_video_names):
            for fake_dataset_path in self.fake_dataset_paths:
                video_path = os.path.join( 
                    fake_dataset_path, video_name)
                img_names = os.listdir(video_path)
                random.shuffle(img_names)
                if self.phase == "train" and len(img_names) > 270:
                    img_names = img_names[:270]
                if self.phase == "test":
                    img_names = img_names[:140]
                for img_name in img_names :
                    img_path = os.path.join(video_path, img_name)
                    hq_path = img_path.replace("/c23/", self.hq)
                    lq_path = img_path.replace("/c23/", self.lq)
                    mask_path = img_path.replace("/c23/", "/masks/")
                    if os.path.exists(img_path) and os.path.exists(hq_path) and os.path.exists(lq_path) and os.path.exists(mask_path):
                        self.fake_imgs.append(img_path)
        
        # self.real_imgs = self.real_imgs * len(self.fake_dataset_paths)
        print(len(self.real_imgs), len(self.fake_imgs))
        self.imgs = self.real_imgs + self.fake_imgs 
        self.cls_labels = len(self.real_imgs) * [0] + len(self.fake_imgs) * [1]

    def __getitem__(self, index):
        cls_label = self.cls_labels[index]
        # print(self.imgs[index])
        if cls_label == 1: # false
            mask_path = self.imgs[index].replace("/c23/", "/masks/")
            
            msk = self.load_mask(mask_path)
            msk[msk > 0.2] = 1.0
            msk[msk < 0.2] = 0.0
        else:
            msk = torch.zeros(1,19,19)
        
        hq_path = self.imgs[index].replace("/c23/", self.hq)
        hq_img = self.load_img(hq_path)
        
        lq_path = self.imgs[index].replace("/c23/", self.lq)
        lq_img = self.load_img(lq_path)
        # print(hq_path, lq_path)
        return hq_img, lq_img, msk, cls_label
    def __len__(self):
        return len(self.imgs)

        
if __name__ == "__main__":
    dataset = PairedDeepFakeDataset()

    print(len(dataset))
    hq_img, lq_img, msk, cls_label = dataset[1]
    
    for _, _, _, _ in dataset:
        pass
    # 
    # for i in range(len(dataset)):
        # print(len(dataset[i]))
    # img_path = "/home/caoshenhao/develop/FaceForensics/data/images/Deepfakes/cropped_face/fake/c40/915_895/0040.jpg"
    # img = pil_loader(img_path)
    