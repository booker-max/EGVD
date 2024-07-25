"""
ep: based on dataset_e2_aug_1, but add the dataset"preprocess" in the h5file
ep1: no torch.from_numpy, because dataloader package the data in tensor
ep2: based on the ep1, but had intergrate the aug into h5
"""
from audioop import reverse

from types import GetSetDescriptorType
from torch.utils.data import Dataset
import os
import glob
import h5py
import torch
import torchvision
import numpy as np
import os, argparse, sys, math, random, glob, cv2
sys.path.append('/code/EGVD/network/utils')

from torch.utils.data import DataLoader
from data_augmentation import *
import commentjson as json

def quick_norm_event(event_tensor):

    nonzero_ev = (event_tensor != 0)
    num_nonzeros = nonzero_ev.sum()
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = event_tensor.sum() / num_nonzeros
        stddev = np.sqrt((event_tensor ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.astype(float)
        event_tensor = mask * (event_tensor - mean) / stddev
    
    return event_tensor

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw
        
    def __call__(self, img):
        if len(img.shape) == 3:
            return img[:, self.h1 : self.h2, self.w1 : self.w2]
        else:
            return img[self.h1 : self.h2, self.w1 : self.w2]

class ELNRainDataset(Dataset):
    def __init__(self, root_dir, data_type, mode, idx, num_bins=10, 
                voxel_method = {'method': 'between_frames'}, combined_voxel_channels=True):
        
        super(ELNRainDataset, self).__init__()
        self.num_bins = num_bins
        self.event_videos = []
        self.scene_idx = idx
        self.combined_voxel_channels = combined_voxel_channels  #if False: seperate p and n
        self.mode = mode

        if data_type == "NN":
            event_list_filename = "/code/EGVD/data/file/NN_train_event_file.txt"
        elif data_type == "NG":
            event_list_filename = "/code/EGVD/data/file/NG_train_event_file.txt"
        elif data_type == "NAH":
            event_list_filename = "/code/EGVD/data/file/NAH_train_event_file.txt"
        else:
            event_list_filename = "/code/EGVD/data/file/NAL_train_event_file.txt"

        with open(event_list_filename) as f:
            self.event_videos = [line.rstrip() for line in f.readlines()]

        self.event_dir = self.event_videos[self.scene_idx]
        self.load_data(self.event_dir, self.mode)
    
    def load_data(self, data_path, mode):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))
        
        # assert self.h5_file["aug_pre/event00001"].attrs["num_bins"] == self.num_bins
        # assert self.h5_file["aug_pre/event00001"].attrs["combine"] == self.combined_voxel_channels
        if mode == "train":
            self.num_frames = int(len(self.h5_file['aug_pre'].keys())/3)
        else:
            self.num_frames = self.h5_file['images'].attrs["num_images"] -1
        self.length = self.num_frames

    def get_frame(self, index, mode="train"):
        
        if mode == "train":
            frame = self.h5_file['aug_pre']['input{:05d}'.format(index)][:]
            gt_frame = self.h5_file['aug_pre']['gt{:05d}'.format(index)][:]
            event = self.h5_file['aug_pre']['event{:05d}'.format(index)][:]
        else:
            frame = self.h5_file['preprocess']['input{:05d}'.format(index)][:]
            gt_frame = self.h5_file['preprocess']['gt{:05d}'.format(index)][:]
            event = self.h5_file['preprocess']['event{:05d}'.format(index)][:]

        return frame, gt_frame, event

    def get_gt_frame(self, index):
        return self.h5_file['preprocess']['gt{:05d}'.format(index)][:]
    
    def get_event(self, index):
        return self.h5_file['preprocess']['event{:05d}'.format(index)][:]
    
    def transform_frame(self, frame, gt_frame, event):
        frame = torch.from_numpy(frame)
        gt_frame = torch.from_numpy(gt_frame)
        event = torch.from_numpy(event)

        return frame, gt_frame, event
    
    def __len__(self):
        return self.length 

    def __getitem__(self, index):
        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        
        frame, gt_frame, voxel = self.get_frame(index, self.mode)
        
        #是否需要归一化
        # voxel = quick_norm_event(voxel) #float64
        # voxel = torch.from_numpy(np.ascontiguousarray(voxel)).float() #float32

        # frame, gt_frame, voxel = self.transform_frame(frame, gt_frame, voxel)

        item = {"frame":frame,
                "gt":gt_frame,
                "events":voxel}
            
        return item

### 这里的data_type就是"event_image""pol_count""time_surfce"
class EventRepresentation(Dataset):
    def __init__(self, root_dir, data_type, mode, idx, num_bins=10, 
                voxel_method = {'method': 'between_frames'}, combined_voxel_channels=True):
        
        super(EventRepresentation, self).__init__()
        self.num_bins = num_bins
        self.event_videos = []
        self.scene_idx = idx
        self.combined_voxel_channels = combined_voxel_channels  #if False: seperate p and n
        self.mode = mode

        event_list_filename = "/code/EGVD/data/file/EventRepresentation_keys.txt"
        self.h5_file = f"/data/booker/LN_base/TNNLS_re/{data_type}/train.h5"
        
        if not hasattr(self, "h5"):
            self.open_h5()

        with open(event_list_filename) as f:
            self.event_videos = [line.rstrip() for line in f.readlines()]

        self.event_dir = self.event_videos[self.scene_idx]
        self.length = len(self.h5[f"{self.event_dir}/voxel"].keys())

    def open_h5(self):

        self.h5 = h5py.File(self.h5_file, "r")

    def load_data(self, data_path, mode):
        
        # assert self.h5_file["aug_pre/event00001"].attrs["num_bins"] == self.num_bins
        # assert self.h5_file["aug_pre/event00001"].attrs["combine"] == self.combined_voxel_channels
        if mode == "train":
            self.num_frames = len(self.h5[f"{data_path}/voxel"].keys())
        else:
            self.num_frames = self.h5_file['images'].attrs["num_images"] -1
        self.length = self.num_frames

    def get_frame(self, index, mode="train"):
        
        if mode == "train":
            frame = self.h5[f"{self.event_dir}/rainy/{index:08d}"][:][:,:,::-1].astype(np.float32) / 255.
            gt_frame = self.h5[f"{self.event_dir}/gt/{index:08d}"][:][:,:,::-1].astype(np.float32) / 255.

            frame = torch.from_numpy(np.ascontiguousarray(np.transpose(frame, (2, 0, 1)))).float()
            gt_frame = torch.from_numpy(np.ascontiguousarray(np.transpose(gt_frame, (2, 0, 1)))).float()
            
            event = self.h5[f"{self.event_dir}/voxel/{index:08d}"][:]
            event = quick_norm_event(event) #float64
            event = torch.from_numpy(np.ascontiguousarray(event)).float() #float32

        else:
            frame = self.h5['preprocess']['input{:05d}'.format(index)][:]
            gt_frame = self.h5['preprocess']['gt{:05d}'.format(index)][:]
            event = self.h5['preprocess']['event{:05d}'.format(index)][:]

        return frame, gt_frame, event

    def get_gt_frame(self, index):
        return self.h5_file['preprocess']['gt{:05d}'.format(index)][:]
    
    def get_event(self, index):
        return self.h5_file['preprocess']['event{:05d}'.format(index)][:]
    
    def transform_frame(self, frame, gt_frame, event):
        frame = torch.from_numpy(frame)
        gt_frame = torch.from_numpy(gt_frame)
        event = torch.from_numpy(event)

        return frame, gt_frame, event
    
    def __len__(self):
        return self.length 

    def __getitem__(self, index):
        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())

        frame, gt_frame, voxel = self.get_frame(index, self.mode)

        # frame, gt_frame, voxel = self.transform_frame(frame, gt_frame, voxel)

        item = {"frame":frame,
                "gt":gt_frame,
                "events":voxel}
        return item

### 这里的data_type就是"event_image""pol_count""time_surfce"
class RainDrop(Dataset):
    def __init__(self, root_dir, data_type, mode, idx, num_bins=10, 
                voxel_method = {'method': 'between_frames'}, combined_voxel_channels=True):
        
        super(RainDrop, self).__init__()
        self.num_bins = num_bins
        self.event_videos = []
        self.scene_idx = idx
        self.combined_voxel_channels = combined_voxel_channels  #if False: seperate p and n
        self.mode = mode

        event_list_filename = "/code/EGVD/data/file/EventRepresentation_key_raindrop.txt"
        self.h5_file = f"/data/booker/LN_base/TNNLS_re/RainDrop/TrainCrop_voxel_10.h5"
        
        if not hasattr(self, "h5"):
            self.open_h5()

        with open(event_list_filename) as f:
            self.event_videos = [line.rstrip() for line in f.readlines()]

        self.event_dir = self.event_videos[self.scene_idx]
        self.length = len(self.h5[f"video_all/{self.event_dir}/voxel"].keys())

    def open_h5(self):

        self.h5 = h5py.File(self.h5_file, "r")

    def load_data(self, data_path, mode):
        
        # assert self.h5_file["aug_pre/event00001"].attrs["num_bins"] == self.num_bins
        # assert self.h5_file["aug_pre/event00001"].attrs["combine"] == self.combined_voxel_channels
        if mode == "train":
            self.num_frames = len(self.h5[f"{data_path}/voxel"].keys())
        else:
            self.num_frames = self.h5_file['images'].attrs["num_images"] -1
        self.length = self.num_frames

    def get_frame(self, index, mode="train"):
        
        if mode == "train":
            frame = self.h5[f"video_all/{self.event_dir}/rainy/{index:08d}"][:][:,:,::-1].astype(np.float32) / 255.
            gt_frame = self.h5[f"video_all/{self.event_dir}/gt/{index:08d}"][:][:,:,::-1].astype(np.float32) / 255.

            frame = torch.from_numpy(np.ascontiguousarray(np.transpose(frame, (2, 0, 1)))).float()
            gt_frame = torch.from_numpy(np.ascontiguousarray(np.transpose(gt_frame, (2, 0, 1)))).float()
            
            event = self.h5[f"video_all/{self.event_dir}/voxel/{index:08d}"][:]
            event = quick_norm_event(event) #float64
            event = torch.from_numpy(np.ascontiguousarray(event)).float() #float32

        else:
            frame = self.h5['preprocess']['input{:05d}'.format(index)][:]
            gt_frame = self.h5['preprocess']['gt{:05d}'.format(index)][:]
            event = self.h5['preprocess']['event{:05d}'.format(index)][:]

        return frame, gt_frame, event

    def get_gt_frame(self, index):
        return self.h5_file['preprocess']['gt{:05d}'.format(index)][:]
    
    def get_event(self, index):
        return self.h5_file['preprocess']['event{:05d}'.format(index)][:]
    
    def transform_frame(self, frame, gt_frame, event):
        frame = torch.from_numpy(frame)
        gt_frame = torch.from_numpy(gt_frame)
        event = torch.from_numpy(event)

        return frame, gt_frame, event
    
    def __len__(self):
        return self.length 

    def __getitem__(self, index):
        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())

        frame, gt_frame, voxel = self.get_frame(index, self.mode)

        # frame, gt_frame, voxel = self.transform_frame(frame, gt_frame, voxel)

        item = {"frame":frame,
                "gt":gt_frame,
                "events":voxel}
        return item

class SequenceDataset(Dataset):
    def __init__(self, root_dir, data_type, mode, idx, sequence_length = 7, num_bins = 10, dataset_type='ELNRainDataset', opts = None, step_size = None):

        self.mode = mode
        self.L = sequence_length
        self.dataset = eval(dataset_type)(root_dir, data_type, mode, idx, num_bins)
        self.opts = opts
        self.step_size = step_size if step_size is not None else self.L

        assert(self.L > 0)
        assert(self.step_size > 0)

        if self.L >= self.dataset.length:
            self.length = 0
        else:
            self.length = (self.dataset.length - self.L) // self.step_size + 1
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
                e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """

        assert(i >= 0)
        assert(i < self.length)

        sequence = []

        k=0
        j = i * self.step_size
        item = self.dataset.__getitem__(j)
        sequence.append(item)
        
        for n in range(self.L - 1):
            k+=1
            item = self.dataset.__getitem__(j + k)
            sequence.append(item)

        return sequence

def get_args(base_path):

    save_dir = "/data/booker/LN_base/TMM/exp"
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=f"{save_dir}/EAVD_NAL")
    parser.add_argument('--local_rank', default = -1, type = int, help = "node rank for distributed training")

    args, _ = parser.parse_known_args()

    with open(base_path) as f:

        config = yaml.load(f, Loader=yaml.FullLoader)

    parser.set_defaults(**config)

    return parser.parse_args()

if __name__ == "__main__":

    import argparse, yaml, os, random
    
    base_options_path = "/code/EGVD/options/base_NAL.yaml"
    args = get_args(base_options_path)
    train_dataset = SequenceDataset(args.root_dir, "NN", "train", 0, 7, \
                                                    num_bins = args.num_bins, dataset_type="ELNRainDataset", opts = args)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers = 4, pin_memory = True)
    for iteration, sequence in enumerate(train_dataloader):
        print(iteration)

    