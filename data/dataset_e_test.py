import h5py
import numpy as np
import torch
import os
import sys
from spatial_transform import Random_crop, ToTorchFormatTensor, Center_crop, ToTensorList
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import time
import torch.distributed as dist

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

def make_dataset(source, mode):
    #root:'./datasets/hmdb51_frames'
    #source:'./datasets/settings/hmdb51/train_rgb_split1.txt'
    if not os.path.exists(source):
        print("Setting file %s for hmdb51 dataset doesn't exist." % (source))
        sys.exit()
    else:
        rgb_samples = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()[0]
                rgb_samples.append(line_info)

        print('{}: {} sequences have been loaded'.format(mode, len(rgb_samples)))
    return rgb_samples

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SequenceDataset_test(Dataset):

    def __init__(self, h5_file, seq_len, scene_type, data_type, transform_flag=True, scdformer_resolution=None):

        self.h5_file = h5_file

        self.seq_len = seq_len

        if scdformer_resolution is None:
            self.crop_size_H = 128
            self.crop_size_W = 128
        else:
            self.crop_size_H = (int(scdformer_resolution[0]) // 128)*128
            self.crop_size_W = (int(scdformer_resolution[1]) // 128)*128

        self.crop_size = [self.crop_size_H, self.crop_size_W]
        self.scene_type = scene_type
        
        if data_type == "N_N":
            scene_len = {'a1': 168, 'a2': 116, 'a3': 125, 'a4': 298, 'b1': 256, 'b2': 250, 'b3': 219, 'b4': 250}
        elif data_type == "N_G":
            scene_len = {'a1': 50, 'a10': 110, 'a11': 148, 'a2': 85, 'a3': 110, 'a4': 110, 'a5': 110, 'a6': 110, 'a7': 110, 'a8': 110, 'a9': 110}
        else:
            scene_len = {'110': 100, '111': 100, '112': 100, '113': 100, '114': 100, '115': 100, '116': 100, '117': 100, '118': 100, '119': 100, '120': 100, '121': 100, '122': 100, '123': 100, '124': 100, '125': 100, '126': 100, '127': 100, '128': 100, '129': 100, '130': 100, '131': 100, '132': 100, '133': 100}
        
        self.num_sequences = scene_len[scene_type] - seq_len + 1
        #是否需要归一化
        # self.transform_flag = transform_flag

        if transform_flag:
            self.transform = torchvision.transforms.Compose([
                                Center_crop(self.crop_size, self.seq_len),
                                ToTensorList()
                                ])
        else:
            self.transform = torchvision.transforms.Compose([
                                ToTensorList()
                                ])

        # print("{} has {} sequences".format(self.scene_type, self.num_sequences))
    
    def open_h5(self):

        self.h5 = h5py.File(self.h5_file, "r")
    
    ## 是否需要归一化
    if False:
        def get_sequence(self, A_idx):

            rainy_frame_list, clean_frame_list, rainy_event_list = [], [], []

            # start = time.time()
            
            for i in range(self.seq_len):
                
                rainy_frame = self.h5["{}/{}/{}/{:05d}".format(self.scene_type, "1", "rainy", A_idx + i)][:][:,:,::-1].astype(np.float32) / 255.
                clean_frame = self.h5["{}/{}/{:05d}".format(self.scene_type, "gt", A_idx + i)][:][:,:,::-1].astype(np.float32) / 255.
                
                H,W,_ = rainy_frame.shape

                rainy_frame = torch.from_numpy(np.ascontiguousarray(np.transpose(rainy_frame, (2, 0, 1)))).float()
                clean_frame = torch.from_numpy(np.ascontiguousarray(np.transpose(clean_frame, (2, 0, 1)))).float()

                rainy_frame_list.append(rainy_frame)
                clean_frame_list.append(clean_frame)

                voxel = self.h5["{}/{}/{}/{:05d}".format(self.scene_type, "1", "voxel", A_idx + i)][:]
                voxel = quick_norm_event(voxel)
                voxel = torch.from_numpy(np.ascontiguousarray(voxel)).float()
                rainy_event_list.append(voxel)
                
            # print("Access Event time is {:f}".format(time.time() - start))

            if self.transform_flag:
                rnd_h = int(round((H - self.crop_size[0]) / 2.))
                rnd_w = int(round((W - self.crop_size[0]) / 2.))

                rainy_frame_list = [v[:, rnd_h:rnd_h + self.crop_size[0], rnd_w:rnd_w + self.crop_size[1]] for v in rainy_frame_list]
                clean_frame_list = [v[:, rnd_h:rnd_h + self.crop_size[0], rnd_w:rnd_w + self.crop_size[1]] for v in clean_frame_list]
                rainy_event_list = [v[:, rnd_h:rnd_h + self.crop_size[0], rnd_w:rnd_w + self.crop_size[1]] for v in rainy_event_list]

            item = {"rainy": rainy_frame_list,
                    "gt": clean_frame_list,
                    "rainy_events": rainy_event_list}

        # print("Access time is {:f}".format(time.time() - start))
        # start = time.time()
        # item = self.transform(item)

        # if dist.get_rank() == 0:
        #     print("Transform time is {:f}".format(time.time() - start))
        # print("Transform time is {:f}".format(time.time() - start))

            return item
    
    
    def get_sequence(self, A_idx):

        rainy_frame, clean_frame, rainy_event = [], [], []

        # start = time.time()
        
        for i in range(self.seq_len):
            
            rainy_frame.append(self.h5["{}/{}/{}/{:05d}".format(self.scene_type, "1", "rainy", A_idx + i)][:][:,:,::-1])
            clean_frame.append(self.h5["{}/{}/{:05d}".format(self.scene_type, "gt", A_idx + i)][:][:,:,::-1])
        
        # print("Access Frame time is {:f}".format(time.time() - start))
        # start = time.time()

        for i in range(self.seq_len):
            rainy_event.append(self.h5["{}/{}/{}/{:05d}".format(self.scene_type, "1", "voxel", A_idx + i)][:])
            
        # print("Access Event time is {:f}".format(time.time() - start))

        item = {"rainy": rainy_frame,
                "gt": clean_frame,
                "rainy_events": rainy_event}

        # print("Access time is {:f}".format(time.time() - start))
        # start = time.time()
        item = self.transform(item)

        # if dist.get_rank() == 0:
        #     print("Transform time is {:f}".format(time.time() - start))
        # print("Transform time is {:f}".format(time.time() - start))

        return item

    def __len__(self):

        return self.num_sequences - 1
    
    def __getitem__(self, idx):

        if not hasattr(self, "h5"):
            self.open_h5()

        A_idx = idx

        sequence = self.get_sequence(A_idx)

        return sequence


class SequenceDataset_test_EventRepresentation(Dataset):

    def __init__(self, seq_len, scene_type, data_type, transform_flag=True):

        self.h5_file = f"/data/booker/LN_base/TNNLS_re/{data_type}/test.h5"

        if not hasattr(self, "h5"):
            self.open_h5()

        self.seq_len = seq_len

        self.crop_size = 128

        self.scene_type = scene_type
        
        self.num_sequences = len(self.h5[f"{scene_type}/1/voxel"].keys()) - seq_len + 1

        self.transform_flag = transform_flag

        # print("{} has {} sequences".format(self.scene_type, self.num_sequences))
    
    def open_h5(self):

        self.h5 = h5py.File(self.h5_file, "r")

    def get_sequence(self, A_idx):

        rainy_frame_list, clean_frame_list, rainy_event_list = [], [], []
        
        for i in range(self.seq_len):

            rainy_frame = self.h5[f"{self.scene_type}/1/rainy/{(A_idx + i):08d}"][:][:,:,::-1].astype(np.float32) / 255.
            clean_frame = self.h5[f"{self.scene_type}/1/gt/{(A_idx + i):08d}"][:][:,:,::-1].astype(np.float32) / 255.
            
            H,W,_ = rainy_frame.shape

            rainy_frame = torch.from_numpy(np.ascontiguousarray(np.transpose(rainy_frame, (2, 0, 1)))).float()
            clean_frame = torch.from_numpy(np.ascontiguousarray(np.transpose(clean_frame, (2, 0, 1)))).float()
            
            rainy_frame_list.append(rainy_frame)
            clean_frame_list.append(clean_frame)

            voxel = self.h5["{}/{}/{}/{:08d}".format(self.scene_type, "1", "voxel", A_idx + i)][:]
            voxel = quick_norm_event(voxel) #这一步使得voxel的类型从float32到float64
            voxel = torch.from_numpy(np.ascontiguousarray(voxel)).float()
            rainy_event_list.append(voxel)

        # print("Access Frame time is {:f}".format(time.time() - start))
        # start = time.time()
            
        # print("Access Event time is {:f}".format(time.time() - start))
        
        if self.transform_flag:
            rnd_h = int(round((H - self.crop_size) / 2.))
            rnd_w = int(round((W - self.crop_size) / 2.))

            rainy_frame_list = [v[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size] for v in rainy_frame_list]
            clean_frame_list = [v[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size] for v in clean_frame_list]
            rainy_event_list = [v[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size] for v in rainy_event_list]
        
        item = {"rainy": rainy_frame_list,
                "gt": clean_frame_list,
                "rainy_events": rainy_event_list}

        return item

    def __len__(self):

        return self.num_sequences
    
    def __getitem__(self, idx):

        A_idx = idx

        sequence = self.get_sequence(A_idx)

        return sequence


class SequenceDataset_seq_test(Dataset):

    def __init__(self, h5_file, seq_len, scene_type, data_type, transform_flag=True):

        self.h5_file = h5_file

        self.seq_len = seq_len

        self.crop_size = 128

        self.scene_type = scene_type
        
        if data_type == "N_N":
            scene_len = {'a1': 168, 'a2': 116, 'a3': 125, 'a4': 298, 'b1': 256, 'b2': 250, 'b3': 219, 'b4': 250}
        elif data_type == "N_G":
            scene_len = {'a1': 50, 'a10': 110, 'a11': 148, 'a2': 85, 'a3': 110, 'a4': 110, 'a5': 110, 'a6': 110, 'a7': 110, 'a8': 110, 'a9': 110}
        else:
            scene_len = {'110': 100, '111': 100, '112': 100, '113': 100, '114': 100, '115': 100, '116': 100, '117': 100, '118': 100, '119': 100, '120': 100, '121': 100, '122': 100, '123': 100, '124': 100, '125': 100, '126': 100, '127': 100, '128': 100, '129': 100, '130': 100, '131': 100, '132': 100, '133': 100}
        
        self.num_sequences = scene_len[scene_type] // seq_len

        if transform_flag:
            self.transform = torchvision.transforms.Compose([
                                Center_crop(self.crop_size, self.seq_len),
                                ToTensorList()
                                ])
        else:
            self.transform = torchvision.transforms.Compose([
                                ToTensorList()
                                ])

        # print("{} has {} sequences".format(self.scene_type, self.num_sequences))
    
    def open_h5(self):

        self.h5 = h5py.File(self.h5_file, "r")

    def get_sequence(self, A_idx):

        rainy_frame, clean_frame, rainy_event = [], [], []

        # start = time.time()
        
        for i in range(self.seq_len):
            
            rainy_frame.append(self.h5["{}/{}/{}/{:05d}".format(self.scene_type, "1", "rainy", A_idx*(self.seq_len) + i)][:][:,:,::-1])
            clean_frame.append(self.h5["{}/{}/{:05d}".format(self.scene_type, "gt", A_idx*(self.seq_len) + i)][:][:,:,::-1])
        
        # print("Access Frame time is {:f}".format(time.time() - start))
        # start = time.time()

        for i in range(self.seq_len):
            rainy_event.append(self.h5["{}/{}/{}/{:05d}".format(self.scene_type, "1", "voxel", A_idx*(self.seq_len) + i)][:])
            
        # print("Access Event time is {:f}".format(time.time() - start))

        item = {"rainy": rainy_frame,
                "gt": clean_frame,
                "rainy_events": rainy_event}

        # print("Access time is {:f}".format(time.time() - start))
        # start = time.time()
        item = self.transform(item)

        # if dist.get_rank() == 0:
        #     print("Transform time is {:f}".format(time.time() - start))
        # print("Transform time is {:f}".format(time.time() - start))

        return item

    def __len__(self):

        return self.num_sequences - 1
    
    def __getitem__(self, idx):

        if not hasattr(self, "h5"):
            self.open_h5()

        A_idx = idx

        sequence = self.get_sequence(A_idx)

        return sequence

class SequenceDataset_test_RainDrop(Dataset):

    def __init__(self, h5_file, seq_len, scene_type, data_type, transform_flag=True, scdformer_resolution=None):

        self.h5_file = f"/data/booker/LN_base/TNNLS_re/RainDrop/TestCrop_voxel_10.h5"

        if not hasattr(self, "h5"):
            self.open_h5()

        self.seq_len = seq_len
        
        if scdformer_resolution is None:
            self.crop_size_H = 128
            self.crop_size_W = 128
        else:
            self.crop_size_H = (int(scdformer_resolution[0]) // 128)*128
            self.crop_size_W = (int(scdformer_resolution[1]) // 128)*128

        self.scene_type = scene_type
        
        self.num_sequences = len(self.h5[f"video_all/{scene_type}/voxel"].keys()) - seq_len + 1

        self.transform_flag = transform_flag

        # print("{} has {} sequences".format(self.scene_type, self.num_sequences))
    
    def open_h5(self):

        self.h5 = h5py.File(self.h5_file, "r")

    def get_sequence(self, A_idx):

        rainy_frame_list, clean_frame_list, rainy_event_list = [], [], []
        
        for i in range(self.seq_len):

            rainy_frame = self.h5[f"video_all/{self.scene_type}/rainy/{(A_idx + i):08d}"][:][:,:,::-1].astype(np.float32) / 255.
            clean_frame = self.h5[f"video_all/{self.scene_type}/gt/{(A_idx + i):08d}"][:][:,:,::-1].astype(np.float32) / 255.
            
            H,W,_ = rainy_frame.shape

            rainy_frame = torch.from_numpy(np.ascontiguousarray(np.transpose(rainy_frame, (2, 0, 1)))).float()
            clean_frame = torch.from_numpy(np.ascontiguousarray(np.transpose(clean_frame, (2, 0, 1)))).float()
            
            rainy_frame_list.append(rainy_frame)
            clean_frame_list.append(clean_frame)

            voxel = self.h5[f"video_all/{self.scene_type}/voxel/{(A_idx + i):08d}"][:]
            voxel = quick_norm_event(voxel) #这一步使得voxel的类型从float32到float64
            voxel = torch.from_numpy(np.ascontiguousarray(voxel)).float()
            rainy_event_list.append(voxel)

        # print("Access Frame time is {:f}".format(time.time() - start))
        # start = time.time()
            
        # print("Access Event time is {:f}".format(time.time() - start))
        
        if self.transform_flag:
            rnd_h = int(round((H - self.crop_size_H) / 2.))
            rnd_w = int(round((W - self.crop_size_W) / 2.))

            rainy_frame_list = [v[:, rnd_h:rnd_h + self.crop_size_H, rnd_w:rnd_w + self.crop_size_W] for v in rainy_frame_list]
            clean_frame_list = [v[:, rnd_h:rnd_h + self.crop_size_H, rnd_w:rnd_w + self.crop_size_W] for v in clean_frame_list]
            rainy_event_list = [v[:, rnd_h:rnd_h + self.crop_size_H, rnd_w:rnd_w + self.crop_size_W] for v in rainy_event_list]
        
        item = {"rainy": rainy_frame_list,
                "gt": clean_frame_list,
                "rainy_events": rainy_event_list}

        return item

    def __len__(self):

        return self.num_sequences
    
    def __getitem__(self, idx):

        A_idx = idx

        sequence = self.get_sequence(A_idx)

        return sequence


### RealRain_v1是真实雨线、真实event
class SequenceDataset_test_RealRain_v1(Dataset):

    def __init__(self, h5_file, seq_len, scene_type, data_type, transform_flag=True, scdformer_resolution=None):

        self.h5_file = f"/data/booker/LNRain_v2/Rain_DAVIS/A1/{scene_type}/event/voxel_B_d4000.h5"

        self.seq_len = seq_len
        
        if scdformer_resolution is None:
            self.crop_size_H = 128
            self.crop_size_W = 128
        else:
            self.crop_size_H = (int(scdformer_resolution[0]) // 128)*128
            self.crop_size_W = (int(scdformer_resolution[1]) // 128)*128

        self.scene_type = scene_type

        if not hasattr(self, "h5"):
            self.open_h5()
        
        voxels_len = len(self.h5[f"data"].keys()) // 2
        self.num_sequences = voxels_len - seq_len + 1

        self.transform_flag = transform_flag

        # print("{} has {} sequences".format(self.scene_type, self.num_sequences))
    
    def open_h5(self):

        self.h5 = h5py.File(self.h5_file, "r")

    def get_sequence(self, A_idx):

        rainy_frame_list, clean_frame_list, rainy_event_list = [], [], []
        
        for i in range(self.seq_len):

            rainy_frame = self.h5[f"data/input{(A_idx + i):08d}"][:][:,:,::-1].astype(np.float32) / 255.
            clean_frame = self.h5[f"data/input{(A_idx + i):08d}"][:][:,:,::-1].astype(np.float32) / 255.
            
            H,W,_ = rainy_frame.shape

            rainy_frame = torch.from_numpy(np.ascontiguousarray(np.transpose(rainy_frame, (2, 0, 1)))).float()
            clean_frame = torch.from_numpy(np.ascontiguousarray(np.transpose(clean_frame, (2, 0, 1)))).float()
            
            rainy_frame_list.append(rainy_frame)
            clean_frame_list.append(clean_frame)

            voxel = self.h5[f"data/voxel{(A_idx + i):08d}"][:]
            # voxel = quick_norm_event(voxel) #这一步使得voxel的类型从float32到float64,因为预训练模型中没有使用norm
            voxel = torch.from_numpy(np.ascontiguousarray(voxel)).float()
            rainy_event_list.append(voxel)

        # print("Access Frame time is {:f}".format(time.time() - start))
        # start = time.time()
            
        # print("Access Event time is {:f}".format(time.time() - start))
        
        if self.transform_flag:
            rnd_h = int(round((H - self.crop_size_H) / 2.))
            rnd_w = int(round((W - self.crop_size_W) / 2.))

            rainy_frame_list = [v[:, rnd_h:rnd_h + self.crop_size_H, rnd_w:rnd_w + self.crop_size_W] for v in rainy_frame_list]
            clean_frame_list = [v[:, rnd_h:rnd_h + self.crop_size_H, rnd_w:rnd_w + self.crop_size_W] for v in clean_frame_list]
            rainy_event_list = [v[:, rnd_h:rnd_h + self.crop_size_H, rnd_w:rnd_w + self.crop_size_W] for v in rainy_event_list]
        
        item = {"rainy": rainy_frame_list,
                "gt": clean_frame_list,
                "rainy_events": rainy_event_list}

        return item

    def __len__(self):

        return self.num_sequences
    
    def __getitem__(self, idx):

        A_idx = idx

        sequence = self.get_sequence(A_idx)

        return sequence

class SequenceDataset_test_RealRain_v2(Dataset):

    def __init__(self, h5_file, seq_len, scene_type, data_type, transform_flag=True, scdformer_resolution=None):

        self.h5_file = f"/data/booker/LN_base/TNNLS_re/realrain/voxel_10.h5"

        if not hasattr(self, "h5"):
            self.open_h5()

        self.seq_len = seq_len
        
        if scdformer_resolution is None:
            self.crop_size_H = 128
            self.crop_size_W = 128
        else:
            self.crop_size_H = (int(scdformer_resolution[0]) // 16)*16
            self.crop_size_W = (int(scdformer_resolution[1]) // 16)*16

        self.scene_type = scene_type
        
        self.num_sequences = len(self.h5[f"video_all/{scene_type}/voxel"].keys()) - seq_len + 1

        self.transform_flag = transform_flag

        # print("{} has {} sequences".format(self.scene_type, self.num_sequences))
    
    def open_h5(self):

        self.h5 = h5py.File(self.h5_file, "r")

    def get_sequence(self, A_idx):

        rainy_frame_list, clean_frame_list, rainy_event_list = [], [], []
        
        for i in range(self.seq_len):

            rainy_frame = self.h5[f"video_all/{self.scene_type}/images/{(A_idx + i):08d}"][:][:,:,::-1].astype(np.float32) / 255.
            clean_frame = self.h5[f"video_all/{self.scene_type}/images/{(A_idx + i):08d}"][:][:,:,::-1].astype(np.float32) / 255.
            
            H,W,_ = rainy_frame.shape

            rainy_frame = torch.from_numpy(np.ascontiguousarray(np.transpose(rainy_frame, (2, 0, 1)))).float()
            clean_frame = torch.from_numpy(np.ascontiguousarray(np.transpose(clean_frame, (2, 0, 1)))).float()
            
            rainy_frame_list.append(rainy_frame)
            clean_frame_list.append(clean_frame)

            voxel = self.h5[f"video_all/{self.scene_type}/voxel/{(A_idx + i):08d}"][:]
            # voxel = quick_norm_event(voxel) #这一步使得voxel的类型从float32到float64
            voxel = torch.from_numpy(np.ascontiguousarray(voxel)).float()
            rainy_event_list.append(voxel)

        # print("Access Frame time is {:f}".format(time.time() - start))
        # start = time.time()
            
        # print("Access Event time is {:f}".format(time.time() - start))
        
        if self.transform_flag:
            rnd_h = int(round((H - self.crop_size_H) / 2.))
            rnd_w = int(round((W - self.crop_size_W) / 2.))

            rainy_frame_list = [v[:, rnd_h:rnd_h + self.crop_size_H, rnd_w:rnd_w + self.crop_size_W] for v in rainy_frame_list]
            clean_frame_list = [v[:, rnd_h:rnd_h + self.crop_size_H, rnd_w:rnd_w + self.crop_size_W] for v in clean_frame_list]
            rainy_event_list = [v[:, rnd_h:rnd_h + self.crop_size_H, rnd_w:rnd_w + self.crop_size_W] for v in rainy_event_list]
        
        item = {"rainy": rainy_frame_list,
                "gt": clean_frame_list,
                "rainy_events": rainy_event_list}

        return item

    def __len__(self):

        return self.num_sequences
    
    def __getitem__(self, idx):

        A_idx = idx

        sequence = self.get_sequence(A_idx)

        return sequence



if __name__ == "__main__":

    val_dataset = SequenceDataset_test_RealRain_v2("", 7, current_type, data_type = self.args.test_data_type, transform_flag=self.args.transform_flag, scdformer_resolution=self.args.scdformer_resolution)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=self.args.batch_size_test, shuffle=False, pin_memory = True, drop_last=False)

    