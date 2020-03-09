import os
import os.path as osp
import numpy as np
from PIL import Image
import json
import torch
import torchvision
from torch.utils import data

import glob

class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='', single_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', resolution)
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)
        _meta_file = os.path.join(root, 'meta.json')
        
        with open(_meta_file) as json_file:
            _meta_data = json.load(json_file)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.image_list = {}
        self.size_480p = {}
        mx = 0
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                img_frames = sorted(os.listdir(os.path.join(self.image_dir, _video)))
                self.image_list[_video] = img_frames
                ff = os.listdir(os.path.join(self.mask_dir, _video))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, ff[0])).convert("P"))
                num_objs = len(_meta_data["videos"][_video]["objects"])
                print("Num objs ", num_objs)
                self.num_objects[_video] = num_objs
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, ff[0])).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.K = 11
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)


    def To_onehot(self, mask):
        #offset = 49
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)

        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]
        info['obj_ids'] = {}

        N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        #print("Check ", self.num_frames[video], len(self.image_list[video]))
        for f, ff  in enumerate(self.image_list[video]):
            seq_name = video
            img_file = os.path.join(self.image_dir, seq_name, ff)
            N_frames[f] = np.array(Image.open(img_file).convert('RGB'))/255.
            try:
                mf = ff
                mf = mf[-9:][:-4] # -9 for YTVOS
                mask_file = os.path.join(self.mask_dir, video, ff[-9:][:-4] + '.png')  
                print("Mask file ", mask_file)
                if os.path.exists(mask_file):
                    N_masks[f] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
                    info['obj_ids'][f] = np.unique(N_masks[f])
                else:
                    N_masks[f] = 0
                    info['obj_ids'][f] = np.array([0])

            except:
                # print('a')
                N_masks[f] = 0
        
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        #print("FS ", Fs.shape)
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            print("mask size ", Ms.shape)
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info



if __name__ == '__main__':
    pass

