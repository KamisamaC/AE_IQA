import os 
import torch 
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
import PIL 
import numpy as np 
import h5py
from scipy.signal import convolve2d


def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P,Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln


def OverlappingCropPatches(im, patch_size=224, stride=224):
    w, h = im.size
    patches = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patch = LocalNormalization(patch[0].numpy())
            patches = patches + (patch,)
    return patches

class LIVE(Dataset):
    def __init__(self,conf,status = 'train'):
        im_dir = conf['im_dir']
        self.patch_size = conf['patch_size']
        self.stride = conf['stride']
        datainfo = conf['datainfo']
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        Info = h5py.File(datainfo,'r')
        ref_ids = Info['ref_ids'][0, :] #
        index = Info['index'][:,1%1000]
        train_ratio = conf['train_ratio']

        trainindex = index[:int(train_ratio * len(index))]
        valindex = index[int((1-train_ratio) * len(index)):]
        testindex = index[len(index):]
        train_index,val_index,test_index = [],[],[]
        for i in range(len(ref_ids)):
             if (ref_ids[i] in trainindex):
                train_index.append(i)
             elif (ref_ids[i] in valindex) :
                    val_index.append(i) 

        for i in range(len(ref_ids)):
            if (ref_ids[i] in testindex):
                test_index.append(i)
        
        if status == 'train':
            self.index = train_index
        if status == 'val':
            self.index = val_index
        
        if status == 'test':
            self.index = test_index

        self.mos = Info['subjective_scores'][0, self.index] #
        self.mos_std = Info['subjective_scoresSTD'][0, self.index] #
        im_names = [Info[Info['im_names'][0, :][i]].value.tobytes()\
                        [::2].decode() for i in self.index]

        self.patches = ()
        self.label = []
        self.label_std = []
        for idx in range(len(self.index)):
            im = PIL.Image.open(os.path.join(im_dir, im_names[idx]))
            patches = OverlappingCropPatches(im, self.patch_size, self.stride)
            if status == 'train':
                self.patches = self.patches + patches #
                for i in range(len(patches)):
                    self.label.append(1-self.mos[idx])
                    self.label_std.append(1-self.mos_std[idx])
            else:
                self.patches = self.patches + (torch.stack(patches),)  #
                self.label.append(1-self.mos[idx])
                self.label_std.append(1-self.mos_std[idx])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return (self.patches[idx], (torch.Tensor([self.label[idx],]),
                torch.Tensor([self.label_std[idx],])))
        


        
