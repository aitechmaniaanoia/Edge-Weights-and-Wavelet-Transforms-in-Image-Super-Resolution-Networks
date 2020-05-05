import torch.utils.data as data
import sys
from canny.net_canny import CannyNet
from data import common
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from pytorch_wavelets import DWT


class LRHRWLDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR'])


    def __init__(self, opt):
        super(LRHRWLDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR, self.paths_LR = None, None

        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 2

        # read image list from image/binary files
        self.paths_HR = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR'])
        self.paths_LR = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR'])

        self.canny = CannyNet(threshold=0.1,use_cuda=False)
        self.dwt = DWT(wave='haar')

        assert self.paths_HR, '[Error] HR paths are empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                '[Error] HR: [%d] and LR: [%d] have different number of images.'%(
                len(self.paths_LR), len(self.paths_HR))


    def __getitem__(self, idx):
        lr, hr, lr_path, hr_path = self._load_file(idx)
    
        if self.train:
            lr, hr = self._get_patch(lr, hr)

        hr = np.ascontiguousarray(hr) 
        hr_edge = self.canny(torch.unsqueeze(torch.from_numpy(hr).float().permute(2,0,1),0)).detach()
        hr_edge = torch.squeeze(hr_edge,0)

        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.opt['rgb_range'])

        hr_dwt_LL,hr_dwt_rest = self.dwt(torch.unsqueeze(hr_tensor,0))
        hr_dwt_LL = torch.squeeze(hr_dwt_LL,dim=0)
        hr_dwt_LH = torch.squeeze(hr_dwt_rest[0][:,0,:,:,:],dim=0)
        hr_dwt_HL = torch.squeeze(hr_dwt_rest[0][:,1,:,:,:],dim=0)
        hr_dwt_HH = torch.squeeze(hr_dwt_rest[0][:,2,:,:,:],dim=0)
        hr_dwt = torch.cat((hr_dwt_LL,hr_dwt_LH,hr_dwt_HL,hr_dwt_HH),dim=0)

        return {'LR': lr_tensor, 'HR': hr_tensor, 'LR_path': lr_path, 'HR_path': hr_path,'HR_edge':hr_edge,'HR_dwt':hr_dwt}


    def __len__(self):
        if self.train:
            return len(self.paths_HR) * self.repeat
        else:
            return len(self.paths_LR)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path = self.paths_LR[idx]
        hr_path = self.paths_HR[idx]
        lr = common.read_img(lr_path, self.opt['data_type'])
        hr = common.read_img(hr_path, self.opt['data_type'])

        return lr, hr, lr_path, hr_path


    def _get_patch(self, lr, hr):

        LR_size = self.opt['LR_size']
        # random crop and augment
        lr, hr = common.get_patch(
            lr, hr, LR_size, self.scale)
        lr, hr = common.augment([lr, hr])
        lr = common.add_noise(lr, self.opt['noise'])

        return lr, hr
