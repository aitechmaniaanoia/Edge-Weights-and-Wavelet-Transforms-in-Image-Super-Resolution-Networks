import torch.utils.data as data
from data import common
import pywt
import pywt.data
import numpy as np
import torch
import matplotlib.pyplot as plt

class LRHRWaveletDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR'])


    def __init__(self, opt):
        super(LRHRWaveletDataset, self).__init__()
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

        assert self.paths_HR, '[Error] HR paths are empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                '[Error] HR: [%d] and LR: [%d] have different number of images.'%(
                len(self.paths_LR), len(self.paths_HR))


    def __getitem__(self, idx):
        lr, hr, lr_path, hr_path = self._load_file(idx)
        wavelet = 'haar'
        if self.train:
            lr, hr = self._get_patch(lr, hr)
            
        if (lr.shape[0]%2==1): 
            lr=lr[0:lr.shape[0]-1,:,:]
            hr=hr[0:hr.shape[0]-4,:,:]

        if (lr.shape[1]%2==1):
            lr=lr[:,0:lr.shape[1]-1,:]
            hr=hr[:,0:hr.shape[1]-4,:]
        
        coeffs2_r = pywt.swt2(lr[:,:,0], wavelet=wavelet,level=1, norm=True, axes=(0,1))
        coeffs2_g = pywt.swt2(lr[:,:,1], wavelet=wavelet,level=1, norm=True, axes=(0,1))
        coeffs2_b = pywt.swt2(lr[:,:,2], wavelet=wavelet,level=1, norm=True, axes=(0,1))

        # coeffs_lr = pywt.swtn(lr, wavelet=wavelet,level=1, norm=True, trim_approx=True, axes=(0,1))
        # coeffs_hr = pywt.swtn(hr, wavelet=wavelet,level=1, norm=True, trim_approx=True, axes=(0,1))

        coeffs2_hr_r = pywt.swt2(hr[:,:,0], wavelet=wavelet, level=1, norm=True, axes=(0,1))
        coeffs2_hr_g = pywt.swt2(hr[:,:,1], wavelet=wavelet, level=1, norm=True, axes=(0,1))
        coeffs2_hr_b = pywt.swt2(hr[:,:,2], wavelet=wavelet, level=1, norm=True, axes=(0,1))
        # LL_lr, (LH_lr, HL_lr, HH_lr) = coeffs_lr[0],coeffs_lr[1].values()
        # LL_hr, (LH_hr, HL_hr, HH_hr) = coeffs_hr[0],coeffs_hr[1].values()
        
        LL_r, (LH_r, HL_r, HH_r) = coeffs2_r[0]
        LL_g, (LH_g, HL_g, HH_g) = coeffs2_g[0]
        LL_b, (LH_b, HL_b, HH_b) = coeffs2_b[0]

        LL_hr_r, (LH_hr_r, HL_hr_r, HH_hr_r) = coeffs2_hr_r[0]
        LL_hr_g, (LH_hr_g, HL_hr_g, HH_hr_g) = coeffs2_hr_g[0]
        LL_hr_b, (LH_hr_b, HL_hr_b, HH_hr_b) = coeffs2_hr_b[0]
        
        lr_wavelet = np.stack((LL_r,LH_r,HL_r,HH_r,LL_g,LH_g,HL_g,HH_g,LL_b,LH_b,HL_b,HH_b),axis=2)
        hr_wavelet = np.stack((LL_hr_r,LH_hr_r,HL_hr_r,HH_hr_r,LL_hr_g,LH_hr_g,HL_hr_g,HH_hr_g,LL_hr_b,LH_hr_b,HL_hr_b,HH_hr_b),axis=2)

        lr_wavelet = np.ascontiguousarray(lr_wavelet.transpose((2, 0, 1)))
        hr_wavelet = np.ascontiguousarray(hr_wavelet.transpose((2, 0, 1)))

        lr_tensor, hr_tensor = torch.from_numpy(lr_wavelet), torch.from_numpy(hr_wavelet)

        return {'LR': lr_tensor, 'HR': hr_tensor, 'LR_path': lr_path, 'HR_path': hr_path}


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
