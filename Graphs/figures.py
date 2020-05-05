import numpy as np
import matplotlib.pyplot as plt
import os.path
import cv2
import pywt
import matplotlib.gridspec as gridspec  
from net_canny import CannyNet
import torch

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

dir_path = os.path.abspath(os.path.dirname(__file__))
HR_img_path= os.path.join(dir_path,'../results/HR/Set5/x4/butterfly_HR_x4.png')
wavelet = 'haar'
save_path = os.path.join(dir_path,'WV/')

HR_img = cv2.imread(HR_img_path)
HR_img = cv2.cvtColor(HR_img,cv2.COLOR_BGR2RGB)
net = CannyNet(threshold=0.9,use_cuda=False)
edge = net(torch.unsqueeze(torch.from_numpy(HR_img).float().permute(2,0,1),0))

HR_r = HR_img[:,:,0] 
HR_g = HR_img[:,:,1]
HR_b = HR_img[:,:,2]

# coeffs2_r = pywt.swt2(HR_r, wavelet=wavelet, axes=(0,1), level=1, norm=True, )
# coeffs2_g = pywt.dwt2(HR_g, wavelet=wavelet,axes=(0,1))#level=1, norm=True, )
# coeffs2_g = pywt.dwt2(HR_g, wavelet=wavelet,axes=(0,1))#level=1, norm=True, )
# coeffs2_b = pywt.dwt2(HR_b, wavelet=wavelet,axes=(0,1))#level=1, norm=True, )

# wavelets = np.zeros((1,4,HR_r.shape[0],HR_r.shape[1]))

# wavelets[0,0,:,:], (wavelets[0,1,:,:], wavelets[0,2,:,:],wavelets[0,3,:,:]) = coeffs2_r[0]
# wavelets[1,0,:,:], (wavelets[1,1,:,:], wavelets[1,2,:,:],wavelets[1,3,:,:]) = coeffs2_g
# wavelets[2,0,:,:], (wavelets[2,1,:,:], wavelets[2,2,:,:],wavelets[2,3,:,:]) = coeffs2_b
files = ['y_swt_wavelets.png']#,'green_dwt_wavelets.png','blue_dwt_wavelets.png']
final_files = ['y.png','cr.png','cb.png']#,'green.png','blue.png']



# plt.figure(figsize = (5,5))
# gs1 = gridspec.GridSpec(2, 2)   
#  # set the spacing between axes. 
# for j in range(wavelets.shape[0]):
#     for i in range(wavelets.shape[1]):
#     # i = i + 1 # grid spec indexes from 0
#         ax1 = plt.subplot(2,2,i+1)
#         plt.imshow(wavelets[j,i,:,:])
#         plt.axis('off')
#         # plt.tight_layout()
#         # ax1.set_xticklabels([])
#         # ax1.set_yticklabels([])
#         ax1.set_aspect('equal')
#     plt.subplots_adjust(wspace=0, hspace=0)
#     # gs1.update(wspace=0.0, hspace=0.0)
#     plt.savefig(save_path+files[j])

# plt.figure(figsize = (5,5))
# gs1 = gridspec.GridSpec(2, 1)
# plt.subplot(2,1)

# for i in range (3):
#     plt.imshow(HR_img[:,:,i])
#     plt.axis('off')   
#     plt.savefig(save_path+final_files[i])

cv2.imshow('test', np.uint8(torch.squeeze(torch.squeeze(edge)).detach().numpy()))
cv2.waitKey()
plt.axis('off')   
plt.savefig(save_path+'edge.png')