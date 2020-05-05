import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from PIL import ImageFilter
from utils import calc_patch_size, convert_rgb_to_y
import pywt
#from sklearn.preprocessing import normalize
from skimage.transform import resize

def waveletweight(hr_img, weight):
    coeffs2 = pywt.dwt2(hr_img, 'haar')
    #LL, (LH, HL, HH) = coeffs2 # ['Approximation', ' Horizontal detail','Vertical detail', 'Diagonal detail']
    hr_edge = abs(coeffs2[1][0]) + abs(coeffs2[1][1]) + abs(coeffs2[1][2])

    # normalized 
    hr_edge /= np.max(hr_edge)
    #hr_edge = hr_edge / np.linalg.norm(hr_edge)
    
    mask = np.zeros(hr_edge.shape)
    mask[hr_edge > np.mean(hr_edge)] = 1
    
    #weighted_edge = weight*hr_edge + 1
    
    return mask #weighted_edge


@calc_patch_size
def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []
    hr_edges_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_images = []

        if args.with_aug:
            for s in [1.0, 0.9, 0.8, 0.7, 0.6]:
                for r in [0, 90, 180, 270]:
                    tmp = hr.resize((int(hr.width * s), int(hr.height * s)), resample=pil_image.BICUBIC)
                    tmp = tmp.rotate(r, expand=True)
                    hr_images.append(tmp)
        else:
            hr_images.append(hr)

        for hr in hr_images:
            hr_width = (hr.width // args.scale) * args.scale
            hr_height = (hr.height // args.scale) * args.scale
            
            hr_edge = hr.convert('L').filter(ImageFilter.FIND_EDGES)
            
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            #hr_edge = hr_edge.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
            
            hr = np.array(hr).astype(np.float32)
            hr_edge = np.array(hr_edge).astype(np.float32)
            
            # convert edges to grayscale
            #hr_edge = 0.2989 * hr_edge[:,:,0] + 0.5870 * hr_edge[:,:,1] + 0.1140 * hr_edge[:,:,2]
            
            # convert edge to 0/1 mask
            hr_edge[hr_edge > 0] = 1
            
            lr = np.array(lr).astype(np.float32)
            
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)
                           
            #hr_edge = convert_rgb_to_y(hr_edge)
            ########################## EDGES WEIGHT #############################
            #hr_edge[hr_edge > 0] = 1.5
            #hr_edge[hr_edge == 0] = 0.5
            
            # hr_edge = hr_edge / 255.0
            # threshold = 0.05
            # hr_edge[hr_edge > threshold] = 2.0
            # hr_edge[hr_edge <= threshold] = 1.0
            
            # # CODE TO MAKE THE WEIGHTS DYNAMIC
            # hr_edge -= 1
            # n_edges = len((hr_edge == 1).nonzero()[0])
            # n_plain = len((hr_edge == 0).nonzero()[0])
            # weight = n_plain // n_edges
            # hr_edge *= weight
            # hr_edge += 1
            
            # import matplotlib.pyplot as plt
            # plt.plot(hr_edge)
            # plt.show()
            
            ######################## wavelet #####################
            weighted_edge = waveletweight(hr, 1)
            weighted_edge = resize(weighted_edge, (hr_height, hr_width))
            
            clip_size = 40
            
            ## clip iamges as small part
            # for i in range(0, lr.shape[0] - args.patch_size + 1, args.scale):
            #     for j in range(0, lr.shape[1] - args.patch_size + 1, args.scale):
            #         lr_patches.append(lr[i:i+args.patch_size, j:j+args.patch_size])
            #         hr_patches.append(hr[i*args.scale:i*args.scale+args.patch_size*args.scale, j*args.scale:j*args.scale+args.patch_size*args.scale])
            #         hr_edges_patches.append(hr_edge[i*args.scale:i*args.scale+args.patch_size*args.scale, j*args.scale:j*args.scale+args.patch_size*args.scale])
                    
            for i in range(0, lr.shape[0] - args.patch_size + 1, clip_size):
                for j in range(0, lr.shape[1] - args.patch_size + 1, clip_size):
                    lr_patches.append(lr[i:i+args.patch_size, j:j+args.patch_size])
                    hr_patches.append(hr[i*args.scale:i*args.scale+args.patch_size*args.scale, j*args.scale:j*args.scale+args.patch_size*args.scale])
                    hr_edges_patches.append(weighted_edge[i*args.scale:i*args.scale+args.patch_size*args.scale, j*args.scale:j*args.scale+args.patch_size*args.scale])
                

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    hr_edges_patches = np.array(hr_edges_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.create_dataset('hr_edges', data=hr_edges_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=False)
    parser.add_argument('--output_path', type=str, required=False)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--with_aug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    
    # args.images_dir = '/scratch/lao/Datasets/DIV2K_train_HR/'
    # args.output_path = "/scratch/lao/Datasets/DIV2K_train_HR.h5"
    
    # train
    args.images_dir = 'C:/Users/Zoe/Desktop/743project/CMPT-743-FSRCNN/DIV2K_train_HR_fullsize/'
    args.output_path = "C:/Users/Zoe/Desktop/743project/CMPT-743-FSRCNN/my_training.h5"
    
    # val
    #args.images_dir = 'C:/Users/Zoe/Desktop/743project/CMPT-743-FSRCNN/DIV2K_valid_HR/'
    #args.output_path = "C:/Users/Zoe/Desktop/743project/CMPT-743-FSRCNN/my_eval.h5"
    
    args.patch_size = 40
    args.scale = 4
    args.with_aug = False
    args.eval = False

    if not args.eval:
        train(args)
    else:
        eval(args)

