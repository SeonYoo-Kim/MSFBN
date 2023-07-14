import os
import random
import numpy as np
import scipy.misc as misc
import imageio
from tqdm import tqdm

import torch

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
BINARY_EXTENSIONS = ['.npy']
BENCHMARK = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109', 'DIV2K', 'DF2K']


####################
# Files & IO
####################
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_binary_file(filename):
    return any(filename.endswith(extension) for extension in BINARY_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '[%s] has no valid image file' % path
    return images


def _get_paths_from_binary(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_binary_file(fname):
                binary_path = os.path.join(dirpath, fname)
                files.append(binary_path)
    assert files, '[%s] has no valid binary file' % path
    return files


def get_image_paths(data_type, dataroot):
    paths = None
    if dataroot is not None:
        if data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        elif data_type == 'npy':
            if dataroot.find('_npy') < 0 :
                old_dir = dataroot
                dataroot = dataroot + '_npy'
                if not os.path.exists(dataroot):
                    print('===> Creating binary files in [%s]' % dataroot)
                    os.makedirs(dataroot)
                    img_paths = sorted(_get_paths_from_images(old_dir))
                    path_bar = tqdm(img_paths)
                    for v in path_bar:
                        img = imageio.imread(v, pilmode='RGB')
                        ext = os.path.splitext(os.path.basename(v))[-1]
                        name_sep = os.path.basename(v.replace(ext, '.npy'))
                        np.save(os.path.join(dataroot, name_sep), img)
                else:
                    print('===> Binary files already exists in [%s]. Skip binary files generation.' % dataroot)

            paths = sorted(_get_paths_from_binary(dataroot))

        else:
            raise NotImplementedError("[Error] Data_type [%s] is not recognized." % data_type)
    return paths


def find_benchmark(dataroot):
    bm_list = [dataroot.find(bm)>=0 for bm in BENCHMARK]
    if not sum(bm_list) == 0:
        bm_idx = bm_list.index(True)
        bm_name = BENCHMARK[bm_idx]
    else:
        bm_name = 'MyImage'
    return bm_name


def read_img(path, data_type):
    # read image by misc or from .npy
    # return: Numpy float32, HWC, RGB, [0,255]
    if data_type == 'img':
        img = imageio.imread(path, pilmode='RGB')
    elif data_type.find('npy') >= 0:
        img = np.load(path)
    else:
        raise NotImplementedError

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


####################
# image processing
# process on numpy image
####################
def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        # if img.shape[2] == 3: # for opencv imread
        #     img = img[:, :, [2, 1, 0]]
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1))) #채널축을 제일 앞으로 바꿈
        tensor = torch.from_numpy(np_transpose).float() #넘파이를 파이토치로 바꿈
        tensor.mul_(rgb_range / 255.) # 0~1 정규화

        return tensor

    return [_np2Tensor(_l) for _l in l]


def get_patch(img_in, img_tar, patch_size, scale): #lr, hr, LR_size, self.scale 인풋
    ih, iw = img_in.shape[:2] #인풋의 가로세로 (shape은 가로, 세로, 채널)
    oh, ow = img_tar.shape[:2] #타겟의 가로세로

    ip = patch_size

    if ih == oh: #인-아웃간 높이가 같으면
        tp = ip #패치사이즈
        ix = random.randrange(0, iw - ip + 1) # 0 ~ (인풋가로 - 패치크기 + 1) 사이의 랜덤값
        iy = random.randrange(0, ih - ip + 1) # 0 ~ (인풋세로 - 패치크기 + 1) 사이의 랜덤값
        tx, ty = ix, iy

    else:
        tp = ip * scale #패치사이즈 * 스케일
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :] # 가로세로크기 : ip
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :] # 가로세로크기 : tp

    return img_in, img_tar # 각각 패치크기가 ip, tp인 인풋 아웃풋 이미지 리턴


def add_noise(x, noise='.'): # x = lr이미지, 디폴트는 .
    if noise is not '.': #노이즈 없음
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G': #가우시안 노이즈 추가
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S': #포이즌 노이즈? 추가
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x


def augment(img_list, hflip=True, rot=True): #이미지 어그멘테이션
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def modcrop(img_in, scale): #2채널 이미지 / 3채널 이미지를 
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [%d].' % img.ndim)
    return img
