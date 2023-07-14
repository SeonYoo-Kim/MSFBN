import torch.utils.data as data

from data import common


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR1'])

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR1, self.paths_LR1 = None, None
        self.paths_HR2, self.paths_LR2 = None, None
        self.paths_HR3, self.paths_LR3 = None, None
        self.paths_HR4, self.paths_LR4 = None, None
        self.paths_HR5, self.paths_LR5 = None, None
        self.num_steps = opt['num_steps']

        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 2

        # read image list from image/binary files
        # for N in range(len(self.opt['dataroot_HR'])):
        #     print(self.opt['dataroot_HR'][N]+"##############")
        self.paths_HR1 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR1'])
        self.paths_LR1 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR1'])

        self.paths_HR2 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR2'])
        self.paths_LR2 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR2'])

        self.paths_HR3 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR3'])
        self.paths_LR3 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR3'])

        self.paths_HR4 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR4'])
        self.paths_LR4 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR4'])

        self.paths_HR5 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR5'])
        self.paths_LR5 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR5'])

        assert self.paths_HR1, '[Error] HR paths are empty.'
        if self.paths_LR1 and self.paths_HR1:
            assert len(self.paths_LR1) == len(self.paths_HR1), \
                '[Error] HR: [%d] and LR: [%d] have different number of images.'%(
                len(self.paths_LR1), len(self.paths_HR1))

        # assert self.paths_HR2, '[Error] HR paths are empty.'
        # if self.paths_LR2 and self.paths_HR2:
        #     assert len(self.paths_LR2) == len(self.paths_HR2), \
        #         '[Error] HR: [%d] and LR: [%d] have different number of images.' % (
        #             len(self.paths_LR2), len(self.paths_HR2))
        #
        # assert self.paths_HR3, '[Error] HR paths are empty.'
        # if self.paths_LR3 and self.paths_HR3:
        #     assert len(self.paths_LR3) == len(self.paths_HR3), \
        #         '[Error] HR: [%d] and LR: [%d] have different number of images.' % (
        #             len(self.paths_LR3), len(self.paths_HR3))
        #
        # assert self.paths_HR4, '[Error] HR paths are empty.'
        # if self.paths_LR4 and self.paths_HR4:
        #     assert len(self.paths_LR4) == len(self.paths_HR4), \
        #         '[Error] HR: [%d] and LR: [%d] have different number of images.' % (
        #             len(self.paths_LR4), len(self.paths_HR4))
        #
        # assert self.paths_HR5, '[Error] HR paths are empty.'
        # if self.paths_LR5 and self.paths_HR5:
        #     assert len(self.paths_LR5) == len(self.paths_HR5), \
        #         '[Error] HR: [%d] and LR: [%d] have different number of images.' % (
        #             len(self.paths_LR5), len(self.paths_HR5))


    def __getitem__(self, idx):

        lr1, hr1, lr_path1, hr_path1, lr2, hr2, lr_path2, hr_path2, lr3, hr3, lr_path3, hr_path3, lr4, hr4, lr_path4, hr_path4, lr5, hr5, lr_path5, hr_path5 = self._load_file(idx)
        # if self.train:
        #     lr1, hr1 = self._get_patch(lr1, hr1)
        #     lr2, hr2 = self._get_patch(lr2, hr2)
        #     lr3, hr3 = self._get_patch(lr3, hr3)
        #     lr4, hr4 = self._get_patch(lr4, hr4)
        #     lr5, hr5 = self._get_patch(lr5, hr5)
        lr_tensor1, hr_tensor1 = common.np2Tensor([lr1, hr1], self.opt['rgb_range']) # 넘파이 파일을 텐서로 변환 / 정규화
        lr_tensor2, hr_tensor2 = common.np2Tensor([lr2, hr2], self.opt['rgb_range'])
        lr_tensor3, hr_tensor3 = common.np2Tensor([lr3, hr3], self.opt['rgb_range'])
        lr_tensor4, hr_tensor4 = common.np2Tensor([lr4, hr4], self.opt['rgb_range'])
        lr_tensor5, hr_tensor5 = common.np2Tensor([lr5, hr5], self.opt['rgb_range'])

        #print(hr_tensor1.size)
        #print(lr_tensor3.size)
        print("=================##################=================")

        return {'LR1': lr_tensor1, 'HR1': hr_tensor1, 'HR': hr_tensor1, 'LR_path1': lr_path1, 'HR_path1': hr_path1,
                'LR2': lr_tensor2, 'HR2': hr_tensor2, 'LR_path2': lr_path2, 'HR_path2': hr_path2,
                'LR3': lr_tensor3, 'HR3': hr_tensor3, 'LR_path3': lr_path3, 'HR_path3': hr_path3,
                'LR4': lr_tensor4, 'HR4': hr_tensor4, 'LR_path4': lr_path4, 'HR_path4': hr_path4,
                'LR5': lr_tensor5, 'HR5': hr_tensor5, 'LR_path5': lr_path5, 'HR_path5': hr_path5 }


    def __len__(self):
        if self.train:
            return len(self.paths_HR1) * self.repeat
        else:
            return len(self.paths_LR1)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR1)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path1 = self.paths_LR1[idx]
        hr_path1 = self.paths_HR1[idx]
        lr1 = common.read_img(lr_path1, self.opt['data_type'])
        hr1 = common.read_img(hr_path1, self.opt['data_type'])

        lr_path2 = self.paths_LR2[idx]
        hr_path2 = self.paths_HR2[idx]
        lr2 = common.read_img(lr_path2, self.opt['data_type'])
        hr2 = common.read_img(hr_path2, self.opt['data_type'])

        lr_path3 = self.paths_LR3[idx]
        hr_path3 = self.paths_HR3[idx]
        lr3 = common.read_img(lr_path3, self.opt['data_type'])
        hr3 = common.read_img(hr_path3, self.opt['data_type'])

        lr_path4 = self.paths_LR4[idx]
        hr_path4 = self.paths_HR4[idx]
        lr4 = common.read_img(lr_path4, self.opt['data_type'])
        hr4 = common.read_img(hr_path4, self.opt['data_type'])

        lr_path5 = self.paths_LR5[idx]
        hr_path5 = self.paths_HR5[idx]
        lr5 = common.read_img(lr_path5, self.opt['data_type'])
        hr5 = common.read_img(hr_path5, self.opt['data_type'])

        return lr1, hr1, lr_path1, hr_path1, lr2, hr2, lr_path2, hr_path2, lr3, hr3, lr_path3, hr_path3, lr4, hr4, lr_path4, hr_path4, lr5, hr5, lr_path5, hr_path5


    def _get_patch(self, lr, hr):
        # 패치크기로 크롭, 어그멘테이션, 노이즈 추가
        LR_size = self.opt['LR_size']
        # random crop and augment
        lr, hr = common.get_patch(lr, hr, LR_size, self.scale)
        lr, hr = common.augment([lr, hr])
        lr = common.add_noise(lr, self.opt['noise'])

        return lr, hr
