import torch.utils.data as data

from data import common


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return
        #return common.find_benchmark(self.opt['dataroot_LR'])


    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR, self.paths_LR = {}, {}
        # self.paths_HR2, self.paths_LR2 = None, None
        # self.paths_HR3, self.paths_LR3 = None, None
        # self.paths_HR4, self.paths_LR4 = None, None
        # self.paths_HR5, self.paths_LR5 = None, None
        self.num_steps = opt['num_steps']

        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 2

        # read image list from image/binary files
        for N in range(5):
            print(self.opt['dataroot_HR'][N]+"##############")
            self.paths_HR.append = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR'][N])
            self.paths_LR.append = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR'][N])
            #print(self.paths_HR)
            #print(self.paths_LR)
        print(self.paths_HR)
        # self.paths_HR2 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR2'])
        # self.paths_LR2 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR2'])
        #
        # self.paths_HR3 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR3'])
        # self.paths_LR3 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR3'])
        #
        # self.paths_HR4 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR4'])
        # self.paths_LR4 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR4'])
        #
        # self.paths_HR5 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR5'])
        # self.paths_LR5 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR5'])

        for N in range(5):
            assert self.paths_HR[N], '[Error] HR paths are empty.'
            if self.paths_LR[N] and self.paths_HR[N]:
                assert len(self.paths_LR[N]) == len(self.paths_HR[N]), \
                    '[Error] HR: [%d] and LR: [%d] have different number of images.'%(
                    len(self.paths_LR[N]), len(self.paths_HR[N]))

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

        for N in range(5):
            lr, hr, lr_path, hr_path = self._load_file(idx)
            if self.train:
                lr[N], hr[N] = self._get_patch(lr[N], hr[N])
            lr_tensor, hr_tensor = common.np2Tensor([lr[N], hr[N]], self.opt['rgb_range']) # 넘파이 파일을 텐서로 변환 / 정규화

        print(hr.size)
        print("==================================")

        return {'LR': lr_tensor, 'HR': hr_tensor, 'LR_path': lr_path, 'HR_path': hr_path}


    def __len__(self):
        if self.train:
            return len(self.paths_HR[0]) * self.repeat
        else:
            return len(self.paths_LR[0])


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR[0])
        else:
            return idx


    def _load_file(self, idx):
        for N in range(5):
            idx = self._get_index(idx)
            #lr_path = self.paths_LR[N][idx]
            lr_path = self.paths_LR[N][idx]
            hr_path = self.paths_HR[N][idx]
            lr = common.read_img(lr_path[N], self.opt['data_type'])
            hr = common.read_img(hr_path[N], self.opt['data_type'])

        return lr, hr, lr_path, hr_path


    def _get_patch(self, lr, hr):
        # 패치크기로 크롭, 어그멘테이션, 노이즈 추가
        LR_size = self.opt['LR_size']
        # random crop and augment
        lr, hr = common.get_patch(lr, hr, LR_size, self.scale)
        lr, hr = common.augment([lr, hr])
        lr = common.add_noise(lr, self.opt['noise'])

        return lr, hr
