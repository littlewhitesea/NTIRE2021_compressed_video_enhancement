import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps

def modcrop(img, scale):
    (iw, ih) = img.size
    ih = ih - (ih % scale)
    iw = iw - (iw % scale)
    img = img.crop((0, 0, iw, ih))
    return img

# #real data test (only including LQ video)
class DataloadFromFolderTest(data.Dataset):  # load test dataset
    def __init__(self, image_dir, scale, scene_name, transform):
        super(DataloadFromFolderTest, self).__init__()
        ori_dir = os.path.join(image_dir, scene_name)
        com_alist = os.listdir(ori_dir)
        com_alist.sort()
        self.com_image_filenames = [os.path.join(ori_dir, x) for x in com_alist]
        self.L = len(com_alist)
        self.scale = scale
        self.transform = transform  # To_tensor

    def __getitem__(self, index):
        com = []
        img_list = [index - 2, index - 1, index, index + 1, index + 2]
        for i in range(5):
            temp_list = img_list[i]
            if temp_list < 0:
                temp_list = 0
            elif temp_list > self.L - 1:
                temp_list = self.L - 1
            com_temp = modcrop(Image.open(self.com_image_filenames[temp_list]).convert('RGB'), self.scale)
            com.append(com_temp)
        com = [np.asarray(temp) for temp in com]
        com = np.asarray(com)
        t, h, w, c = com.shape
        com = com.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT']
        if self.transform:
            com = self.transform(com)  # Tensor, [CT',H',W']
        com = com.view(c, t, h, w)

        return com

    def __len__(self):
        return self.L
