# import os
# import numpy as np
# from PIL import Image
# import torch
# from torchvision.models.resnet import resnet50
# import torchvision.transforms as transforms
#
# class Mydataset():
#     def __init__(self, root, filelist, transform=None):
#         self.path = list()
#         path = os.path.join(filelist)
#         with open(path, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 fullpath = os.path.join(root, line.strip('\n'))
#                 self.path.append(fullpath)
#
#         if transform is not None:
#             self.transform = transform
#
#
#     def __getitem__(self, index):
#         img = Image.open(self.path[index])
#         img = img.convert('RGB')
#
#         return img if self.transform is None else self.transform(img)
#
#     def __len__(self):
#         return len(self.path)
#
# if __name__ == '__main__':
#     label = {}
#     with open('imagenet1000_clsidx_to_labels.txt', 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             index, name = line.split(':')
#             index = int(index.lstrip(' '))
#             label[index] = name[:-1]
#
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     trans = transforms.Compose([transforms.Resize(256),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor(),
#                                 normalize,
#                                 ])
#
#     pridata = Mydataset('/home/lqy/data/dogs_vs_cats/test1', '/home/lqy/data/dogs_vs_cats/test1.txt', trans)
#
#     model = resnet50(pretrained=True)
#     model.cuda()
#     model.eval()
#
#     for idx, img in enumerate(pridata):
#         output = model(torch.unsqueeze(img, 0).cuda())
#         output = output.data.cpu().numpy()
#         print(idx, ':', np.max(output[0]), np.argmax(output[0]), label[np.argmax(output[0])])

import shutil
import os
import glob
import random

if __name__ == '__main__':
    traindir_dog = '/home/lqy/data/dogs_vs_cats/train/dogs'
    traindir_cat = '/home/lqy/data/dogs_vs_cats/train/cats'

    traindir_mini_dog = '/home/lqy/data/dogs_vs_cats/train_mini/dogs'
    traindir_mini_cat = '/home/lqy/data/dogs_vs_cats/train_mini/cats'

    testdir_mini_dog = '/home/lqy/data/dogs_vs_cats/test_mini/dogs'
    testdir_mini_cat = '/home/lqy/data/dogs_vs_cats/test_mini/cats'

    if not os.path.exists(traindir_mini_dog):
        os.makedirs(traindir_mini_dog)
    if not os.path.exists(traindir_mini_cat):
        os.makedirs(traindir_mini_cat)
    if not os.path.exists(testdir_mini_dog):
        os.makedirs(testdir_mini_dog)
    if not os.path.exists(testdir_mini_cat):
        os.makedirs(testdir_mini_cat)

    dogs = glob.glob(os.path.join(traindir_dog, "*.jpg"))
    cats = glob.glob(os.path.join(traindir_cat, '*.jpg'))
    # print(dogs)
    # print(cats)
    random.shuffle(dogs)
    random.shuffle(cats)
    # make mini train
    for file in dogs[:1000]:
        dst = os.path.join(traindir_mini_dog, os.path.basename(file))
        print("move {} to {}".format(file, dst))
        shutil.move(file, dst)
    for file in cats[:1000]:
        dst = os.path.join(traindir_mini_cat, os.path.basename(file))
        print("move {} to {}".format(file, dst))
        shutil.move(file, dst)

    # make mini test
    for file in dogs[1000:1400]:
        dst = os.path.join(testdir_mini_dog, os.path.basename(file))
        print("move {} to {}".format(file, dst))
        shutil.move(file, dst)
    for file in cats[1000:1400]:
        dst = os.path.join(testdir_mini_cat, os.path.basename(file))
        print("move {} to {}".format(file, dst))
        shutil.move(file, dst)





