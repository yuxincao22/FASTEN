import os, functools
from collections import defaultdict
from tqdm import tqdm
import random
import cv2
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.utils.data as data


def PILOpen(path):
    img = Image.open(path.encode("UTF-8"))
    if hasattr(ImageOps, 'exif_transpose'):
        img = ImageOps.exif_transpose(img)
    return img

def default_loader(path_imgs, to_bgr):
    imgs = [PILOpen(path) for path in path_imgs]
    if to_bgr:
        imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
    return imgs


class ClipDataset(data.Dataset):
    def __init__(self, path_list, anno_parser, nframes, max_skip, ignore_labels=[], transform=dict(),
                 loader=default_loader, to_bgr=False, phase='train'):

        self = anno_parser(self, path_list, nframes, ignore_labels)
        self.nframes = nframes
        self.max_skip = max_skip
        self.phase = phase

        self.loader = functools.partial(loader, to_bgr=to_bgr)

        self.alb_transform = transform.pop('alb_transform', None)
        self.input_transform = transform.pop('transform', None)
        self.norm_transform = transform.pop('norm_transform', None)
        self.target_transform = transform.pop('target_transform', None)
        self.co_transform = transform.pop('co_transform', None)

        self.fourier_transform = transform.pop('fourier_transform', None)

    def __getitem__(self, index):
        annos = self.item_list[index]
        vid= annos[0]
        f_num = annos[1] 
        target = annos[2]

        clips = self.video_dict[vid]
        if self.phase == 'train':
            clip_hit = random.sample(clips, self.nframes)
            clip_hit.sort()
        else:
            skip = (f_num - 1) // (self.nframes - 1)
            clip_hit = clips[:skip*self.nframes:skip]

        inputs = [f'{vid}/{fname}' for fname in clip_hit]
        inputs = self.loader(inputs)

        if self.co_transform is not None:
            inputs = self.co_transform(inputs)
        if self.alb_transform is not None:
            inputs = [self.alb_transform(image=np.array(inp))["image"] for inp in inputs]
        elif self.input_transform is not None:
            inputs = [self.input_transform(inp) for inp in inputs]

        org_inputs = inputs.copy()
        if self.norm_transform is not None:
            inputs = [self.norm_transform(inp) for inp in org_inputs]
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if len(inputs) == 1:
            inputs = inputs[0]
        else:
            assert len(inputs) == self.nframes, print(annos[0])
            inputs = torch.cat(inputs, dim=0)
 
        
        if self.fourier_transform is not None:
            return index, inputs, ft_img, target
        return index, inputs, vid, target

    def __len__(self):
        return len(self.item_list)


def load_clip_list(obj, annofile, nframes, ignore_labels=[]):
    with open(annofile, 'r', encoding='utf-8') as f:
        ls = f.readlines()
    
    item_list = []
    video_dict = defaultdict(list)
    vid_prev = ''; item_idx = -1

    for idx, line in enumerate(tqdm(ls)):
        splits = line.strip().split('\t')
        try:
            assert len(splits) == 2, \
            'line No. {}: wrong data format, please check --- {}'.format(idx+1, line.strip())

            filename, label = splits[0], int(splits[1])
            if ignore_labels is not None and label in ignore_labels:
                continue
            vid, fname = os.path.dirname(filename), os.path.basename(filename)
            video_dict[vid].append(fname)

            if vid != vid_prev:
                if item_idx >= 0 and item_list[item_idx][1] < nframes:
                    item_list.pop()
                    item_idx -= 1
                item_list.append([vid, 0, label])
                vid_prev = vid
                item_idx += 1
            item_list[item_idx][1] += 1
        except:
            continue

    if item_list[item_idx][1] < nframes:
        item_list.pop()

    obj.item_list = item_list
    obj.video_dict = video_dict

    return obj
