#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pdb
import sys
import re
import numpy as np
import glob

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Utils.utils import write_file, depth_read
'''
attention:
    There is mistake in 2011_09_26_drive_0009_sync/proj_depth 4 files were
    left out 177-180 .png. Hence these files were also deleted in rgb
'''


class Kitti_preprocessing(object):
    def __init__(self, dataset_path, input_type='depth', side_selection=''):
        self.train_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.val_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.selected_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.test_files = {'img': [], 'lidar_in': []}
        self.dataset_path = dataset_path
        self.side_selection = side_selection
        self.left_side_selection = 'image_02'
        self.right_side_selection = 'image_03'
        self.depth_keyword = 'proj_depth'
        self.rgb_keyword = 'Rgb'
        # self.use_rgb = input_type == 'rgb'
        self.use_rgb = True
        self.date_selection = '2011_09_26'
        
        
    #root_d = os.path.join('..', 'data', 'kitti_depth')
    #root_rgb = os.path.join('..', 'data', 'kitti_rgb')
    def get_paths_excetly(self):
        #assert (args.use_d or args.use_rgb or args.use_g), 'no proper input selected'
        root_d = os.path.join(self.dataset_path,'kitti_depth')
        root_rgb = os.path.join(self.dataset_path,'kitti_rgb')
        
        ##the train dataset
        glob_gt = "train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
        pattern_d = ("groundtruth","velodyne_raw")
        def get_rgb_paths(p):
          ps = p.split('/')
          pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
          return pnew
        if glob_gt is not None:
            glob_gt = os.path.join(root_d,glob_gt)
            paths_gt = sorted(glob.glob(glob_gt))
            paths_d = [p.replace(pattern_d[0],pattern_d[1]) for p in paths_gt]
            paths_rgb = [get_rgb_paths(p) for p in paths_gt]
        self.train_paths['lidar_in'] = paths_d
        self.train_paths['gt'] = paths_gt
        self.train_paths['img'] = paths_rgb
        
        #the val datasets
        glob_gt = "val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
        pattern_d = ("groundtruth","velodyne_raw")
        def get_rgb_paths(p):
          ps = p.split('/')
          pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
          return pnew
        if glob_gt is not None:
            glob_gt = os.path.join(root_d,glob_gt)
            paths_gt = sorted(glob.glob(glob_gt))
            paths_d = [p.replace(pattern_d[0],pattern_d[1]) for p in paths_gt]
            paths_rgb = [get_rgb_paths(p) for p in paths_gt]
        self.val_paths['lidar_in'] = paths_d
        self.val_paths['gt'] = paths_gt
        self.val_paths['img'] = paths_rgb
        
        self.train_paths['lidar_in'] += paths_d
        self.train_paths['gt'] += paths_gt
        self.train_paths['img'] += paths_rgb
        
        #selected val dataset
        glob_gt = "val_selection_cropped/groundtruth_depth/*.png"
        pattern_d = ("groundtruth_depth","velodyne_raw")
        def get_rgb_paths(p):
          return p.replace("groundtruth_depth","image")
        if glob_gt is not None:
            glob_gt = os.path.join(root_d,glob_gt)
            paths_gt = sorted(glob.glob(glob_gt))
            paths_d = [p.replace(pattern_d[0],pattern_d[1]) for p in paths_gt]
            paths_rgb = [get_rgb_paths(p) for p in paths_gt]
        self.selected_paths['lidar_in'] = paths_d
        self.selected_paths['gt'] = paths_gt
        self.selected_paths['img'] = paths_rgb
        
        #the test datasets
        glob_gt  = None #"test_depth_completion_anonymous/"
        base = "/test_depth_completion_anonymous/"
        glob_d   = root_d+base+"/velodyne_raw/*.png"
        glob_rgb = root_d+base+"/image/*.png"
        paths_rgb = sorted(glob.glob(glob_rgb))
        #paths_gt = [None]*len(paths_rgb)
        paths_d = sorted(glob.glob(glob_d))
        self.test_files['lidar_in'] = paths_d
        #self.test_files['gt'] = paths_gt
        self.test_files['img'] = paths_rgb
        
        '''
        if split == "train":
            transform = train_transform
            glob_gt = "train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
            pattern_d = ("groundtruth","velodyne_raw")
            def get_rgb_paths(p):
              ps = p.split('/')
              pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
              return pnew
        elif split == "val":
            if args.val == "full":
                transform = val_transform
                glob_gt = "val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
                pattern_d = ("groundtruth","velodyne_raw")
                def get_rgb_paths(p):
                  ps = p.split('/')
                  pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
                  return pnew
            elif args.val == "select":
                transform = no_transform
                glob_gt = "val_selection_cropped/groundtruth_depth/*.png"
                pattern_d = ("groundtruth_depth","velodyne_raw")
                def get_rgb_paths(p):
                  return p.replace("groundtruth_depth","image")
        elif split == "test_completion":
            transform = no_transform
            glob_gt  = None #"test_depth_completion_anonymous/"
            base = "/test_depth_completion_anonymous/"
            glob_d   = root_d+base+"/velodyne_raw/*.png"
            glob_rgb = root_d+base+"/image/*.png"
        elif split == "test_prediction":
            transform = no_transform
            glob_gt  = None #"test_depth_completion_anonymous/"
            base = "/test_depth_prediction_anonymous/"
            glob_d   = root_d+base+"/velodyne_raw/*.png"
            glob_rgb = root_d+base+"/image/*.png"
        else:
            raise ValueError("Unrecognized split "+str(split))
    
        if glob_gt is not None:
            glob_gt = os.path.join(root_d,glob_gt)
            paths_gt = sorted(glob.glob(glob_gt))
            paths_d = [p.replace(pattern_d[0],pattern_d[1]) for p in paths_gt]
            paths_rgb = [get_rgb_paths(p) for p in paths_gt]
        else: # test and only has d or rgb
            paths_rgb = sorted(glob.glob(glob_rgb))
            paths_gt = [None]*len(paths_rgb)
            if split == "test_prediction":
                paths_d = [None]*len(paths_rgb) # test_prediction has no sparse depth
            else:
                paths_d = sorted(glob.glob(glob_d))
    
        if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
            raise(RuntimeError("Found 0 images in data folders"))
        if len(paths_d) == 0 and args.use_d:
            raise(RuntimeError("Requested sparse depth but none was found"))
        if len(paths_rgb) == 0 and args.use_rgb:
            raise(RuntimeError("Requested rgb images but none was found"))
        if len(paths_rgb) == 0 and args.use_g:
            raise(RuntimeError("Requested gray images but no rgb was found"))
        if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
            raise(RuntimeError("Produced different sizes for datasets"))
    
        paths = {"rgb":paths_rgb, "d":paths_d, "gt":paths_gt}
        return paths, transform        
        '''
    

    def get_paths(self):
        # train and validation dirs
        for type_set in os.listdir(self.dataset_path):
            for root, dirs, files in os.walk(os.path.join(self.dataset_path, type_set)):
                if re.search(self.depth_keyword, root):
                    self.train_paths['lidar_in'].extend(sorted([os.path.join(root, file) for file in files
                                                        if re.search('velodyne_raw', root)
                                                        and re.search('train', root)
                                                        and re.search(self.side_selection, root)]))
                    self.val_paths['lidar_in'].extend(sorted([os.path.join(root, file) for file in files
                                                              if re.search('velodyne_raw', root)
                                                              and re.search('val', root)
                                                              and re.search(self.side_selection, root)]))
                    self.train_paths['gt'].extend(sorted([os.path.join(root, file) for file in files
                                                          if re.search('groundtruth', root)
                                                          and re.search('train', root)
                                                          and re.search(self.side_selection, root)]))
                    self.val_paths['gt'].extend(sorted([os.path.join(root, file) for file in files
                                                        if re.search('groundtruth', root)
                                                        and re.search('val', root)
                                                        and re.search(self.side_selection, root)]))
                if self.use_rgb:
                    if re.search(self.rgb_keyword, root) and re.search(self.side_selection, root):
                        self.train_paths['img'].extend(sorted([os.path.join(root, file) for file in files
                                                               if re.search('train', root)]))
                                                               # and (re.search('image_02', root) or re.search('image_03', root))
                                                               # and re.search('data', root)]))
                       
                        self.val_paths['img'].extend(sorted([os.path.join(root, file) for file in files
                                                            if re.search('val', root)]))
                                                            # and (re.search('image_02', root) or re.search('image_03', root))
                                                            # and re.search('data', root)]))
               # if len(self.train_paths['lidar_in']) != len(self.train_paths['img']):
                   # print(root)
                   # pdb.set_trace()


    def convert_png_to_rgb(self, rgb_images, destination):
        for i, img_set_path in tqdm.tqdm(enumerate(rgb_images)):
            name = os.path.splitext(os.path.basename(img_set_path))[0]
            im = Image.open(img_set_path)
            rgb_im = im.convert('RGB')
            folder = os.path.join(*str.split(img_set_path, os.path.sep)[8:12])
            os.makedirs(os.path.join(destination, os.path.join(folder)), exist_ok=True)
            rgb_im.save(os.path.join(destination, os.path.join(folder, name)) + '.jpg')
            # rgb_im.save(os.path.join(destination, name) + '.jpg')

    def get_selected_paths(self, selection):
        files = []
        for file in sorted(os.listdir(os.path.join(self.dataset_path, selection))):
            files.append(os.path.join(self.dataset_path, os.path.join(selection, file)))
        return files

    def prepare_dataset(self):
        path_to_val_sel = 'depth_selection/val_selection_cropped'
        path_to_test = 'depth_selection/test_depth_completion_anonymous'
        '''
        self.get_paths()
        self.selected_paths['lidar_in'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'velodyne_raw'))
        self.selected_paths['gt'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'groundtruth_depth'))
        self.selected_paths['img'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'image'))
        self.test_files['lidar_in'] = self.get_selected_paths(os.path.join(path_to_test, 'velodyne_raw'))
        '''
        self.get_paths_excetly()
        if self.use_rgb:
            #self.selected_paths['img'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'image'))
            #self.test_files['img'] = self.get_selected_paths(os.path.join(path_to_test, 'image'))
            print(len(self.train_paths['lidar_in']))
            print(len(self.train_paths['img']))
            print(len(self.train_paths['gt']))
            print(len(self.val_paths['lidar_in']))
            print(len(self.val_paths['img']))
            print(len(self.val_paths['gt']))
            print(len(self.test_files['lidar_in']))
            print(len(self.test_files['img']))

    def compute_mean_std(self):
        nums = np.array([])
        means = np.array([])
        stds = np.array([])
        max_lst = np.array([])
        for i, raw_img_path in tqdm.tqdm(enumerate(self.train_paths['lidar_in'])):
            raw_img = Image.open(raw_img_path)
            raw_np = depth_read(raw_img)
            vec = raw_np[raw_np >= 0]
            # vec = vec/84.0
            means = np.append(means, np.mean(vec))
            stds = np.append(stds, np.std(vec))
            nums = np.append(nums, len(vec))
            max_lst = np.append(max_lst, np.max(vec))
        mean = np.dot(nums, means)/np.sum(nums)
        std = np.sqrt((np.dot(nums, stds**2) + np.dot(nums, (means-mean)**2))/np.sum(nums))
        return mean, std, max_lst


if __name__ == '__main__':
    import tqdm
    from PIL import Image
    rgb_im2png = True
    calc_params = False
    datapath = '/esat/pyrite/wvangans/Datasets/KITTI/Data'
    dataset = Kitti_preprocessing(datapath, input_type='rgb')
    dataset.prepare_dataset()
    if rgb_im2png:
        destination_train = '/usr/data/tmp/kitti/Rgb/train'
        destination_valid = '/usr/data/tmp/kitti/Rgb/valid'
        dataset.convert_png_to_rgb(dataset.train_paths['img'], destination_train)
        dataset.convert_png_to_rgb(dataset.val_paths['img'], destination_valid)
    if calc_params:
        import matplotlib.pyplot as plt
        params = dataset.compute_mean_std()
        mu_std = params[0:2]
        max_lst = params[-1]
        print('Means and std equals {} and {}'.format(*mu_std))
        plt.hist(max_lst, bins='auto')
        plt.title('Histogram for max depth')
        plt.show()
        # mean, std = 14.969576188369581, 11.149000139428104
        # Normalized
        # mean, std = 0.17820924033773314, 0.1327261921360489
