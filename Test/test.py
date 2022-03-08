"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import argparse
import torch
import torchvision.transforms as transforms
import os, sys
from PIL import Image
import glob
import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '..'))
cwd = os.getcwd()
print(cwd)
import numpy as np
from Utils.utils import str2bool, AverageMeter, depth_read 
import Models
import Datasets
from PIL import ImageOps
import matplotlib.pyplot as plt
import time

from collections import OrderedDict
import cv2
import torch.nn.functional as F


#Training setttings
parser = argparse.ArgumentParser(description='KITTI Depth Completion Task TEST')
parser.add_argument('--dataset', type=str, default='kitti', choices = Datasets.allowed_datasets(), help='dataset to work with')
parser.add_argument('--mod', type=str, default='mod', choices = Models.allowed_models(), help='Model for use')
parser.add_argument('--no_cuda', action='store_true', help='no gpu usage')
parser.add_argument('--input_type', type=str, default='rgb', help='use rgb for rgbdepth')
# Data augmentation settings
parser.add_argument('--crop_w', type=int, default=1216, help='width of image after cropping')
parser.add_argument('--crop_h', type=int, default=256, help='height of image after cropping')

# Paths settings
parser.add_argument('--save_path', type= str, default='../Saved/best', help='save path')
parser.add_argument('--data_path', type=str, required=True, help='path to desired datasets')

# Cudnn
parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True, default=True, help="cudnn optimization active")
parser.add_argument('--multi', type=str2bool, nargs='?', const=True, default=False, help="use multiple gpus")
parser.add_argument('--normal', type=str2bool, nargs='?', const=True, default=False, help="Normalize input")
parser.add_argument('--max_depth', type=float, default=85.0, help="maximum depth of input")
parser.add_argument('--sparse_val', type=float, default=0.0, help="encode sparse values with 0")
parser.add_argument('--num_samples', default=0, type=int, help='number of samples')


Depth_Ft = np.zeros((3, 1000))
Ft_18 = []
Ft_32 = []
Ft_56 = []

Depth = 0
Depth_Ft_channel = np.zeros((128,))


def plot_Xsamples(y):
    x = np.linspace(0, 999, 1000)
    for i in range(y.shape[0]):
        plt.plot(x, y[i], label=str(i))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('plot')
    plt.legend()
    plt.show()


def plot_Xchannels(y):
    x = np.linspace(0, 127, 128).astype(np.uint8)
    # plt.plot(x, y, label=str('channel'))
    # plt.figure(figsize=(12.8, 4.8))
    plt.bar(x, y, label=str('channel'))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('plot')
    plt.legend()
    # plt.savefig('Xchannels.pdf')
    plt.show()


def calculate_depth_ft(depth_, ft):
    depth = depth_.clone().cpu()
    depth = depth.squeeze()
    ft = ft.squeeze().unsqueeze(0).unsqueeze(0).cpu()
    h, w = depth.shape
    ft = F.interpolate(ft, size=(h, w), mode='nearest')
    ft = ft.squeeze()
    depth_mask = (depth > 0).float()
    ft_mask = (ft > 0).float()
    ft = ft * depth_mask
    if ft.max() > 0:
        ft = (ft - ft.min()) / ft.max()
        depth_ft = torch.sum(ft * depth) / torch.sum(ft * depth_mask)
        return depth_ft.cpu().numpy()
    else:
        return 0.


def show_cam_on_image(mask, file):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cv2.imwrite(file, np.uint8(255 * heatmap))


counter = 0
save_path = './features'


def get_features_hook(self, input, output):
    global Depth
    global counter
    path = os.path.join(save_path, str(counter))
    if not os.path.isdir(path):
        os.makedirs(os.path.join(save_path, str(counter)))
    for i in range(output.size()[1]):
        tensor = output[0, i, :, :]

        if i == 18:
            Ft_18.append(tensor)
        if i == 32:
            Ft_32.append(tensor)
        if i == 56:
            Ft_56.append(tensor)

        Depth_Ft_channel[i] += calculate_depth_ft(Depth, tensor)

        feature = tensor.detach().cpu().numpy()
        w, h = feature.shape[0], feature.shape[1]
        # feature = 1.0 / (1 + np.exp(-1 * feature))
        if feature.max() > 0:
            feature = (feature - feature.min()) / feature.max()
        # feature = np.round(feature * 255)
        # img = Image.fromarray(np.uint8(feature))
        # tw, th = w * 2, h * 2
        # img = img.resize((th, tw), Image.BILINEAR)
        file_name = os.path.join(path, str(i) + '.png')
        # img.save(file_name)
        show_cam_on_image(feature, file_name)
    counter += 1


counter_1 = 0
save_path_1 = './attention_map'


def get_features_hook_attention_map(self, input, output):
    global counter_1
    path = save_path_1  # os.path.join(save_path_1,str(counter_1))
    if not os.path.isdir(path):
        os.makedirs(path)
    # print("input size: ",input[0].size())
    # print("output size: ",output.size())
    output = output[1]
    print(output.size())
    tensor = output[0, :, :]
    # print(tensor.size())
    feature = tensor.detach().cpu().numpy()
    # print(feature)
    for i in range(feature.shape[-1]):
        feature[:, i] = feature[:, i] / feature[:, i].max()
    # feature = feature / feature.max()
    feature = np.round(feature * 255)
    img = Image.fromarray(np.uint8(feature))
    file_name = os.path.join(path, str(counter_1) + '.png')
    img.save(file_name)
    # global counter_1
    counter_1 += 1


counter_2 = 0
save_path_2 = './channel_weight'


def get_features_hook_weight(self, input, output):
    global counter_2
    path = save_path_2  # os.path.join(save_path_1,str(counter_1))
    if not os.path.isdir(path):
        os.makedirs(path)
    # print("input size: ",input[0].size())
    # print("output size: ",output.size())
    output = output[1]
    print(output.size())
    tensor = output[0, :]
    feature = tensor.detach().cpu().numpy()
    file_name = os.path.join(path, str(counter_2) + '.npy')
    np.save(file_name, feature)
    counter_2 += 1


def main():
    global args
    global dataset
    global Depth
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = args.cudnn

    best_file_name = glob.glob(os.path.join(args.save_path, 'model_best*'))[0]

    save_root = os.path.join(os.path.dirname(best_file_name), 'results')
    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    print("==========\nArgs:{}\n==========".format(args))
    # INIT
    print("Init model: '{}'".format(args.mod))
    channels_in = 1 if args.input_type == 'depth' else 4
    model = Models.define_model(mod=args.mod, in_channels=channels_in)
    print("Number of parameters in model {} is {:.3f}M".format(args.mod.upper(), sum(tensor.numel() for tensor in model.parameters())/1e6))

    # Visualize the features of encoder, the spatial attention map, and the channel attention weights.
    handle_1 = model.depthnet.encoder.layers[14].register_forward_hook(get_features_hook)
    handle_2 = model.depthnet.encoder.scenhancer.spatial.register_forward_hook(get_features_hook_attention_map)
    handle_3 = model.depthnet.encoder.scenhancer.channel.register_forward_hook(get_features_hook_weight)

    if not args.no_cuda:
        # Load on gpu before passing params to optimizer
        if not args.multi:
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    if os.path.isfile(best_file_name):
        print("=> loading checkpoint '{}'".format(best_file_name))
        checkpoint = torch.load(best_file_name)
        # for name, val in checkpoint['state_dict'].items():
        #     if 'hourglass' in name:
        #         print(name)

        loaded = checkpoint['state_dict']
        load_net_clean = OrderedDict()
        for k, v in loaded.items():
            if 'module.' in k:
                k = k.replace('module.', '')
                load_net_clean[k] = v
            if 'gamma' in k or 'lamb' in k:
                print(k, v)
            else:
                load_net_clean[k] = v
        model.load_state_dict(load_net_clean, strict=True)

        lowest_loss = checkpoint['loss']
        best_epoch = checkpoint['best epoch']
        print('Lowest RMSE for selection validation set was {:.4f} in epoch {}'.format(lowest_loss, best_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(best_file_name))
        return

    if not args.no_cuda:
        model = model.cuda()
    print("Initializing dataset {}".format(args.dataset))
    dataset = Datasets.define_dataset(args.dataset, args.data_path, args.input_type)
    dataset.prepare_dataset()
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    depth_norm = transforms.Normalize(mean=[14.97/args.max_depth], std=[11.15/args.max_depth])
    model.eval()
    print("===> Start testing")
    total_time = []

    with torch.no_grad():
        # for i, (img, rgb, gt) in tqdm.tqdm(enumerate(zip(dataset.selected_paths['lidar_in'],
        #                                    dataset.selected_paths['img'], dataset.selected_paths['gt']))):
        for i, (img, rgb, gt) in tqdm.tqdm(enumerate(zip(dataset.test_files['lidar_in'],
                                           dataset.test_files['img'], dataset.selected_paths['gt']))):

            raw_path = os.path.join(img)
            raw_pil = Image.open(raw_path)
            gt_path = os.path.join(gt)
            gt_pil = Image.open(gt)
            assert raw_pil.size == (1216, 352)

            crop = 352-args.crop_h
            raw_pil_crop = raw_pil.crop((0, crop, 1216, 352))
            gt_pil_crop = gt_pil.crop((0, crop, 1216, 352))

            raw = depth_read(raw_pil_crop, args.sparse_val)
            raw = to_tensor(raw).float()
            gt = depth_read(gt_pil_crop, args.sparse_val)
            gt = to_tensor(gt).float()
            valid_mask = (raw > 0).detach().float()

            Depth = gt

            input = torch.unsqueeze(raw, 0).cuda()
            gt = torch.unsqueeze(gt, 0).cuda()

            if args.normal:
                # Put in {0-1} range and then normalize
                input = input/args.max_depth
                # input = depth_norm(input)

            if args.input_type == 'rgb':
                rgb_path = os.path.join(rgb)
                rgb_pil = Image.open(rgb_path)
                assert rgb_pil.size == (1216, 352)
                rgb_pil_crop = rgb_pil.crop((0, crop, 1216, 352))
                rgb = to_tensor(rgb_pil_crop).float()
                rgb = torch.unsqueeze(rgb, 0).cuda()
                if not args.normal:
                    rgb = rgb*255.0

                input = torch.cat((input, rgb), 1)

            torch.cuda.synchronize()
            a = time.perf_counter()
            output, _, _, _ = model(input)
            torch.cuda.synchronize()
            b = time.perf_counter()
            total_time.append(b-a)
            if args.normal:
                output = output*args.max_depth
            output = torch.clamp(output, min=0, max=85)

            output = output * 256.
            raw = raw * 256.
            output = output[0][0:1].cpu()
            data = output[0].numpy()
    
            if crop != 0:
                padding = (0, 0, crop, 0)
                output = torch.nn.functional.pad(output, padding, "constant", 0)
                output[:, 0:crop] = output[:, crop].repeat(crop, 1)

            pil_img = to_pil(output.int())
            assert pil_img.size == (1216, 352)
            pil_img.save(os.path.join(save_root, os.path.basename(img)))

            Depth_Ft[0, i] = calculate_depth_ft(gt, Ft_18[i])
            Depth_Ft[1, i] = calculate_depth_ft(gt, Ft_32[i])
            Depth_Ft[2, i] = calculate_depth_ft(gt, Ft_56[i])

    # Visualize the attention
    # plot_Xsamples(Depth_Ft)
    # plot_Xchannels(Depth_Ft_channel / 1000.)
    np.savetxt('depth_ft_raw.txt', Depth_Ft)
    np.savetxt('depth_ft_channel_raw.txt', Depth_Ft_channel / 1000.)

    print('average_time: ', sum(total_time[100:])/(len(total_time[100:])))
    print('num imgs: ', i + 1)


if __name__ == '__main__':
    main()
