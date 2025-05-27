# ref:https://github.com/ShunyuYao/DFA-NeRF
import sys
import os
from tqdm import tqdm
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'core'))
from pathlib import Path
from data_test_flow import *
from models.network_flow import NeuralNRT
from options_test_flow import TestOptions
import torch
import numpy as np
from matplotlib import pyplot as plt

def block_average_flow(flow, n):
    h, w = flow.shape[:2]
    bh, bw = h // n, w // n
    avg_flow = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            avg_flow[i, j] = np.mean(flow[i*bh:(i+1)*bh, j*bw:(j+1)*bw], axis=(0, 1))
    
    return avg_flow

def plot_flow_arrows(flow, n):
    avg_flow = block_average_flow(flow, n)
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h:h/n, 0:w:w/n] + h/(2*n)
    plt.figure(figsize=(12, 10))
    plt.imshow(np.zeros((h, w)), cmap='gray')  # 创建一个黑色背景
    # 绘制箭头
    plt.quiver(x, y, avg_flow[..., 0], -avg_flow[..., 1], 
               color='r', angles='xy', scale_units='xy', scale=1, 
               width=0.001, headwidth=3, headlength=4)
    
    plt.title(f'Optical Flow Directions ({n}x{n} grid)')
    plt.axis('off')
    # plt.gca().invert_yaxis()  # 图像坐标系y轴向下
    plt.tight_layout()


if __name__ == "__main__":
    # width = 272
    # height = 480

    test_opts = TestOptions().parse()
    test_opts.datapath = "data/fy2/flow_list_torso.txt"
    
    test_opts.pretrain_model_path = os.path.join(
        dir_path, 'pretrain_model/raft-small.pth')
    data_loader = CreateDataLoader(test_opts)
    testloader = data_loader.load_data()
    model_path = os.path.join(dir_path, 'sgd_NNRT_model_epoch19008_50000.pth')
    model = NeuralNRT(test_opts, os.path.join(
        dir_path, 'pretrain_model/raft-small.pth'))
    state_dict = torch.load(model_path)

    model.CorresPred.load_state_dict(state_dict["net_C"])
    model.ImportanceW.load_state_dict(state_dict["net_W"])

    model = model.cuda()

    save_path = test_opts.savepath
    Path(save_path).mkdir(parents=True, exist_ok=True)
    total_length = len(testloader)

    for batch_idx, data in tqdm(enumerate(testloader), total=total_length):
        with torch.no_grad():
            model.eval()
            path_flow = data["path_flow"]
            src_crop_im = data["src_crop_color"].cuda()
            tar_crop_im = data["tar_crop_color"].cuda()
            src_im = data["src_color"].cuda()
            tar_im = data["tar_color"].cuda()
            src_mask = data["src_mask"].cuda()
            crop_param = data["Crop_param"].cuda()
            B = src_mask.shape[0]
            flow = model(src_crop_im, tar_crop_im, src_im, tar_im, crop_param)
            for i in range(B):
                flow_tmp = flow[i].cpu().numpy() * src_mask[i].cpu().numpy()
        break
    plot_flow_arrows(flow_tmp.transpose(1, 2, 0), 64)
    plt.show()