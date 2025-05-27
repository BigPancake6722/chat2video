# SyncGaussianTalker

## 项目简介
这个项目是基于 [gaussiantalker](https://github.com/KU-CVLAB/GaussianTalker) 和 [synctalk](https://github.com/ZiqiaoPeng/SyncTalk/tree/main) 进行结合和改进的版本。
基于两个项目的贡献，利用synctalk的头部稳定器和AVE编码以及gaussiantalker的高斯溅射渲染，实时高效/稳定/唇音一致的高斯溅射版本模型。

此版本为2dgs实现版。surfel-rasterization更改了代码，让attention map（通道4维非3维）可以被渲染出来。使用3060TI可以达到30FPS的写入速度，仅推断可以达到60FPS。

## News
- 2024.9.14: 新增帧内容编码器，通过对语音特征以及帧编码的注意力权重做正则，可以让语音特征更聚焦于脸部，在原视频处于比较频繁多动以及语义分割效果不是特别好的情况下，全身建模可能会出现抖动/闪屏较大的情况，可以令帧编码保持不动以换得更好的画面稳定性。后续会尝试不同的帧编码方案，在训练集质量有限的情况下最大程度保证稳定性。
https://www.bilibili.com/video/BV1pRtNeUE6d/

- 2024.9.21: render边缘增加高斯模糊，在头发等细节较多无法使用全身建模时，可以仅对face进行建模（固定BFM实验，为未来通用模型作基础）。增加上半脸的表情特征（来自blendshape），同时帧编码方案改为Sinusoidal位置编码，相对于embedding更稳定（但是表达能力也弱了些）。
https://www.bilibili.com/video/BV1sQtzeUE4Z

## 主要更改内容

1. 功能增强：
   - synctalk的读取方案进TalkingDatasetReader
   - 将原本使用deepspeech和hubert特征改为synctalk的ave
   - 采用synctalk的光流法来估计旋转角度和相机位置等特征
   - 去除因使用au.csv带来的外部操作，参考synctalk使用blendshape表达眼部信息
   - 将nerf的旋转矩阵更改至和3dgs渲染一致
   - 训练时去除背景，让3dgs过程聚焦于头部
   - 增加fused.ply的生成，不将脸部信息存于tracking_params.pt中，减少硬盘占用，fused.ply采用个体的平均id_params和exp_prams构成
   - 输出模型时增加cameras.json，方便被[SIBR](https://sibr.gitlabpages.inria.fr/)预览预生成的gs模型

2. 性能优化：
   - 利用lazyload优化了内存加载溢出问题，但render时会因此造成减速
   - render时用多线程读取数据，优化了render速度
   - 3DMM人脸优化时为id系数和exp系数添加5σ取值范围，避免脸部崩溃
   - face_tracker阶段将拟合人脸作为fused.ply，减少预处理时间

3. 后续
   - [x] 剔除干扰帧，如无面部/面部区域不够等/训练和推断时过滤掉
   - [x] 添加位姿估计选项，可直接用3dmm或光流法
   - [ ] 集成包和UI页面
   - [ ] 输入音频流实时渲染
   - [ ] 躯干的训练(估计躯干位姿)
   - [x] 上半脸表情的拟合
   - [ ] mesh化
   - [ ] readme_en.md
   - [ ] 固定BFM训练通用模型

### 安装
1. 安装依赖
```bash
git clone https://gitee.com/Ky1eYang/sync-gaussian-talker --recursive
cd sync-gaussian-talker
pip install -r requirements.txt
pip install submodules/custom-diff-surfel-rasterization
pip install submodules/simple-knn
```
2. 获取数据处理模型
### [处理视频](https://github.com/ZiqiaoPeng/SyncTalk/blob/main/README.md)
- Prepare face-parsing model.

  ```bash
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O data_utils/face_parsing/79999_iter.pth
  ```

- Prepare the 3DMM model for head pose estimation.

  ```bash
  mkdir -p data_utils/face_tracking/3DMM
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/exp_info.npy?raw=true -O data_utils/face_tracking/3DMM/exp_info.npy
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/keys_info.npy?raw=true -O data_utils/face_tracking/3DMM/keys_info.npy
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/sub_mesh.obj?raw=true -O data_utils/face_tracking/3DMM/sub_mesh.obj
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/topology_info.npy?raw=true -O data_utils/face_tracking/3DMM/topology_info.npy
  ```

- Prepare the UNFaceFlow model.

  ```bash
  mkdir -p data_utils/UNFaceFlow/pretrain_model
  wget https://github.com/ZiqiaoPeng/SyncTalk/raw/main/data_utils/UNFaceFlow/sgd_NNRT_model_epoch19008_50000.pth -O data_utils/UNFaceFlow/sgd_NNRT_model_epoch19008_50000.pth
  wget https://github.com/ZiqiaoPeng/SyncTalk/raw/main/data_utils/UNFaceFlow/pretrain_model/raft-small.pth -O data_utils/UNFaceFlow/pretrain_model/raft-small.pth
  ```

- Prepare the Audio Visual Encoder.

  ```bash
  mkdir -p scene/checkpoints
  wget https://raw.githubusercontent.com/ZiqiaoPeng/SyncTalk/refs/heads/main/nerf_triplane/checkpoints/audio_visual_encoder.pth -O scene/checkpoints/audio_visual_encoder.pth
  ```

- Download 3DMM model from [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details):

  ```
  # 1. copy 01_MorphableModel.mat to data_util/face_tracking/3DMM/
  # 2.
    cd data_utils/face_tracking
    python convert_BFM.py
  ```


## 使用方法
1.  将mp4文件放入data/xxx中，并执行数据预处理
```bash
# 使用默认光流法
python data_utils/process.py data/xxx/xxx.mp4 
# 不使用光流法，使用3dmm的位姿估计
python data_utils/process.py data/xxx/xxx.mp4 --use_3dmm
# 仅执行某几步骤，方便人为干涉中间结果
python data_utils/process.py data/xxx/xxx.mp4 --task 2 --end_task 4
# 分辨率较低的情况可以放大裁剪范围
python data_utils/process.py data/xxx/xxx.mp4 --scale 1.2
```
2. （可选）脸部位姿检查

可以通过3dmm拟合先检查位姿信息是否正确，执行后会在数据目录下生成3dmm_render.mp4，查看位姿以及抖动情况，如有极端位姿请更换位姿估计方法
```bash
python data_utils/face_tracking/test_3dmm_proj.py --path data/xxx
```

3. 训练模型

建模部分使用`--part`可选`head`,`face`和`all`，face仅脸部和颈部不包括头发，head包括头发，all包括躯干，默认为head。不同的部分可能需要调整`--max_points`，默认50000高斯点，需根据实际情况调整。
```bash
# 默认情况 / 指定batch_size
python train.py -s data/xxx --model_path model/trial_xxx --configs arguments/custom.py --batch_size 16
# 渲染全身 / 添加高斯点上限
python train.py -s data/xxx --model_path model/trial_xxx --part all --max_points 70000 --configs arguments/custom.py
```

4. 渲染
```bash
# 渲染训练集，跳过测试集，输出注意力视频
python render.py -s data/xxx --model_path model/trial_xxx --batch 16 --visualize_attention --skip_test
# 自定义语音
python render.py -s data/xxx --model_path model/trial_xxx --batch 16 --custom_aud custom.wav
# 从视频1分钟作为起点开始渲染
python render.py -s data/xxx --model_path model/trial_xxx --batch 16 --ss 00:01:00 --custom_aud custom.wav
```

## 其它
感谢该项目的所有相关项目，不仅限于[Gaussiantalker](https://github.com/KU-CVLAB/GaussianTalker), [Synctalk](https://github.com/ZiqiaoPeng/SyncTalk/tree/main), [2dgs](https://github.com/hbb1/2d-gaussian-splatting), [3dgs](https://github.com/graphdeco-inria/gaussian-splatting)等，同时如有侵权请联系作者删除。

本工作流的效果很大程度基于优秀的face-parsing模型和位姿估计，有优秀的方案欢迎沟通。同时如果有感兴趣的开发者或者有使用问题的，请通过站内联系作者。