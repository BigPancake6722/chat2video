import os
import glob
import tqdm
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import face_alignment
import subprocess
from face_tracking.util import euler2rot
from scipy.ndimage import label

dirpath = os.path.dirname(os.path.realpath(__file__))
mainpath = os.path.dirname(dirpath)
import sys
sys.path.append(mainpath)

def get_valid_frame_ids(base_dir):
    ori_imgs_dir = os.path.join(base_dir, 'ori_imgs')
    mask_dir = os.path.join(base_dir, 'face_mask')
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    frame_num = max([int(os.path.basename(x).split('.')[0]) for x in image_paths]) + 1
    valid_img_ids = []
    for i in range(frame_num):
        lms_path = os.path.join(ori_imgs_dir, '{:d}.lms'.format(i))
        mask_path = os.path.join(mask_dir, '{:d}.png'.format(i))
        if os.path.isfile(lms_path) and os.path.isfile(mask_path):
            valid_img_ids.append(i)
    return valid_img_ids

def get_face_bounding_boxes(video_path, step=20):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    cap = cv2.VideoCapture(video_path)

    all_boxes = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm.tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        if _%step!=0:
            continue
        preds = fa.get_landmarks(frame)
        if preds is not None:
            bbox = cv2.boundingRect(np.array(preds[0]))
            all_boxes.append(bbox)

    cap.release()
    return all_boxes

def get_anchor_box(boxes, abra=0.3):
    # 提取四个顶点的坐标
    x_mins = [box[0] for box in boxes]
    y_mins = [box[1] for box in boxes]
    x_maxs = [box[0] + box[2] for box in boxes]
    y_maxs = [box[1] + box[3] for box in boxes]

    # 按照坐标排序
    x_mins.sort()
    y_mins.sort()
    x_maxs.sort()
    y_maxs.sort()

    # 计算去除10%最小值和10%最大值后的索引范围
    n = len(boxes)
    lower_bound = int(n * abra)
    upper_bound = int(n * (1 - abra))

    # 去除10%最小值和10%最大值
    x_min = min(x_mins[lower_bound:upper_bound])
    y_min = min(y_mins[lower_bound:upper_bound])
    x_max = max(x_maxs[lower_bound:upper_bound])
    y_max = max(y_maxs[lower_bound:upper_bound])

    # 返回锚定框
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def crop_video_to_anchor_box(video_path, output_path, anchor_box, scale_ratio=1.0):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    x, y, w, h = anchor_box
    cx, cy = int(x + w/2), int(y + h/2)
    hw, hh = max(w,h)/2, max(w,h)/2
    # hw, hh = 256, 256
    r = (1 + scale_ratio)
    ymin, ymax, xmin, xmax = max(0, int(cy - hh*r)), min(height, int(cy + hh*r)), max(0, int(cx - hw*r)), min(width, int(cx + hw*r))
    # 将区域写入文件
    with open(os.path.dirname(video_path)+"/anchor_box.txt", 'w') as f:
        f.write(f'{xmin}\n{ymin}\n{xmax-xmin}\n{ymax-ymin}')# 使用ffmpeg进行裁剪
        
    cmd = [
      'ffmpeg',
      '-i', video_path,
      '-vf', f'crop={xmax-xmin}:{ymax-ymin}:{xmin}:{ymin},scale=512:512',
      '-r', '25',
      output_path,
    ]
    if os.system(' '.join(cmd))!=0:
        raise Exception("Failed to crop video")

def extract_anchor_video(video_path, output_path, scale_ratio):
    temp_path = 'out_noaudio.mp4'
    print("[INFO] Extracting face bounding boxes from the video...")
    all_boxes = get_face_bounding_boxes(video_path)
    print("[INFO] Calculating the anchor box that covers all detected faces...")
    anchor_box = get_anchor_box(all_boxes)
    print(f"[INFO] Anchor box: {anchor_box}")
    print("[INFO] Cropping video to the anchor box...")
    crop_video_to_anchor_box(video_path, output_path, anchor_box, scale_ratio)
    print("[INFO] Cropped video saved successfully.")

def extract_audio(path, out_path, sample_rate=16000):
    
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')

def extract_audio_features(aud_path, mode='ave', checkpoint_path='scene/checkpoints/audio_visual_encoder.pth'):

    print(f'[INFO] ===== extract audio labels for {aud_path} =====')
    if mode == 'ave':
        from scene.utils import AudDataset
        from scene.networks import AudioEncoder
        from torch.utils.data import DataLoader
        # ave aud features
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AudioEncoder().to(device).eval()
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
        dataset = AudDataset(aud_path)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        outputs = []
        for mel in data_loader:
            mel = mel.to(device)
            with torch.no_grad():
                out = model(mel)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0).cpu()
        first_frame, last_frame = outputs[:1], outputs[-1:]
        aud_features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)],
                                    dim=0).unsqueeze(1).detach().cpu()
        np.save(os.path.join(aud_path.replace('.wav', '_ave.npy')), aud_features.numpy())
    elif mode == "deepspeech": # deepspeech
        cmd = f'python data_utils/deepspeech_features/extract_ds_features.py --input {aud_path}'
        os.system(cmd)
    elif mode == 'hubert':
        cmd = f'python data_utils/hubert.py --wav {aud_path}' # save to data/<name>_hu.npy
        os.system(cmd)
    print(f'[INFO] ===== extracted audio labels =====')


def extract_images(path, out_path, fps=25):

    print(f'[INFO] ===== extract images from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -vf fps={fps} -qmin 1 -q:v 1 -start_number 0 {os.path.join(out_path, "%d.jpg")}'
    os.system(cmd)
    print(f'[INFO] ===== extracted images =====')


def extract_semantics(ori_imgs_dir, parsing_dir):

    print(f'[INFO] ===== extract semantics from {ori_imgs_dir} to {parsing_dir} =====')
    cmd = f'python data_utils/face_parsing/test.py --respath={parsing_dir} --imgpath={ori_imgs_dir}'
    os.system(cmd)
    print(f'[INFO] ===== extracted semantics =====')


def extract_landmarks(ori_imgs_dir):

    print(f'[INFO] ===== extract face landmarks from {ori_imgs_dir} =====')
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    except:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    mask_dir = ori_imgs_dir.replace('ori_imgs', 'face_mask')
    for image_path in tqdm.tqdm(image_paths):
        idx = image_path.split('/')[-1].replace('.jpg', '')
        mask_path = os.path.join(mask_dir, f'{idx}.png')
        if not os.path.exists(mask_path):
            continue
        input = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(input)
        if not preds is None and len(preds) > 0:
            lands = preds[0].reshape(-1, 2)[:,:2]
            np.savetxt(image_path.replace('jpg', 'lms'), lands, '%f')
        else:
            if os.path.exists(mask_path):
                os.remove(mask_path)
    del fa
    print(f'[INFO] ===== extracted face landmarks =====')


def extract_background(base_dir, ori_imgs_dir):
    
    print(f'[INFO] ===== extract background image from {ori_imgs_dir} =====')

    from sklearn.neighbors import NearestNeighbors

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # only use 1/20 image_paths 
    image_paths = image_paths[np.random.randint(0, 20)::20]
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    # nearest neighbors
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    for image_path in tqdm.tqdm(image_paths):
        path = image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png')
        if not os.path.exists(path):
            continue
        parse_img = cv2.imread(path)
        bg_white = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        bg_black = (parse_img[..., 0] == 0) & (parse_img[..., 1] == 0) & (parse_img[..., 2] == 0)
        bg = bg_white | bg_black
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)

    distss = np.stack(distss)
    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)

    bc_pixs = max_dist > 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs]

    imgs = []
    num_pixs = distss.shape[1]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        imgs.append(img)
    imgs = np.stack(imgs).reshape(-1, num_pixs, 3)

    bc_img = np.zeros((h*w, 3), dtype=np.uint8)
    bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    bc_img = bc_img.reshape(h, w, 3)

    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 5
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
    try:
        distances, indices = nbrs.kneighbors(bg_xys)
        bg_fg_xys = fg_xys[indices[:, 0]]
        bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]
    except:
        print(f'[WARNING] cannot extract background. return background with all black.')

    cv2.imwrite(os.path.join(base_dir, 'bc.jpg'), bc_img)

    print(f'[INFO] ===== extracted background image =====')

def extract_background_improved(base_dir, ori_imgs_dir):
    print(f'[INFO] ===== extract background image from {ori_imgs_dir} =====')

    from sklearn.cluster import KMeans

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    image_paths = image_paths[np.random.randint(0, 30)::30]
    
    # Read one image to get dimensions
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    h, w, c = tmp_image.shape

    # Initialize array to store all background pixels
    all_bg_pixels = np.zeros((len(image_paths), h, w, c), dtype=np.uint8)
    
    # Store all images
    all_images = []

    for i, image_path in enumerate(tqdm.tqdm(image_paths, desc="Processing images")):
        # Read image and corresponding parsing result
        img = cv2.imread(image_path)
        all_images.append(img)
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        
        # Identify background pixels (assuming white in parsing image is background)
        bg_mask = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        
        # Store background pixels
        all_bg_pixels[i][bg_mask] = img[bg_mask]

    # Reshape for easier processing
    all_bg_pixels = all_bg_pixels.reshape(-1, h*w, c)

    # Initialize background image
    bg_img = np.zeros((h, w, c), dtype=np.uint8)

    print("[INFO] Determining most likely background pixel values...")
    for i in tqdm.tqdm(range(h*w), desc="Processing pixels"):
        pixel_values = all_bg_pixels[:, i, :]
        valid_pixels = pixel_values[np.any(pixel_values != 0, axis=1)]
        
        if len(valid_pixels) > 0:
            # Use K-means to find the most common color
            kmeans = KMeans(n_clusters=1, n_init=10)
            kmeans.fit(valid_pixels)
            most_likely_color = kmeans.cluster_centers_[0]
            
            # Find the actual pixel closest to the most likely color
            distances = np.sum((valid_pixels - most_likely_color) ** 2, axis=1)
            closest_index = np.argmin(distances)
            
            # Use the closest actual pixel color
            bg_img.reshape(-1, c)[i] = valid_pixels[closest_index].astype(np.uint8)
        else:
            # If no valid pixels, leave as black (0, 0, 0)
            pass

    # Save the background image
    cv2.imwrite(os.path.join(base_dir, 'bc.jpg'), bg_img)

    print(f'[INFO] ===== extracted background image =====')


def extract_torso_and_gt(base_dir, ori_imgs_dir):

    print(f'[INFO] ===== extract torso and gt images for {base_dir} =====')

    from scipy.ndimage import binary_erosion, binary_dilation

    # load bg
    bg_image = cv2.imread(os.path.join(base_dir, 'bc.jpg'), cv2.IMREAD_UNCHANGED)
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))

    for image_path in tqdm.tqdm(image_paths):
        # read ori image
        ori_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]

        # read semantics
        seg = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        face_mask = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '_face.png'))
        if seg is None or face_mask is None:
            continue
        face_part = (face_mask[:, :, 0]==255) & (face_mask[:, :, 1]==0) & (face_mask[:, :, 2]==0)
        # 过滤脸部掩码
        labeled_mask, num_features = label(face_part)
        area_sizes = np.bincount(labeled_mask.ravel())[1:] 
        largest_region_label = np.argmax(area_sizes) + 1
        face_part = labeled_mask == largest_region_label
        # 生成掩码
        mask_img = np.zeros_like(seg)
        mask_torso = np.zeros_like(seg)
        head_part = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
        neck_part = (seg[..., 0] == 0) & (seg[..., 1] == 255) & (seg[..., 2] == 0)
        torso_part = (seg[..., 0] == 0) & (seg[..., 1] == 0) & (seg[..., 2] == 255)
        bg_part = ~head_part & ~neck_part & ~torso_part
        # mask_img[head_part, :] = 255
        mask_img[face_part, :] = 255
        mask_torso[(neck_part)|(torso_part), :] = 255
        cv2.imwrite(image_path.replace('ori_imgs', 'face_mask').replace('.jpg', '.png'), mask_img)
        cv2.imwrite(image_path.replace('ori_imgs', 'torso_mask').replace('.jpg', '.png'), mask_torso)
        # get gt image
        gt_image = ori_image.copy()
        gt_image[bg_part] = 0 # 令背景为黑色
        # gt_image[bg_part] = bg_image[bg_part]
        cv2.imwrite(image_path.replace('ori_imgs', 'gt_imgs'), gt_image)

        # get torso image
        torso_image = gt_image.copy() # rgb
        torso_image[head_part] = bg_image[head_part]
        torso_alpha = 255 * np.ones((gt_image.shape[0], gt_image.shape[1], 1), dtype=np.uint8) # alpha
        
        # torso part "vertical" in-painting...
        L = 8 + 1
        torso_coords = np.stack(np.nonzero(torso_part), axis=-1) # [M, 2]
        # lexsort: sort 2D coords first by y then by x, 
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
        torso_coords = torso_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(torso_coords[:, 1], return_index=True, return_counts=True)
        top_torso_coords = torso_coords[uid] # [m, 2]
        # only keep top-is-head pixels
        top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_torso_coords_up.T)] 
        if mask.any():
            top_torso_coords = top_torso_coords[mask]
            # get the color
            top_torso_colors = gt_image[tuple(top_torso_coords.T)] # [m, 3]
            # construct inpaint coords (vertically up, or minus in x)
            inpaint_torso_coords = top_torso_coords[None].repeat(L, 0) # [L, m, 2]
            inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
            inpaint_torso_coords += inpaint_offsets
            inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2) # [Lm, 2]
            inpaint_torso_colors = top_torso_colors[None].repeat(L, 0) # [L, m, 3]
            darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
            inpaint_torso_colors = (inpaint_torso_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
            # set color
            torso_image[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors

            inpaint_torso_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
            inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
        else:
            inpaint_torso_mask = None

        push_down = 4
        L = 48 + push_down + 1

        neck_part = binary_dilation(neck_part, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool), iterations=3)

        neck_coords = np.stack(np.nonzero(neck_part), axis=-1) # [M, 2]
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords = neck_coords[inds]
        u, uid, ucnt = np.unique(neck_coords[:, 1], return_index=True, return_counts=True)
        top_neck_coords = neck_coords[uid] # [m, 2]
        top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_neck_coords_up.T)] 
        
        top_neck_coords = top_neck_coords[mask]
        offset_down = np.minimum(ucnt[mask] - 1, push_down)
        top_neck_coords += np.stack([offset_down, np.zeros_like(offset_down)], axis=-1)
        # get the color
        top_neck_colors = gt_image[tuple(top_neck_coords.T)] # [m, 3]

        # construct inpaint coords (vertically up, or minus in x)
        inpaint_neck_coords = top_neck_coords[None].repeat(L, 0) # [L, m, 2]
        inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
        inpaint_neck_coords += inpaint_offsets
        inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2) # [Lm, 2]

        #add
        neck_avg_color = np.mean(gt_image[neck_part], axis=0)
        inpaint_neck_colors = top_neck_colors[None].repeat(L, 0)  # [L, m, 3]
        alpha_values = np.linspace(1, 0, L).reshape(L, 1, 1)  # [L, 1, 1]
        inpaint_neck_colors = inpaint_neck_colors * alpha_values + neck_avg_color * (1 - alpha_values)
        inpaint_neck_colors = inpaint_neck_colors.reshape(-1, 3)  # [Lm, 3]
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors

        inpaint_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
        inpaint_mask[tuple(inpaint_neck_coords.T)] = True

        blur_img = torso_image.copy()
        blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)

        torso_image[inpaint_mask] = blur_img[inpaint_mask]

        # set mask
        mask = (neck_part | torso_part | inpaint_mask)
        if inpaint_torso_mask is not None:
            mask = mask | inpaint_torso_mask
        torso_image[~mask] = 0
        torso_alpha[~mask] = 0

        cv2.imwrite(image_path.replace('ori_imgs', 'torso_imgs').replace('.jpg', '.png'), np.concatenate([torso_image, torso_alpha], axis=-1))
    print(f'[INFO] ===== extracted torso and gt images =====')


def face_tracking(ori_imgs_dir):

    print(f'[INFO] ===== perform face tracking =====')
    base_dir = os.path.dirname(ori_imgs_dir)
    valid_img_ids = get_valid_frame_ids(base_dir)
    end_frame_id = valid_img_ids[-1]
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    cmd = f'python data_utils/face_tracking/face_tracker.py --path={ori_imgs_dir} --img_h={h} --img_w={w} --frame_num={end_frame_id+1}'
    print("excecuting: ", cmd)
    os.system(cmd)

    print(f'[INFO] ===== finished face tracking =====')

# ref: https://github.com/ShunyuYao/DFA-NeRF
def extract_flow(base_dir,ori_imgs_dir,mask_dir, flow_dir):
    print(f'[INFO] ===== extract flow =====')
    torch.cuda.empty_cache()
    valid_img_ids = get_valid_frame_ids(base_dir)
    ref_id = valid_img_ids[2]
    tmp_image = cv2.imread(os.path.join(ori_imgs_dir, f'{ref_id}.jpg'), cv2.IMREAD_UNCHANGED)
    h,w = tmp_image.shape[:2]
    print("height:", h, "width:", w)
    valid_img_num = len(valid_img_ids)
    with open(os.path.join(base_dir, 'flow_list.txt'), 'w') as file:
        for i in valid_img_ids:
            file.write(base_dir + '/ori_imgs/' + '{:d}.jpg '.format(ref_id) +
                       base_dir + '/face_mask/' + '{:d}.png '.format(ref_id) +
                       base_dir + '/ori_imgs/' + '{:d}.jpg '.format(i) +
                       base_dir + '/face_mask/' + '{:d}.png\n'.format(i))
        file.close()
    ext_flow_cmd = 'python data_utils/UNFaceFlow/test_flow.py --datapath=' + base_dir + '/flow_list.txt ' + \
        '--savepath=' + base_dir + '/flow_result' + \
        ' --width=' + str(w) + ' --height=' + str(h)
    os.system(ext_flow_cmd)
    face_img = cv2.imread(os.path.join(ori_imgs_dir, '{:d}.jpg'.format(ref_id)))
    face_img_mask = cv2.imread(os.path.join(mask_dir, '{:d}.png'.format(ref_id)))

    rigid_mask = face_img_mask[..., 0] > 250
    rigid_num = np.sum(rigid_mask)
    flow_frame_num = 2500
    flow_frame_num = min(flow_frame_num, valid_img_num)
    rigid_flow = np.zeros((flow_frame_num, 2, rigid_num), np.float32)
    for i in range(flow_frame_num):
        flow = np.load(os.path.join(flow_dir, '{:d}_{:d}.npy'.format(ref_id, valid_img_ids[i])))
        rigid_flow[i] = flow[:, rigid_mask]
    rigid_flow = rigid_flow.transpose((2, 1, 0))
    rigid_flow = torch.as_tensor(rigid_flow).cuda()
    lap_kernel = torch.Tensor(
        (-0.5, 1.0, -0.5)).unsqueeze(0).unsqueeze(0).float().cuda()
    flow_lap = F.conv1d(
        rigid_flow.reshape(-1, 1, rigid_flow.shape[-1]), lap_kernel)
    flow_lap = flow_lap.view(rigid_flow.shape[0], 2, -1)
    flow_lap = torch.norm(flow_lap, dim=1)
    valid_frame = torch.mean(flow_lap, dim=0) < (torch.mean(flow_lap) * 3)
    flow_lap = flow_lap[:, valid_frame]
    rigid_flow_mean = torch.mean(flow_lap, dim=1)
    rigid_flow_show = (rigid_flow_mean - torch.min(rigid_flow_mean)) / \
                      (torch.max(rigid_flow_mean) - torch.min(rigid_flow_mean)) * 255
    rigid_flow_show = rigid_flow_show.byte().cpu().numpy()
    rigid_flow_img = np.zeros((h, w, 1), dtype=np.uint8)
    rigid_flow_img[...] = 255
    rigid_flow_img[rigid_mask, 0] = rigid_flow_show
    cv2.imwrite(os.path.join(base_dir, 'rigid_flow.jpg'), rigid_flow_img)
    win_size, d_size = 5, 5
    sel_xys = np.zeros((h, w), dtype=np.int32)
    xys = []
    for y in range(0, h - win_size, win_size):
        for x in range(0, w - win_size, win_size):
            min_v = int(40)
            id_x = -1
            id_y = -1
            for dy in range(0, win_size):
                for dx in range(0, win_size):
                    if rigid_flow_img[y + dy, x + dx, 0] < min_v:
                        min_v = rigid_flow_img[y + dy, x + dx, 0]
                        id_x = x + dx
                        id_y = y + dy
            if id_x >= 0:
                if (np.sum(sel_xys[id_y - d_size:id_y + d_size + 1, id_x - d_size:id_x + d_size + 1]) == 0):
                    cv2.circle(face_img, (id_x, id_y), 1, (255, 0, 0))
                    xys.append(np.array((id_x, id_y), np.int32))
                    sel_xys[id_y, id_x] = 1

    cv2.imwrite(os.path.join(base_dir, 'keypts.jpg'), face_img)
    np.savetxt(os.path.join(base_dir, 'keypoints.txt'), xys, '%d')
    key_xys = np.loadtxt(os.path.join(base_dir, 'keypoints.txt'), np.int32)
    track_dir = os.path.join(base_dir,'flow_result')
    track_xys = np.zeros((valid_img_num, key_xys.shape[0], 3), dtype=np.float32)
    for i, jdx in enumerate(valid_img_ids):
        face_mask_path = os.path.join(mask_dir, f'{jdx}.png')
        face_img_mask = cv2.imread(face_mask_path)
        face_img_mask = face_img_mask[..., 0] > 250
        path = os.path.join(track_dir, f'{ref_id}_{jdx}.npy')
        flow = np.load(path)
        for j in range(key_xys.shape[0]):
            x = key_xys[j, 0]
            y = key_xys[j, 1]
            track_xys[i, j, 0] = x + flow[0, y, x]
            track_xys[i, j, 1] = y + flow[1, y, x]
        track_xys_int = track_xys[i, :, :2]
        invalid_pts = (track_xys_int[:, 0] < 0) | (track_xys_int[:, 1] < 0) | (track_xys_int[:, 0] > w-1) | (track_xys_int[:, 1] > h-1)
        track_xys_int[..., 0] = np.clip(track_xys_int[..., 0], 0, w - 1)
        track_xys_int[..., 1] = np.clip(track_xys_int[..., 1], 0, h - 1)
        track_xys_int = track_xys_int.astype(np.int32)
        track_xys[i, :, 2] = face_img_mask[track_xys_int[:, 1], track_xys_int[:, 0]]
        track_xys[i, invalid_pts, 2] = 0
    np.save(os.path.join(base_dir, 'track_xys.npy'), track_xys)
    pose_opt_cmd = 'python data_utils/face_tracking/bundle_adjustment.py --path=' + base_dir + ' --img_h=' + str(h) + ' --img_w=' + str(w)
    os.system(pose_opt_cmd)

def extract_blendshape(base_dir):
    print(f'[INFO] ===== extract blendshape =====')
    blendshape_cmd = 'python data_utils/blendshape_capture/main.py --path=' + base_dir
    print(blendshape_cmd)
    os.system(blendshape_cmd)


def save_transforms(base_dir, ori_imgs_dir, use3dmm=False):
    print(f'[INFO] ===== save transforms =====')

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    if use3dmm:
        params_dict = torch.load(os.path.join(base_dir, 'track_params.pt'))
    else:
        params_dict = torch.load(os.path.join(base_dir, 'bundle_adjustment.pt'))
    focal_len = params_dict['focal']
    euler_angle = params_dict['euler']
    trans = params_dict['trans']
    valid_num = euler_angle.shape[0]

    train_val_split = int(valid_num * 10 / 11)
    valid_ids = get_valid_frame_ids(base_dir)
    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)

    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))

    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ['train', 'val']
    train_val_ids = [train_ids, val_ids]
    mean_z = -float(torch.mean(trans[:, 2]).item())

    for split in range(2):
        transform_dict = dict()
        transform_dict['focal_len'] = float(focal_len[0])
        transform_dict['cx'] = float(w/2.0)
        transform_dict['cy'] = float(h/2.0)
        transform_dict['frames'] = []
        ids = train_val_ids[split]
        save_id = save_ids[split]

        for i in ids:
            i = i.item()
            frame_dict = dict()
            frame_dict['img_id'] = valid_ids[i]
            frame_dict['aud_id'] = valid_ids[i]

            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i, :, 0]

            frame_dict['transform_matrix'] = pose.numpy().tolist()

            transform_dict['frames'].append(frame_dict)

        with open(os.path.join(base_dir, 'transforms_' + save_id + '.json'), 'w') as fp:
            json.dump(transform_dict, fp, indent=2, separators=(',', ': '))

    print(f'[INFO] ===== finished saving transforms =====')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    parser.add_argument('--task', type=int, default=-1, help="-1 means all")
    parser.add_argument('--task_end', type=int, default=None, help="finish all tasks between task and task_end")
    parser.add_argument('--asr', type=str, default='ave', help="ave, hubert or deepspeech")
    parser.add_argument('--scale', type=float, default=1.0, help="scale the cropped video, default is 1.0")
    parser.add_argument('--use_3dmm', action="store_true", help="use 3dmm model or not")


    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)
    
    wav_path = os.path.join(base_dir, 'aud.wav')
    ori_imgs_dir = os.path.join(base_dir, 'ori_imgs')
    parsing_dir = os.path.join(base_dir, 'parsing')
    gt_imgs_dir = os.path.join(base_dir, 'gt_imgs')
    torso_imgs_dir = os.path.join(base_dir, 'torso_imgs')
    mask_imgs_dir = os.path.join(base_dir, 'face_mask')
    flow_dir = os.path.join(base_dir, 'flow_result')
    mask_torso_dir = os.path.join(base_dir, 'torso_mask')
    anchor_video = os.path.join(base_dir, 'anchor.mp4')
    if os.path.exists(anchor_video):
        opt.path = anchor_video
    if not os.path.exists(opt.path):
        raise ValueError("No such video file: {}".format(opt.path))

    os.makedirs(ori_imgs_dir, exist_ok=True)
    os.makedirs(parsing_dir, exist_ok=True)
    os.makedirs(gt_imgs_dir, exist_ok=True)
    os.makedirs(torso_imgs_dir, exist_ok=True)
    os.makedirs(mask_imgs_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)
    os.makedirs(mask_torso_dir, exist_ok=True)

    check_id = lambda x: (opt.task == -1 or opt.task == x) if opt.task_end is None else (opt.task <= x <= opt.task_end)
    
    # extract anchor video
    if check_id(0):
        extract_anchor_video(opt.path, anchor_video, opt.scale)
        opt.path = anchor_video


    # extract audio
    if check_id(1):
        extract_audio(opt.path, wav_path)
        extract_audio_features(wav_path, mode=opt.asr)

    # extract images
    if check_id(2):
        extract_images(opt.path, ori_imgs_dir)

    # face parsing
    if check_id(3):
        extract_semantics(ori_imgs_dir, parsing_dir)

    # extract bg
    if check_id(4):
        extract_background(base_dir, ori_imgs_dir)
        # extract_background_improved(base_dir, ori_imgs_dir)

    # extract torso images and gt_images
    if check_id(5):
        extract_torso_and_gt(base_dir, ori_imgs_dir)

    # extract face landmarks
    if check_id(6):
        extract_landmarks(ori_imgs_dir)

    # extract blendshape
    if check_id(7):
        extract_blendshape(opt.path)
        
    # face tracking
    if check_id(8):
        face_tracking(ori_imgs_dir)

    # extract flow & pose optimization
    if check_id(9):
        if not opt.use_3dmm:
            extract_flow(base_dir, ori_imgs_dir, mask_imgs_dir, flow_dir)

    # save transforms.json
    if check_id(10):
        save_transforms(base_dir, ori_imgs_dir, opt.use_3dmm)

