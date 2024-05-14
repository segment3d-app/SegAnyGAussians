print("Starting prompt_segmenting.py")

import time
import os
import ast

os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"
import torch
import pytorch3d.ops
from plyfile import PlyData, PlyElement
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from argparse import ArgumentParser, Namespace
import cv2

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel, FeatureGaussianModel
from gaussian_renderer import render, render_contrastive_feature

from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from utils.sh_utils import SH2RGB

def get_combined_args(parser : ArgumentParser, model_path, target_cfg_file = None):
    cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    
    if target_cfg_file is None:
        if args_cmdline.target == 'seg':
            target_cfg_file = "seg_cfg_args"
        elif args_cmdline.target == 'scene':
            target_cfg_file = "cfg_args"
        elif args_cmdline.target == 'feature' or args_cmdline.target == 'contrastive_feature' :
            target_cfg_file = "feature_cfg_args"

    try:
        cfgfilepath = os.path.join(model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file found: {}".format(cfgfilepath))
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    return Namespace(**merged_dict)

def load_point_colors_from_pcd(num_points, path):
    plydata = PlyData.read(path)

    features_dc = np.zeros((num_points, 3))
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    colors = SH2RGB(features_dc)

    # N, 3
    return torch.clamp(torch.from_numpy(colors).squeeze().cuda(), 0.0, 1.0) * 255.

def write_ply(save_path, points, colors = None, normals = None, text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    assert colors is None or normals is None, "Cannot have both colors and normals"
    
    if colors is None and normals is None:
        points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    elif colors is not None:
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=dtype_full)
    else:
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('normal_x', 'f4'), ('normal_y', 'f4'), ('normal_z', 'f4')]
        points = [(points[i,0], points[i,1], points[i,2], normals[i,0], normals[i,1], normals[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=dtype_full)

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)

def write_ply_with_color(save_path, points, colors, text=True):
    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=dtype_full)
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)

def postprocess_statistical_filtering(pcd, precomputed_mask = None, max_time = 5):
    
    if type(pcd) == np.ndarray:
        pcd = torch.from_numpy(pcd).cuda()
    else:
        pcd = pcd.cuda()

    num_points = pcd.shape[0]
    # (N, P1, K)

    std_nearest_k_distance = 10
    
    while std_nearest_k_distance > 0.1 and max_time > 0:
        nearest_k_distance = pytorch3d.ops.knn_points(
            pcd.unsqueeze(0),
            pcd.unsqueeze(0),
            K=int(num_points**0.5),
        ).dists
        mean_nearest_k_distance, std_nearest_k_distance = nearest_k_distance.mean(), nearest_k_distance.std()
#         print(std_nearest_k_distance, "std_nearest_k_distance")

        mask = nearest_k_distance.mean(dim = -1) < mean_nearest_k_distance + std_nearest_k_distance

        mask = mask.squeeze()

        pcd = pcd[mask,:]
        if precomputed_mask is not None:
            precomputed_mask[precomputed_mask != 0] = mask
        max_time -= 1
        
    return pcd.squeeze(), nearest_k_distance.mean(), precomputed_mask

def postprocess_grad_based_statistical_filtering(pcd, precomputed_mask, feature_gaussians, view, sam_mask, pipeline_args):
    start_time = time.time()
    
    background = torch.zeros(feature_gaussians.get_opacity.shape[0], 3, device = 'cuda')

    grad_catch_mask = torch.zeros(feature_gaussians.get_opacity.shape[0], 1, device = 'cuda')
    grad_catch_mask[precomputed_mask, :] = 1
    grad_catch_mask.requires_grad = True

    grad_catch_2dmask = render(
        view, 
        feature_gaussians, 
        pipeline_args, 
        background,
        filtered_mask=~precomputed_mask, 
        override_color=torch.zeros(feature_gaussians.get_opacity.shape[0], 3, device = 'cuda'),
        override_mask=grad_catch_mask,
        )['mask']


    target_mask = torch.tensor(sam_mask, device=grad_catch_2dmask.device)
    target_mask = torch.nn.functional.interpolate(target_mask.unsqueeze(0).unsqueeze(0).float(), size=grad_catch_2dmask.shape[-2:] , mode='bilinear').squeeze(0).repeat([3,1,1])
    target_mask[target_mask > 0.5] = 1
    target_mask[target_mask != 1] = 0

    loss = -(target_mask * grad_catch_2dmask).sum() + 10 * ((1-target_mask)* grad_catch_2dmask).sum()
    loss.backward()

    grad_score = grad_catch_mask.grad[precomputed_mask != 0].clone().squeeze()
    grad_score = -grad_score
    
    pos_grad_score = grad_score.clone()
    pos_grad_score[pos_grad_score <= 0] = 0
    pos_grad_score[pos_grad_score <= pos_grad_score.mean() + pos_grad_score.std()] = 0
    pos_grad_score[pos_grad_score != 0] = 1

    confirmed_mask = pos_grad_score.bool()

    if type(pcd) == np.ndarray:
        pcd = torch.from_numpy(pcd).cuda()
    else:
        pcd = pcd.cuda()

    confirmed_point = pcd[confirmed_mask == 1]
    confirmed_point, _, _ = postprocess_statistical_filtering(confirmed_point, max_time=5)

    test_nearest_k_distance = pytorch3d.ops.knn_points(
        confirmed_point.unsqueeze(0),
        confirmed_point.unsqueeze(0),
        K=2,
    ).dists
    mean_nearest_k_distance, std_nearest_k_distance = test_nearest_k_distance[:,:,1:].mean(), test_nearest_k_distance[:,:,1:].std()
    test_threshold = torch.max(test_nearest_k_distance)
#     print(test_threshold, "test threshold")

    while True:

        nearest_k_distance = pytorch3d.ops.knn_points(
            pcd.unsqueeze(0),
            confirmed_point.unsqueeze(0),
            K=1,
        ).dists
        mask = nearest_k_distance.mean(dim = -1) <= test_threshold
        mask = mask.squeeze()
        true_mask = mask
        if torch.abs(true_mask.count_nonzero() - confirmed_point.shape[0]) / confirmed_point.shape[0] < 0.001:
            break

        confirmed_point = pcd[true_mask,:]

    precomputed_mask[precomputed_mask == 1] = true_mask

#     print(time.time() - start_time)
    return confirmed_point.squeeze().detach().cpu().numpy(), precomputed_mask, test_threshold

def postprocess_growing(original_pcd, point_colors, seed_pcd, seed_point_colors, thresh = 0.05, grow_iter = 1):
    s_time = time.time()
    min_x, min_y, min_z = seed_pcd[:,0].min(), seed_pcd[:,1].min(), seed_pcd[:,2].min()
    max_x, max_y, max_z = seed_pcd[:,0].max(), seed_pcd[:,1].max(), seed_pcd[:,2].max()

    lx, ly, lz = max_x - min_x, max_y - min_y, max_z - min_z
    min_x, min_y, min_z = min_x - lx*0.05, min_y - ly*0.05, min_z - lz*0.05
    max_x, max_y, max_z = max_x + lx*0.05, max_y + ly*0.05, max_z + lz*0.05

    cutout_mask = (original_pcd[:,0] < max_x) * (original_pcd[:,1] < max_y) * (original_pcd[:,2] < max_z)
    cutout_mask *= (original_pcd[:,0] > min_x) * (original_pcd[:,1] > min_y) * (original_pcd[:,2] > min_z)
    
    cutout_point_cloud = original_pcd[cutout_mask > 0]

    for i in range(grow_iter):
        num_points_in_seed = seed_pcd.shape[0]
        res = pytorch3d.ops.ball_query(
            cutout_point_cloud.unsqueeze(0), 
            seed_pcd.unsqueeze(0),
            K=1,
            radius=thresh,
            return_nn=False
        ).idx

        mask = (res != -1).sum(-1) != 0

        mask = mask.squeeze()

        seed_pcd = cutout_point_cloud[mask, :]
    
    final_mask = cutout_mask.clone()
    final_mask[final_mask != 0] = mask > 0

#     print(mask.count_nonzero())
#     print(time.time() - s_time)

    return seed_pcd, final_mask, None

# # Hyper-parameters
parser = ArgumentParser(description="Hyperparameters parameters")
parser.add_argument("--image_root", default="./data/customized_data/laptop-sadin", type=str)
parser.add_argument("--image_idx", default=1, type=int)
parser.add_argument("--mask_idx", default=2, type=int)
parser.add_argument("--object", default="laptop", type=str)
parser.add_argument("--target_coord", default="[[400, 400]]")
parser.add_argument("--iterations", default=30000, type=int)
args = parser.parse_args()

FEATURE_DIM = 32
DATA_ROOT = args.image_root
SCENE_NAME= DATA_ROOT.split('/')[-1]
MODEL_PATH = f'./output/{SCENE_NAME}-output/'
MAIN_OUTPUT_PATH = f'./segmentation_res/{SCENE_NAME}-segment-output'
FEATURE_GAUSSIAN_ITERATION = args.iterations
# MODEL_PATH = './output/laptop-sadin-output/'
# MAIN_OUTPUT_PATH = './segmentation_res/laptop-sadin-segment-output'
# FEATURE_GAUSSIAN_ITERATION = 30000

SAM_PROJ_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/sam_proj.pt')
NEG_PROJ_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/neg_proj.pt')
FEATURE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')

SAM_ARCH = 'vit_h'
SAM_CKPT_PATH = './dependencies/sam_ckpt/sam_vit_h_4b8939.pth'

print(f"""FEATURE_DIM = {FEATURE_DIM}
DATA_ROOT = {DATA_ROOT}
MODEL_PATH = {MODEL_PATH}
MAIN_OUTPUT_PATH = {MAIN_OUTPUT_PATH}
FEATURE_GAUSSIAN_ITERATION = {FEATURE_GAUSSIAN_ITERATION}
SAM_CKPT_PATH = {SAM_CKPT_PATH}
""")

print("Data and model preparation...")
# # Data and Model Preparation
nonlinear = torch.nn.Sequential(
    torch.nn.Linear(256, 64, bias=True),
    torch.nn.LayerNorm(64),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(64, 64, bias=True),
    torch.nn.LayerNorm(64),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(64, FEATURE_DIM, bias=True),
)
nonlinear.load_state_dict(torch.load(SAM_PROJ_PATH))
nonlinear = nonlinear.cuda()
nonlinear.eval()

parser2 = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser2, sentinel=True)
pipeline = PipelineParams(parser2)
parser2.add_argument("--iteration", default=-1, type=int)
parser2.add_argument("--skip_train", action="store_true")
parser2.add_argument("--skip_test", action="store_true")
parser2.add_argument("--quiet", action="store_true")
parser2.add_argument("--segment", action="store_true")
parser2.add_argument('--target', default='scene', const='scene', nargs='?', choices=['scene', 'seg', 'feature', 'coarse_seg_everything', 'contrastive_feature', 'xyz'])
parser2.add_argument('--idx', default=0, type=int)
parser2.add_argument('--precomputed_mask', default=None, type=str)

args2 = get_combined_args(parser2, MODEL_PATH)

dataset = model.extract(args2)
dataset.need_features = False
dataset.need_masks = True

feature_gaussians = FeatureGaussianModel(FEATURE_DIM)

scene = Scene(dataset, None, feature_gaussians, load_iteration=-1, feature_load_iteration=FEATURE_GAUSSIAN_ITERATION, shuffle=False, mode='eval', target='contrastive_feature')

xyz = feature_gaussians.get_xyz
point_features = feature_gaussians.get_point_features.cuda()

model_type = SAM_ARCH
sam = sam_model_registry[model_type](checkpoint=SAM_CKPT_PATH).to('cuda')
predictor = SamPredictor(sam)

# # Begin Segmenting
cameras = scene.getTrainCameras()
print("There are",len(cameras),"views in the dataset.")

print("Begin image preparation ...")

ref_img_camera_id = args.image_idx
random_img = os.listdir(DATA_ROOT + '/images')[0]
img_size = Image.open(DATA_ROOT + '/images/' + random_img)


view = cameras[ref_img_camera_id]
img = view.original_image * 255
img = cv2.resize(img.permute([1,2,0]).detach().cpu().numpy().astype(np.uint8),dsize=(1024,1024),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
predictor.set_image(img)
sam_feature = predictor.features
# print(sam_feature)
# sam_feature = view.original_features

start_time = time.time()
bg_color = [0 for i in range(FEATURE_DIM)] #buat array berisi 0 sepanjang 32
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") #ubah array di atas menjadi tensor array
rendered_feature = render_contrastive_feature(view, feature_gaussians, pipeline.extract(args2), background)['render']
time1 = time.time() - start_time

# print(sam_feature.shape)
H, W = sam_feature.shape[-2:]
# print(sam_feature.shape[-2:])

print(f"Image preparation done in {time1}")
# plt.imshow(img)


print("Begin segmentation ...")
# input_point = np.array([[400, 400]]) #laptop img 2
# input_point = np.array([[250, 500]]) #handphone img 2
# input_point = np.array([[50, 400]]) #gelas kopi img 9
# input_point = np.array([[50, 800]]) #mangkok img 9
input_point = np.array(ast.literal_eval(args.target_coord))
input_label = np.ones(len(input_point))
# box = np.array([200, 850, 400, 1200])
# print(input_point)
# print(input_label)

with torch.no_grad():
    vanilla_masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
#         box = box,
        multimask_output=True,
    )
# print(len(vanilla_masks))
# plt.figure(figsize=(15, 15))
# plt.rcParams["font.size"] = 12
# plt.subplot(1,4,1)
# plt.imshow(vanilla_masks[0])
# plt.subplot(1,4,2)
# plt.imshow(vanilla_masks[1])
# plt.subplot(1,4,3)
# plt.imshow(vanilla_masks[2])
# plt.subplot(1,4,4)
# plt.imshow(img)

masks = torch.nn.functional.interpolate(torch.from_numpy(vanilla_masks).float().unsqueeze(0), (64,64), mode='bilinear').squeeze(0).cuda()
masks[masks > 0.5] = 1
masks[masks != 1] = 0

SEGMENT_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/{args.object}"
if(os.path.exists(SEGMENT_OUTPUT_PATH) == False):
    os.makedirs(SEGMENT_OUTPUT_PATH)

# choose which mask is the best 0/1/2?
mask_id = args.mask_idx
origin_ref_mask = torch.tensor(vanilla_masks[mask_id]).float().cuda()

print(f"Mask choosen is {mask_id}")

if origin_ref_mask.shape != (64,64):
    ref_mask = torch.nn.functional.interpolate(origin_ref_mask[None, None, :, :], (64,64), mode='bilinear').squeeze().cuda()
    ref_mask[ref_mask > 0.5] = 1
    ref_mask[ref_mask != 1] = 0
else:
    ref_mask = origin_ref_mask

# sam features
start_time = time.time()

low_dim_features = nonlinear(
    sam_feature.view(-1, H*W).permute([1,0])
).squeeze().permute([1,0]).reshape([-1, H, W])

# SAM query
mask_low_dim_features = ref_mask.unsqueeze(0) * torch.nn.functional.interpolate(low_dim_features.unsqueeze(0), ref_mask.shape[-2:], mode = 'bilinear').squeeze()
mask_pooling_prototype = mask_low_dim_features.sum(dim = (1,2)) / torch.count_nonzero(ref_mask)

# Feature Field query
# mask_low_dim_features = ref_mask.unsqueeze(0) * torch.nn.functional.interpolate(rendered_feature.unsqueeze(0), ref_mask.shape[-2:], mode = 'bilinear').squeeze()
# mask_pooling_prototype = mask_low_dim_features.sum(dim = (1,2)) / torch.count_nonzero(ref_mask)

time2 = time.time() - start_time
print(f"Segmentation done in {time2}")

import kmeans_pytorch
import importlib
importlib.reload(kmeans_pytorch)
from kmeans_pytorch import kmeans

# K-means or not
print("Initializig K-means ...")
start_time = time.time()

bg_color = [0 for i in range(32)]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
rendered_feature = render_contrastive_feature(view, feature_gaussians, pipeline.extract(args2), background)['render']

similarity_mask = torch.einsum('C,CHW->HW', mask_pooling_prototype.cuda(), rendered_feature)
similarity_mask = torch.nn.functional.interpolate(similarity_mask.float().unsqueeze(0).unsqueeze(0), (64,64), mode='bilinear').squeeze().cuda()
similarity_mask[similarity_mask > 0] = 1
similarity_mask[similarity_mask != 1] = 0

iob = (similarity_mask * ref_mask).sum(dim = (-1, -2)) / ref_mask.sum()

if iob > 0.9:
    fmask_prototype = mask_pooling_prototype.unsqueeze(0)
else:

    downsampled_masks = torch.nn.functional.adaptive_avg_pool2d(ref_mask.unsqueeze(0).unsqueeze(0), (8,8)).squeeze()
    downsampled_features = torch.nn.functional.adaptive_avg_pool2d(mask_low_dim_features.unsqueeze(0), (8,8)).squeeze(0)
    downsampled_features /= downsampled_masks.unsqueeze(0)

    downsampled_masks[downsampled_masks != 0]= 1
    init_prototypes = downsampled_features[:, downsampled_masks.bool()].permute([1,0])


    masked_sam_features = low_dim_features[:, ref_mask.bool()]
    masked_sam_features = masked_sam_features.permute([1,0])

    num_clusters = init_prototypes.shape[0]
#     print(num_clusters)
    if num_clusters <= 1:
        num_clusters = min(int(masked_sam_features.shape[0] ** 0.5), 32)
        init_prototypes = []

    cluster_ids_x, cluster_centers = kmeans(
        X=masked_sam_features, num_clusters=num_clusters, cluster_centers=init_prototypes, distance='cosine', device=torch.device('cuda')
    )

    similarity_mask = torch.sigmoid(torch.einsum('NC,CHW->NHW', cluster_centers.cuda(), rendered_feature))
    similarity_mask = torch.nn.functional.interpolate(similarity_mask.float().unsqueeze(1), (64,64), mode='bilinear').squeeze().cuda()
    similarity_mask[similarity_mask >= 0.5] = 1
    similarity_mask[similarity_mask != 1] = 0
    similarity_mask = similarity_mask.squeeze()

    ioa = (similarity_mask * ref_mask[None,:,:]).sum(dim = (-1, -2)) / (similarity_mask.sum(dim = (-1, -2)) + 1e-5)
    iob = (similarity_mask * ref_mask[None,:,:]).sum(dim = (-1, -2)) / ref_mask.sum()

    ioa = ioa.squeeze()
    iob = iob.squeeze()
    cluster_mask = ioa > 0.75

    # NMS
    for i in range(len(cluster_mask)):
        if not cluster_mask[i]:
            continue

        for j in range(i+1, len(cluster_mask)):
            if not cluster_mask[j]:
                continue

            if (similarity_mask[j] * similarity_mask[i]).sum() / ((similarity_mask[j] + similarity_mask[i]).sum() - (similarity_mask[j] * similarity_mask[i]).sum()) > 0.75:
                if ioa[i] > ioa[j]:
                    cluster_mask[j] = False
                else:
                    cluster_mask[i] = False
                    break

    fmask_prototype = torch.cat([mask_pooling_prototype.unsqueeze(0), cluster_centers[cluster_mask, :].cuda()], dim = 0)

time3 = time.time() - start_time
print(f"K-means done in {time3}")

mask_prototype = fmask_prototype
start_time = time.time()
if mask_prototype.shape[0] == 1 or len(mask_prototype.shape) == 1:
    point_logits = torch.einsum('NC,C->N', point_features, mask_prototype.squeeze())
    point_scores = torch.sigmoid(point_logits)
else:
    point_logits = torch.einsum('NC,LC->NL', point_features, mask_prototype)
    point_logits = point_logits.max(-1)[0]
    point_scores = torch.sigmoid(point_logits)
two_d_point_logits = torch.einsum('NC,CHW->NHW', mask_prototype.cuda(), rendered_feature).max(dim = 0)[0]
two_d_point_logits = torch.nn.functional.interpolate(two_d_point_logits.float()[None, None, ...], ref_mask.shape[-2:], mode='bilinear').squeeze().cuda()
in_mask_logits = two_d_point_logits[ref_mask.bool()]

# Adjustable Threshold
thresh = max(max(in_mask_logits.mean() + in_mask_logits.std(), torch.topk(point_logits, int(point_logits.shape[0]*0.1))[0][-1]), 0)

mask = point_logits > thresh
torch.save(mask, f'{SEGMENT_OUTPUT_PATH}/test_mask.pt')
# print(torch.count_nonzero(mask))
time4 = time.time() - start_time
print(f"test_mask.py saved in {time4}")

start_time = time.time()
selected_xyz = xyz[mask.cpu()].data
selected_score = point_scores[mask.cpu()]
write_ply(f'{SEGMENT_OUTPUT_PATH}/vanilla_seg.ply', selected_xyz)

selected_xyz, thresh, mask_ = postprocess_statistical_filtering(pcd=selected_xyz.clone(), precomputed_mask = mask.clone(), max_time=1)
filtered_points, filtered_mask, thresh = postprocess_grad_based_statistical_filtering(pcd=selected_xyz.clone(), precomputed_mask=mask_.clone(), feature_gaussians=feature_gaussians, view=view, sam_mask=ref_mask.clone(), pipeline_args=pipeline.extract(args2))
# filtered_points, thresh = postprocess_statistical_filtering(pcd=selected_xyz.clone(), max_time=3)

# print(thresh)
write_ply(f'{SEGMENT_OUTPUT_PATH}/filtered_seg.ply', filtered_points)
time5 = time.time() - start_time
print(f"filtered_seg.ply saved in {time5}")

start_time = time.time()
final_xyz, point_mask, final_normals = postprocess_growing(xyz, None, torch.from_numpy(filtered_points).cuda(), None, max(thresh, 0.05), grow_iter = 1)
write_ply(f'{SEGMENT_OUTPUT_PATH}/final_seg.ply', final_xyz)
time6 = time.time() - start_time
print(f"final_seg.ply saved in {time6}")

start_time = time.time()
# torch.save(torch.logical_and(feature_gaussians.get_opacity.squeeze() > 0.1, point_mask.bool()), f'{SEGMENT_OUTPUT_PATH}/pre_final_mask.pt')
torch.save(torch.logical_and(feature_gaussians.get_opacity.squeeze() > 0.1, point_mask.bool()), f'{SEGMENT_OUTPUT_PATH}/final_mask.pt')
time7 = time.time() - start_time
print(f"final_mask.pt saved in {time7}")
print("Time Cost:", time1 + time2 + time3 + time4 + time5 + time6 + time7)

# # # Filter out the points confirmed to be negative
# import gaussian_renderer
# import importlib
# importlib.reload(gaussian_renderer)

# start_time = time.time()

# final_mask = point_mask.float().detach().clone().unsqueeze(-1)
# final_mask.requires_grad = True

# background = torch.zeros(final_mask.shape[0], 3, device = 'cuda')
# rendered_mask_pkg = gaussian_renderer.render_mask(cameras[ref_img_camera_id], feature_gaussians, pipeline.extract(args2), background, precomputed_mask=final_mask)

# # print(rendered_mask_pkg['mask'].min(), rendered_mask_pkg['mask'].max())

# tmp_target_mask = torch.tensor(origin_ref_mask, device=rendered_mask_pkg['mask'].device)
# tmp_target_mask = torch.nn.functional.interpolate(tmp_target_mask.unsqueeze(0).unsqueeze(0).float(), size=rendered_mask_pkg['mask'].shape[-2:] , mode='bilinear').squeeze(0)
# tmp_target_mask[tmp_target_mask > 0.5] = 1
# tmp_target_mask[tmp_target_mask != 1] = 0

# loss = 30*torch.pow(tmp_target_mask - rendered_mask_pkg['mask'], 2).sum()
# loss.backward()

# grad_score = final_mask.grad.clone()
# final_mask = final_mask - grad_score
# final_mask[final_mask < 0] = 0
# final_mask[final_mask != 0] = 1
# final_mask *= point_mask.unsqueeze(-1)

# time7 = time.time() - start_time
# print(time7)

# # torch.save(final_mask.bool(), f'{SEGMENT_OUTPUT_PATH}/final_mask.pt')
# # torch.save(final_mask.bool(), f'{SEGMENT_OUTPUT_PATH}/post_final_mask.pt')

# final_xyz = xyz[final_mask.cpu().bool().squeeze(), ...].data



# mask_img_camera_id = 30
# rendered_mask_pkg = gaussian_renderer.render_mask(cameras[mask_img_camera_id], feature_gaussians, pipeline.extract(args2), background, precomputed_mask=final_mask.float())
# plt.figure(figsize=(15, 15))
# plt.subplot(1,2,1)
# plt.imshow(rendered_mask_pkg['mask'].squeeze().detach().cpu() >= 0.1)
# plt.subplot(1,2,2)
# plt.imshow((cameras[mask_img_camera_id].original_image).permute([1,2,0]).cpu())

# print("Time Cost:", time1 + time2 + time3 + time4 + time5 + time6 + time7)



# dumps
# input_point = np.array([[820, 580], [400, 500]])
# input_point = np.array([[300, 400], [600, 700]])
# input_point = np.array([[800, 600]])
# trex part
# input_point = np.array([[650, 670], [650, 800]])
# orchids part
# input_point = np.array([[520, 550]])
# kitchen part
# input_point = np.array([[600, 620]])
# chesstable
# input_point = np.array([[400, 600], [400, 800], [280, 650]])
# garden
# input_point = np.array([[520, 400], [550, 300]])



# # input_point = np.array([[400, 400]]) #laptop
# input_point = np.array([[220, 500]]) #handphone

# input_label = np.ones(len(input_point))
# box = np.array([200, 850, 400, 1200])
# print(input_point)
# print(input_label)

# with torch.no_grad():
#     vanilla_masks, scores, logits = predictor.predict(
#         point_coords=input_point,
#         point_labels=input_label,
# #         box = box,
#         multimask_output=True,
#     )
# print(len(vanilla_masks))
# plt.subplot(1,3,1)
# plt.imshow(vanilla_masks[0])
# plt.subplot(1,3,2)
# plt.imshow(vanilla_masks[1])
# plt.subplot(1,3,3)
# plt.imshow(vanilla_masks[2])

# masks = torch.nn.functional.interpolate(torch.from_numpy(vanilla_masks).float().unsqueeze(0), (64,64), mode='bilinear').squeeze(0).cuda()
# masks[masks > 0.5] = 1
# masks[masks != 1] = 0
