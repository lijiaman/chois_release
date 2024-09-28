import argparse
import os
import numpy as np
import yaml
import random 
import json 
import copy 

import trimesh 

from matplotlib import pyplot as plt
from pathlib import Path

import wandb

import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import torch.nn.functional as F

import pytorch3d.transforms as transforms 

from ema_pytorch import EMA

from manip.data.cano_traj_dataset import CanoObjectTrajDataset, quat_ik_torch, quat_fk_torch
from manip.data.long_cano_traj_dataset import LongCanoObjectTrajDataset 
from manip.data.unseen_obj_long_cano_traj_dataset import UnseenCanoObjectTrajDataset 

from manip.model.transformer_object_motion_cond_diffusion import ObjectCondGaussianDiffusion 

from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file_w_object

from manip.lafan1.utils import quat_inv, quat_mul, quat_between, normalize, quat_normalize 

from evaluation_metrics import compute_metrics, determine_floor_height_and_contacts, compute_metrics_long_seq   

import clip 

import random
torch.manual_seed(1)
random.seed(1)


def export_to_ply(points, filename='output.ply'):
    # Open the file in write mode
    with open(filename, 'w') as ply_file:
        # Write the PLY header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("comment Created by YourProgram\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")
        
        # Write the points data
        for point in points:
            ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")

def compute_signed_distances(
    sdf, sdf_centroid, sdf_extents,
    query_points):
    # sdf: 1 X 256 X 256 X 256 
    # sdf_centroid: 1 X 3, center of the bounding box.  
    # sdf_extents: 1 X 3, width, height, depth of the box.  
    # query_points: T X Nv X 3 

    # query_pts_norm = (query_points - sdf_centroid[None, :, :]) * 2 / sdf_extents[None, :, :] # Convert to range [-1, 1]
    query_pts_norm = (query_points - sdf_centroid[None, :, :]) * 2 / sdf_extents.cpu().detach().numpy().max() # Convert to range [-1, 1]
     
    query_pts_norm = query_pts_norm[...,[2, 1, 0]] # Switch the order to depth, height, width
    
    num_steps, nv, _ = query_pts_norm.shape # T X Nv X 3 

    query_pts_norm = query_pts_norm[None, :, None, :, :] # 1 X T X 1 X Nv X 3 

    signed_dists = F.grid_sample(sdf[:, None, :, :, :], query_pts_norm, \
    padding_mode='border', align_corners=True)
    # F.grid_sample: N X C X D_in X H_in X W_in, N X D_out X H_out X W_out X 3, output: N X C X D_out X H_out X W_out 
    # sdf: 1 X 1 X 256 X 256 X 256, query_pts: 1 X T X 1 X Nv X 3 -> 1 X 1 X T X 1 X Nv  

    signed_dists = signed_dists[0, 0, :, 0, :] * sdf_extents.cpu().detach().numpy().max() / 2. # T X Nv 
    
    return signed_dists

def run_smplx_model(root_trans, aa_rot_rep, betas, gender, bm_dict, return_joints24=True):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3 
    # betas: BS X 16
    # gender: BS 
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device) # BS X T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2) # BS X T X 52 X 3 

    aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3) # (BS*T) X n_joints X 3 
    betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs*num_steps, -1) # (BS*T) X 16 
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist() # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3  
    smpl_betas = betas # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :] # (BS*T) X 3 
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63) # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90) # (BS*T) X 90 

    B = smpl_trans.shape[0] # (BS*T) 

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female', "neutral"]
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int64)*-1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int64)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue
        
        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(pose_body=cur_pred_pose, pose_hand=cur_pred_pose_hand, \
                betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
        
        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    if return_joints24:
        x_pred_smpl_joints_all = torch.cat(pred_joints, axis=0) # () X 52 X 3 
        lmiddle_index= 28 
        rmiddle_index = 43 
        x_pred_smpl_joints = torch.cat((x_pred_smpl_joints_all[:, :22, :], \
            x_pred_smpl_joints_all[:, lmiddle_index:lmiddle_index+1, :], \
            x_pred_smpl_joints_all[:, rmiddle_index:rmiddle_index+1, :]), dim=1) 
    else:
        x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]
        
    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 22 X 3 

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3 

    
    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3) # BS X T X 22 X 3/BS X T X 24 X 3  
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3 

    mesh_faces = pred_body.f 
    
    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces 

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=40000,
        results_folder='./results',
        use_wandb=True,   
    ):
        super().__init__()

        self.use_wandb = use_wandb           
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, \
            name=opt.exp_name, dir=opt.save_dir)

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt 

        self.window = opt.window

        self.add_language_condition = self.opt.add_language_condition 

        self.use_random_frame_bps = self.opt.use_random_frame_bps 

        self.use_object_keypoints = self.opt.use_object_keypoints 

        self.add_semantic_contact_labels = self.opt.add_semantic_contact_labels 

        self.test_unseen_objects = self.opt.test_unseen_objects 

        self.save_res_folder = self.opt.save_res_folder 

        self.use_object_split = self.opt.use_object_split
        self.data_root_folder = self.opt.data_root_folder 
        self.prep_dataloader(window_size=opt.window)

        self.bm_dict = self.ds.bm_dict 

        self.test_on_train = self.opt.test_on_train 

        self.input_first_human_pose = self.opt.input_first_human_pose 

        self.use_guidance_in_denoising = self.opt.use_guidance_in_denoising 

        self.compute_metrics = self.opt.compute_metrics 

        self.loss_w_feet = self.opt.loss_w_feet 
        self.loss_w_fk = self.opt.loss_w_fk 
        self.loss_w_obj_pts = self.opt.loss_w_obj_pts 

        if self.add_language_condition:
            clip_version = 'ViT-B/32'
            self.clip_model = self.load_and_freeze_clip(clip_version) 

        self.use_long_planned_path = self.opt.use_long_planned_path 
        self.test_object_name = self.opt.test_object_name 
        self.test_scene_name = self.opt.test_scene_name 
        if self.use_long_planned_path:
            self.whole_seq_ds = LongCanoObjectTrajDataset(train=False, data_root_folder=self.data_root_folder, \
            window=opt.window, use_object_splits=self.use_object_split, \
            input_language_condition=self.add_language_condition, \
            use_first_frame_bps=False, use_random_frame_bps=self.use_random_frame_bps, \
            test_object_name=self.test_object_name)

            self.scene_sdf, self.scene_sdf_centroid, self.scene_sdf_extents = \
            self.load_scene_sdf_data(self.test_scene_name)

        self.hand_vertex_idxs, self.left_hand_vertex_idxs, self.right_hand_vertex_idxs = self.load_hand_vertex_ids() 
        
        if self.test_unseen_objects:
            if self.use_long_planned_path:
                test_long_seq = True 
            else:
                test_long_seq = False 

            self.unseen_seq_ds = UnseenCanoObjectTrajDataset(train=False, \
                data_root_folder=self.data_root_folder, \
                window=opt.window, use_object_splits=self.use_object_split, \
                input_language_condition=self.add_language_condition, \
                use_first_frame_bps=False, \
                use_random_frame_bps=self.use_random_frame_bps, \
                test_long_seq=test_long_seq) 

    def load_hand_vertex_ids(self):
        data_folder = "data/part_vert_ids"
        left_hand_npy_path = os.path.join(data_folder, "left_hand_vids.npy")
        right_hand_npy_path = os.path.join(data_folder, "right_hand_vids.npy")

        left_hand_vids = np.load(left_hand_npy_path)
        right_hand_vids = np.load(right_hand_npy_path) 

        hand_vids = np.concatenate((left_hand_vids, right_hand_vids), axis=0)

        return hand_vids, left_hand_vids, right_hand_vids  

    def load_scene_sdf_data(self, scene_name):
        data_folder = os.path.join(self.data_root_folder, "replica_processed/replica_fixed_poisson_sdfs_res256")
        sdf_npy_path = os.path.join(data_folder, scene_name+"_sdf.npy")
        sdf_json_path = os.path.join(data_folder, scene_name+"_sdf_info.json")

        sdf = np.load(sdf_npy_path) # 256 X 256 X 256 
        sdf_json_data = json.load(open(sdf_json_path, 'r'))

        sdf_centroid = np.asarray(sdf_json_data['centroid']) # a list with 3 items -> 3 
        sdf_extents = np.asarray(sdf_json_data['extents']) # a list with 3 items -> 3 

        sdf = torch.from_numpy(sdf).float()[None].cuda()
        sdf_centroid = torch.from_numpy(sdf_centroid).float()[None].cuda()
        sdf_extents = torch.from_numpy(sdf_extents).float()[None].cuda() 

        return sdf, sdf_centroid, sdf_extents

    def load_object_sdf_data(self, object_name):
        if self.test_unseen_objects:
            data_folder = os.path.join(self.data_root_folder, "unseen_objects_data/selected_rotated_zeroed_obj_sdf_256_npy_files")
            sdf_npy_path = os.path.join(data_folder, object_name+".npy")
            sdf_json_path = os.path.join(data_folder, object_name+".json")
        else:
            data_folder = os.path.join(self.data_root_folder, "rest_object_sdf_256_npy_files") 
            sdf_npy_path = os.path.join(data_folder, object_name+".ply.npy")
            sdf_json_path = os.path.join(data_folder, object_name+".ply.json")

        sdf = np.load(sdf_npy_path) # 256 X 256 X 256 
        sdf_json_data = json.load(open(sdf_json_path, 'r'))

        sdf_centroid = np.asarray(sdf_json_data['centroid']) # a list with 3 items -> 3 
        sdf_extents = np.asarray(sdf_json_data['extents']) # a list with 3 items -> 3 

        sdf = torch.from_numpy(sdf).float()[None].cuda()
        sdf_centroid = torch.from_numpy(sdf_centroid).float()[None].cuda()
        sdf_extents = torch.from_numpy(sdf_extents).float()[None].cuda() 

        return sdf, sdf_centroid, sdf_extents

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cuda',
                                jit=False) 
        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.clip_model.parameters()).device
        max_text_len = 30  # Specific hardcoding for the current dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        
        return self.clip_model.encode_text(texts).float().detach() # BS X 512 

    def prep_dataloader(self, window_size):
        # Define dataset
        train_dataset = CanoObjectTrajDataset(train=True, data_root_folder=self.data_root_folder, \
            window=window_size, use_object_splits=self.use_object_split, \
            input_language_condition=self.add_language_condition, \
            use_random_frame_bps=self.use_random_frame_bps, \
            use_object_keypoints=self.use_object_keypoints)
        val_dataset = CanoObjectTrajDataset(train=False, data_root_folder=self.data_root_folder, \
            window=window_size, use_object_splits=self.use_object_split, \
            input_language_condition=self.add_language_condition, \
            use_random_frame_bps=self.use_random_frame_bps, \
            use_object_keypoints=self.use_object_keypoints)

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, \
            shuffle=True, pin_memory=True, num_workers=4))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
            shuffle=False, pin_memory=True, num_workers=4))

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

    def load(self, milestone, pretrained_path=None):
        if pretrained_path is None:
            data = torch.load(os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))
        else:
            data = torch.load(pretrained_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def prep_start_end_condition_mask_pos_only(self, data, actual_seq_len):
        # data: BS X T X D (3+9)
        # actual_seq_len: BS 
        tmp_mask = torch.arange(self.window).expand(data.shape[0], \
                self.window) == (actual_seq_len[:, None].repeat(1, self.window)-1)
                # BS X max_timesteps
        tmp_mask = tmp_mask.to(data.device)[:, :, None] # BS X T X 1

        # Missing regions are ones, the condition regions are zeros. 
        mask = torch.ones_like(data[:, :, :3]).to(data.device) # BS X T X 3
        mask = mask * (~tmp_mask) # Only the actual_seq_len frame is 0

        # Add rotation mask, only the first frame's rotation is given. 
        rotation_mask = torch.ones_like(data[:, :, 3:]).to(data.device)
        mask = torch.cat((mask, rotation_mask), dim=-1) 

        mask[:, 0, :] = torch.zeros(data.shape[0], data.shape[2]).to(data.device) # BS X D  

        return mask 
    
    def prep_mimic_A_star_path_condition_mask_pos_xy_only(self, data, actual_seq_len):
        # data: BS X T X D
        # actual_seq_len: BS 
        tmp_mask = torch.arange(self.window).expand(data.shape[0], \
                self.window) == (actual_seq_len[:, None].repeat(1, self.window)-1)
                # BS X max_timesteps
        tmp_mask = tmp_mask.to(data.device)[:, :, None] # BS X T X 1
        tmp_mask = (~tmp_mask)

        # Use fixed number of waypoints.
        random_steps = [30-1, 60-1, 90-1] 
        for selected_t in random_steps:
            if selected_t < self.window - 1:
                bs_selected_t = torch.from_numpy(np.asarray([selected_t])) # 1 
                bs_selected_t = bs_selected_t[None, :].repeat(data.shape[0], self.window) # BS X T 

                curr_tmp_mask = torch.arange(self.window).expand(data.shape[0], \
                    self.window) == (bs_selected_t)
                    # BS X max_timesteps
                curr_tmp_mask = curr_tmp_mask.to(data.device)[:, :, None] # BS X T X 1

                tmp_mask = (~curr_tmp_mask)*tmp_mask

        # Missing regions are ones, the condition regions are zeros. 
        mask = torch.ones_like(data[:, :, :2]).to(data.device) # BS X T X 2
        mask = mask * tmp_mask # Only the actual_seq_len frame is 0

        # Add rotation mask, only the first frame's rotation is given. 
        # Also, add z mask, only the first frane's z is given. 
        rotation_mask = torch.ones_like(data[:, :, 2:]).to(data.device)
        mask = torch.cat((mask, rotation_mask), dim=-1) 

        mask[:, 0, :] = torch.zeros(data.shape[0], data.shape[2]).to(data.device) # BS X D  

        return mask 

    def train(self):
        init_step = self.step 
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()
        
            nan_exists = False # If met nan in loss or gradient, need to skip to next data. 
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)
                
                human_data = data_dict['motion'].cuda() # BS X T X (24*3 + 22*6)
                obj_data = data_dict['obj_motion'].cuda() # BS X T X (3+9) 

                obj_bps_data = data_dict['input_obj_bps'].cuda().reshape(-1, 1, 1024*3) # BS X 1 X 1024 X 3 -> BS X 1 X (1024*3) 
                
                rest_human_offsets = data_dict['rest_human_offsets'].cuda() # BS X 24 X 3 

                ori_data_cond = obj_bps_data # BS X 1 X (1024*3) 

                # Generate padding mask 
                actual_seq_len = data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(obj_data.shape[0], \
                self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(obj_data.device)

                # Add start & end object positions and waypoints xy as input conditions 
                end_pos_cond_mask = self.prep_start_end_condition_mask_pos_only(obj_data, data_dict['seq_len'])
                
                cond_mask = self.prep_mimic_A_star_path_condition_mask_pos_xy_only(obj_data, data_dict['seq_len'])
                cond_mask = end_pos_cond_mask * cond_mask 
              
                # Add the first human pose as input condition 
                human_cond_mask = torch.ones_like(human_data).to(human_data.device)
                if self.input_first_human_pose:
                    human_cond_mask[:, 0, :] = 0 
                
                cond_mask = torch.cat((cond_mask, human_cond_mask), dim=-1) # BS X T X (3+6+24*3+22*6)

                with autocast(enabled = self.amp):    
                    contact_data = data_dict['contact_labels'].cuda() # BS X T X 4 
                   
                    data = torch.cat((obj_data, human_data, contact_data), dim=-1) 
                    cond_mask = torch.cat((cond_mask, \
                            torch.ones_like(contact_data).to(cond_mask.device)), dim=-1) 
                    
                    if self.add_language_condition:
                        text_anno_data = data_dict['text']
                        language_input = self.encode_text(text_anno_data) # BS X 512 
                        language_input = language_input.to(data.device)
                       
                        loss_diffusion, loss_obj, loss_human, loss_feet, loss_fk, loss_obj_pts = \
                        self.model(data, ori_data_cond, cond_mask, padding_mask, \
                        language_input=language_input, \
                        rest_human_offsets=rest_human_offsets, ds=self.ds, data_dict=data_dict)
                    else:
                        loss_diffusion = self.model(data, ori_data_cond, cond_mask, padding_mask, \
                        rest_human_offsets=rest_human_offsets)
                
                    if self.use_object_keypoints:
                        loss = loss_diffusion + self.loss_w_feet * loss_feet + \
                            self.loss_w_fk * loss_fk + self.loss_w_obj_pts * loss_obj_pts 
                    else:
                        loss = loss_diffusion 

                    if torch.isnan(loss).item():
                        print('WARNING: NaN loss. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    # check gradients
                    parameters = [p for p in self.model.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(obj_data.device) for p in parameters]), 2.0)
                    if torch.isnan(total_norm):
                        print('WARNING: NaN gradients. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    if self.use_wandb:
                        if self.use_object_keypoints:
                            log_dict = {
                                "Train/Loss/Total Loss": loss.item(),
                                "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                                "Train/Loss/Object Loss": loss_obj.item(),
                                "Train/Loss/Human Loss": loss_human.item(),
                                "Train/Loss/Semantic Contact Loss": loss_feet.item(),
                                "Train/Loss/FK Loss": loss_fk.item(),
                                "Train/Loss/Object Pts Loss": loss_obj_pts.item(),
                            }
                        else:
                            log_dict = {
                                "Train/Loss/Total Loss": loss.item(),
                                "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                                "Train/Loss/Object Loss": loss_obj.item(),
                                "Train/Loss/Human Loss": loss_human.item(),
                            }
                        wandb.log(log_dict)

                    if idx % 20 == 0 and i == 0:
                        print("Step: {0}".format(idx))
                        print("Loss: %.4f" % (loss.item()))
                        print("Object Loss: %.4f" % (loss_obj.item()))
                        print("Human Loss: %.4f" % (loss_human.item()))
                        if self.use_object_keypoints:
                            print("Semantic Contact Loss: %.4f" % (loss_feet.item())) 
                            print("FK Loss: %.4f" % (loss_fk.item())) 
                            print("Object Pts Loss: %.4f" % (loss_obj_pts.item())) 

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            if self.step != 0 and self.step % 10 == 0:
                self.ema.ema_model.eval()

                with torch.no_grad():
                    val_data_dict = next(self.val_dl)
                    val_human_data = val_data_dict['motion'].cuda() 
                    val_obj_data = val_data_dict['obj_motion'].cuda()

                    obj_bps_data = val_data_dict['input_obj_bps'].cuda().reshape(-1, 1, 1024*3)
                   
                    ori_data_cond = obj_bps_data 

                    rest_human_offsets = val_data_dict['rest_human_offsets'].cuda() # BS X 24 X 3 

                    # Generate padding mask 
                    actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                    tmp_mask = torch.arange(self.window+1).expand(val_obj_data.shape[0], \
                    self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                    # BS X max_timesteps
                    padding_mask = tmp_mask[:, None, :].to(val_obj_data.device)

                    end_pos_cond_mask = self.prep_start_end_condition_mask_pos_only(val_obj_data, val_data_dict['seq_len'])
                    cond_mask = self.prep_mimic_A_star_path_condition_mask_pos_xy_only(val_obj_data, val_data_dict['seq_len'])
                    cond_mask = end_pos_cond_mask * cond_mask 
                   
                    human_cond_mask = torch.ones_like(val_human_data).to(val_human_data.device)
                    if self.input_first_human_pose:
                        human_cond_mask[:, 0, :] = 0 
                    cond_mask = torch.cat((cond_mask, human_cond_mask), dim=-1) # BS X T X (3+6+24*3+22*6)

                    # Get validation loss 
                    contact_data = val_data_dict['contact_labels'].cuda() # BS X T X 4 
                    
                    data = torch.cat((val_obj_data, val_human_data, contact_data), dim=-1) 
                    cond_mask = torch.cat((cond_mask, \
                            torch.ones_like(contact_data).to(cond_mask.device)), dim=-1) 
                   
                    if self.add_language_condition:
                        text_anno_data = val_data_dict['text']
                        language_input = self.encode_text(text_anno_data) # BS X 512 
                        language_input = language_input.to(data.device)
                      
                        val_loss_diffusion, val_loss_obj, val_loss_human, val_loss_feet, val_loss_fk, val_loss_obj_pts = \
                                        self.model(data, ori_data_cond, cond_mask, padding_mask, \
                                        language_input=language_input, \
                                        rest_human_offsets=rest_human_offsets, \
                                        ds=self.val_ds, data_dict=val_data_dict)
                      
                    else:
                        val_loss_diffusion = self.model(data, ori_data_cond, cond_mask, padding_mask, \
                                        rest_human_offsets=rest_human_offsets)
                
                    val_loss = val_loss_diffusion + self.loss_w_feet * val_loss_feet + \
                        self.loss_w_fk * val_loss_fk + self.loss_w_obj_pts * val_loss_obj_pts
                  
                    if self.use_wandb:
                        val_log_dict = {
                            "Validation/Loss/Total Loss": val_loss.item(),
                            "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
                            "Validation/Loss/Object Loss": val_loss_obj.item(),
                            "Validation/Loss/Human Loss": val_loss_human.item(),
                            "Validation/Loss/Semantic Contact Loss": val_loss_feet.item(),
                            "Validation/Loss/FK Loss": val_loss_fk.item(),
                            "Validation/Loss/Object Pts Loss": val_loss_obj_pts.item(),
                        }
                     
                        wandb.log(val_log_dict)

                    milestone = self.step // self.save_and_sample_every

                    if self.step % self.save_and_sample_every == 0:
                        self.save(milestone)
                       
                        if self.add_language_condition:
                            all_res_list = self.ema.ema_model.sample(data, ori_data_cond, cond_mask, padding_mask, \
                                        language_input=language_input, \
                                        rest_human_offsets=rest_human_offsets)
                        else:
                            all_res_list = self.ema.ema_model.sample(data, ori_data_cond, cond_mask, padding_mask, \
                                        rest_human_offsets=rest_human_offsets)
                       
                        for_vis_gt_data = torch.cat((val_obj_data, val_human_data), dim=-1)
                       
                        all_res_list = all_res_list[:, :, :-4] 
                        cond_mask = cond_mask[:, :, :-4]

                        self.gen_vis_res(for_vis_gt_data, val_data_dict, self.step, cond_mask, vis_gt=True)
                        self.gen_vis_res(all_res_list, val_data_dict, self.step, cond_mask)

            self.step += 1
       
        print('training complete')

        if self.use_wandb:
            wandb.run.finish()

    def append_new_value_to_metrics_list(self, lhand_jpe, rhand_jpe, hand_jpe, mpvpe, mpjpe, rot_dist, trans_err, \
            gt_contact_percent, contact_percent, gt_foot_sliding_jnts, foot_sliding_jnts, \
            contact_precision, contact_recall, contact_acc, contact_f1_score, obj_rot_dist, obj_com_pos_err, \
            start_obj_com_pos_err, end_obj_com_pos_err, waypoints_xy_pos_err, gt_penetration_score, penetration_score, \
            gt_hand_penetration_score, hand_penetration_score, gt_floor_height, pred_floor_height): 
        # Append new sequence's value to list. 
        self.lhand_jpe_list.append(lhand_jpe)
        self.rhand_jpe_list.append(rhand_jpe)
        self.hand_jpe_list.append(hand_jpe)
        self.mpvpe_list.append(mpvpe)
        self.mpjpe_list.append(mpjpe)
        self.rot_dist_list.append(rot_dist)
        self.trans_err_list.append(trans_err)
        
        self.gt_floor_height_list.append(gt_floor_height)
        self.floor_height_list.append(pred_floor_height)

        self.gt_foot_sliding_jnts_list.append(gt_foot_sliding_jnts)
        self.foot_sliding_jnts_list.append(foot_sliding_jnts)

        self.gt_contact_percent_list.append(gt_contact_percent)
        self.contact_percent_list.append(contact_percent)

        self.contact_precision_list.append(contact_precision)
        self.contact_recall_list.append(contact_recall)
        self.contact_acc_list.append(contact_acc)
        self.contact_f1_score_list.append(contact_f1_score)

        self.obj_rot_dist_list.append(obj_rot_dist)
        self.obj_com_pos_err_list.append(obj_com_pos_err)
        
        self.start_obj_com_pos_err_list.append(start_obj_com_pos_err)
        self.end_obj_com_pos_err_list.append(end_obj_com_pos_err)
        self.waypoints_xy_pos_err_list.append(waypoints_xy_pos_err)

        self.gt_penetration_list.append(gt_penetration_score)
        self.penetration_list.append(penetration_score) 

        self.gt_hand_penetration_list.append(gt_hand_penetration_score)
        self.hand_penetration_list.append(hand_penetration_score) 

    def print_evaluation_metrics(self, lhand_jpe_list, rhand_jpe_list, hand_jpe_list, mpvpe_list, mpjpe_list, \
                rot_dist_list, trans_err_list, gt_contact_percent_list, contact_percent_list, \
                gt_foot_sliding_jnts_list, foot_sliding_jnts_list, contact_precision_list, contact_recall_list, \
                contact_acc_list, contact_f1_score_list, obj_rot_dist_list, obj_com_pos_err_list, \
                start_obj_com_pos_err_list, end_obj_com_pos_err_list, waypoints_xy_pos_err_list, \
                gt_penetration_score_list, penetration_score_list, \
                gt_hand_penetration_score_list, hand_penetration_score_list, \
                gt_floor_height_list, pred_floor_height_list, \
                dest_metric_folder, seq_name=None): 
        
        mean_lhand_jpe = np.asarray(lhand_jpe_list).mean()
        mean_rhand_jpe = np.asarray(rhand_jpe_list).mean() 
        mean_hand_jpe = np.asarray(hand_jpe_list).mean() 
        mean_mpjpe = np.asarray(mpjpe_list).mean() 
        mean_mpvpe = np.asarray(mpvpe_list).mean() 
        mean_root_trans_err = np.asarray(trans_err_list).mean()
        mean_rot_dist = np.asarray(rot_dist_list).mean() 

        mean_fsliding_jnts = np.asarray(foot_sliding_jnts_list).mean()
        mean_gt_fsliding_jnts = np.asarray(gt_foot_sliding_jnts_list).mean() 
        
        mean_contact_percent = np.asarray(contact_percent_list).mean()
        mean_gt_contact_percent = np.asarray(gt_contact_percent_list).mean() 

        mean_contact_precision = np.asarray(contact_precision_list).mean()
        mean_contact_recall = np.asarray(contact_recall_list).mean() 
        mean_contact_acc = np.asarray(contact_acc_list).mean()
        mean_contact_f1_score = np.asarray(contact_f1_score_list).mean() 

        mean_obj_rot_dist = np.asarray(obj_rot_dist_list).mean() 
        mean_obj_com_pos_err = np.asarray(obj_com_pos_err_list).mean() 

        mean_start_obj_com_pos_err = np.asarray(start_obj_com_pos_err_list).mean() 
        mean_end_obj_com_pos_err = np.asarray(end_obj_com_pos_err_list).mean() 
        mean_waypoints_xy_pos_err = np.asarray(waypoints_xy_pos_err_list).mean()

        mean_penetration_score = np.asarray(penetration_score_list).mean()
        mean_gt_penetration_score = np.asarray(gt_penetration_score_list).mean() 

        mean_hand_penetration_score = np.asarray(hand_penetration_score_list).mean()
        mean_gt_hand_penetration_score = np.asarray(gt_hand_penetration_score_list).mean() 

        mean_gt_floor_height = np.asarray(gt_floor_height_list).mean() 
        mean_pred_floor_height = np.asarray(pred_floor_height_list).mean() 

        print("The number of sequences: {0}".format(len(mpjpe_list)))
        print("*********************************Human Motion Evaluation**************************************")
        print("Left Hand JPE: {0}, Right Hand JPE: {1}, Two Hands JPE: {2}".format(mean_lhand_jpe, mean_rhand_jpe, mean_hand_jpe))
        print("MPJPE: {0}, MPVPE: {1}, Root Trans: {2}, Global Rot Err: {3}".format(mean_mpjpe, mean_mpvpe, mean_root_trans_err, mean_rot_dist))
        print("Foot sliding jnts: {0}, GT Foot sliding jnts: {1}".format(mean_fsliding_jnts, mean_gt_fsliding_jnts))
        print("Floor Height: {0}, GT Floor Height: {1}".format(mean_pred_floor_height, mean_gt_floor_height))
        
        print("*********************************Object Motion Evaluation**************************************")
        print("Object com pos err: {0}, Object rotation err: {1}".format(mean_obj_com_pos_err, mean_obj_rot_dist))

        print("*********************************Interaction Evaluation**************************************")
        print("Hand-Object Penetration Score: {0}".format(mean_hand_penetration_score))
        print("GT Hand-Object Penetration Score: {0}".format(mean_gt_hand_penetration_score))
        print("Human-Object Penetration Score: {0}".format(mean_penetration_score))
        print("GT Human-Object Penetration Score: {0}".format(mean_gt_penetration_score))
        
        print("Contact precision: {0}, Contact recall: {1}".format(mean_contact_precision, mean_contact_recall))
        print("Contact Acc: {0}, Contact F1 score: {1}".format(mean_contact_acc, mean_contact_f1_score)) 
        print("Contact percentage: {0}, GT Contact percentage: {1}".format(mean_contact_percent, mean_gt_contact_percent))

        print("*********************************Condition Following Evaluation**************************************")
        print("Start obj_com_pos err: {0}, End obj_com_pos err: {1}".format(mean_start_obj_com_pos_err, mean_end_obj_com_pos_err))
        print("waypoints xy err: {0}".format(mean_waypoints_xy_pos_err)) 

        # Save the results to json files. 
        if not os.path.exists(dest_metric_folder):
            os.makedirs(dest_metric_folder) 
        if seq_name is not None: # number for all the testing data
            dest_metric_json_path = os.path.join(dest_metric_folder, seq_name+".json")
        else:
            dest_metric_json_path = os.path.join(dest_metric_folder, "evaluation_metrics_for_all_test_data.json")

        metric_dict = {}
        metric_dict['mean_lhand_jpe'] = mean_lhand_jpe
        metric_dict['mean_rhand_jpe'] = mean_rhand_jpe 
        metric_dict['mean_hand_jpe'] = mean_hand_jpe 
        metric_dict['mean_mpjpe'] = mean_mpjpe 
        metric_dict['mean_mpvpe'] = mean_mpvpe 
        metric_dict['mean_root_trans_err'] = mean_root_trans_err 
        metric_dict['mean_rot_dist'] = mean_rot_dist 

        metric_dict['mean_floor_height'] = mean_pred_floor_height
        metric_dict['mean_gt_floor_height'] = mean_gt_floor_height 

        metric_dict['mean_fsliding_jnts'] = mean_fsliding_jnts 
        metric_dict['mean_gt_fsliding_jnts'] = mean_gt_fsliding_jnts 

        metric_dict['mean_contact_percent'] = mean_contact_percent
        metric_dict['mean_gt_contact_percent'] = mean_gt_contact_percent  

        metric_dict['mean_contact_precision'] = mean_contact_precision 
        metric_dict['mean_contact_recall'] = mean_contact_recall 
        metric_dict['mean_contact_acc'] = mean_contact_acc 
        metric_dict['mean_contact_f1_score'] = mean_contact_f1_score 

        metric_dict['mean_obj_rot_dist'] = mean_obj_rot_dist 
        metric_dict['mean_obj_com_pos_err'] = mean_obj_com_pos_err

        metric_dict['mean_start_obj_com_pos_err'] = mean_start_obj_com_pos_err 
        metric_dict['mean_end_obj_com_pos_err'] = mean_end_obj_com_pos_err 
        metric_dict['mean_waypoints_xy_pos_err'] = mean_waypoints_xy_pos_err 

        metric_dict['mean_penetration_score'] = mean_penetration_score 
        metric_dict['mean_gt_penetration_score'] = mean_gt_penetration_score 

        metric_dict['mean_hand_penetration_score'] = mean_hand_penetration_score 
        metric_dict['mean_gt_hand_penetration_score'] = mean_gt_hand_penetration_score 

        # Convert all to float 
        for k in metric_dict:
            metric_dict[k] = float(metric_dict[k])

        json.dump(metric_dict, open(dest_metric_json_path, 'w'))

    def print_evaluation_metrics_for_long_seq(self, foot_sliding_jnts_list, \
                pred_floor_height_list, contact_percent_list, \
                start_obj_com_pos_err_list, end_obj_com_pos_err_list, waypoints_xy_pos_err_list, \
                penetration_score_list, \
                hand_penetration_score_list, \
                scene_human_penetration_list, \
                scene_object_penetration_list, \
                dest_metric_folder, seq_name=None): 
        
        mean_fsliding_jnts = np.asarray(foot_sliding_jnts_list).mean()
       
        mean_contact_percent = np.asarray(contact_percent_list).mean()

        mean_start_obj_com_pos_err = np.asarray(start_obj_com_pos_err_list).mean() 
        mean_end_obj_com_pos_err = np.asarray(end_obj_com_pos_err_list).mean() 
        mean_waypoints_xy_pos_err = np.asarray(waypoints_xy_pos_err_list).mean()

        mean_penetration_score = np.asarray(penetration_score_list).mean()

        mean_hand_penetration_score = np.asarray(hand_penetration_score_list).mean()

        mean_pred_floor_height = np.asarray(pred_floor_height_list).mean() 

        mean_scene_human_penetration = np.asarray(scene_human_penetration_list).mean()
        mean_scene_object_penetration = np.asarray(scene_object_penetration_list).mean() 

        print("The number of sequences: {0}".format(len(foot_sliding_jnts_list)))
        print("*********************************Human Motion Evaluation**************************************")
        print("Foot sliding jnts: {0}".format(mean_fsliding_jnts))
        print("Floor Height: {0}".format(mean_pred_floor_height))
      
        print("*********************************Interaction Evaluation**************************************")
        print("Hand-Object Penetration Score: {0}".format(mean_hand_penetration_score))
        print("Human-Object Penetration Score: {0}".format(mean_penetration_score))
        print("Contact percentage: {0}".format(mean_contact_percent))
        print("Scene-Human Penetration Score: {0}".format(mean_scene_human_penetration))
        print("Scene-Object Penetration Score: {0}".format(mean_scene_object_penetration))

        print("*********************************Condition Following Evaluation**************************************")
        print("Start obj_com_pos err: {0}, End obj_com_pos err: {1}".format(mean_start_obj_com_pos_err, mean_end_obj_com_pos_err))
        print("waypoints xy err: {0}".format(mean_waypoints_xy_pos_err)) 

        # Save the results to json files. 
        if not os.path.exists(dest_metric_folder):
            os.makedirs(dest_metric_folder) 
        if seq_name is not None: # number for all the testing data
            dest_metric_json_path = os.path.join(dest_metric_folder, seq_name+".json")
        else:
            dest_metric_json_path = os.path.join(dest_metric_folder, \
            self.test_scene_name+"_evaluation_metrics_for_all_test_data.json")

        metric_dict = {}
        metric_dict['mean_floor_height'] = mean_pred_floor_height
    
        metric_dict['mean_fsliding_jnts'] = mean_fsliding_jnts 

        metric_dict['mean_contact_percent'] = mean_contact_percent

        metric_dict['mean_start_obj_com_pos_err'] = mean_start_obj_com_pos_err 
        metric_dict['mean_end_obj_com_pos_err'] = mean_end_obj_com_pos_err 
        metric_dict['mean_waypoints_xy_pos_err'] = mean_waypoints_xy_pos_err 

        metric_dict['mean_penetration_score'] = mean_penetration_score 
        metric_dict['mean_hand_penetration_score'] = mean_hand_penetration_score 

        metric_dict['mean_scene_human_penetration_score'] = mean_scene_human_penetration
        metric_dict['mean_scene_object_penetration_score'] = mean_scene_object_penetration  

        # Convert all to float 
        for k in metric_dict:
            metric_dict[k] = float(metric_dict[k])

        json.dump(metric_dict, open(dest_metric_json_path, 'w'))

    def compute_hand_penetration_metric(self, object_name, ori_verts_pred, \
        pred_obj_com_pos, pred_obj_rot_mat, eval_fullbody=False):
        # ori_verts_pred: T X Nv X 3 
        # pred_obj_com_pos: T X 3
        # pred_obj_rot_mat: T X 3 X 3
        ori_verts_pred = ori_verts_pred[None] # 1 X T X Nv X 3 
        pred_obj_com_pos = pred_obj_com_pos[None] # 1 X T X 3 
        pred_obj_rot_mat = pred_obj_rot_mat[None] # 1 X T X 3 X 3 

        if not eval_fullbody:
            hand_verts = ori_verts_pred[:, :, self.hand_vertex_idxs, :] # BS X T X N_hand X 3
        else:
            hand_verts = ori_verts_pred 

        hand_verts_in_rest_frame = hand_verts - pred_obj_com_pos[:, :, None, :] # BS X T X N_hand X 3 
        hand_verts_in_rest_frame = torch.matmul(pred_obj_rot_mat[:, :, None, :, :].repeat(1, 1, \
                            hand_verts_in_rest_frame.shape[2], 1, 1), \
                            hand_verts_in_rest_frame[:, :, :, :, None]).squeeze(-1) # BS X T X N_hand X 3 

        curr_object_sdf, curr_object_sdf_centroid, curr_object_sdf_extents = \
        self.load_object_sdf_data(object_name)

        # Convert hand vertices to align with rest pose object. 
        signed_dists = compute_signed_distances(curr_object_sdf, curr_object_sdf_centroid, \
            curr_object_sdf_extents, hand_verts_in_rest_frame[0]) # we always use bs = 1 now!!!                          
        # signed_dists: T X N_hand (120 X 1535)

        penetration_score = torch.minimum(signed_dists, torch.zeros_like(signed_dists)).abs().mean() # The smaller, the better 
        # penetration_score = torch.minimum(signed_dists, torch.zeros_like(signed_dists)).abs().sum()
        return penetration_score.detach().cpu().numpy()  

    def prep_evaluation_metrics_list(self):
        self.lhand_jpe_list = [] 
        self.rhand_jpe_list = [] 
        self.hand_jpe_list = [] 
        self.mpvpe_list = [] 
        self.mpjpe_list = [] 
        self.rot_dist_list = [] 
        self.trans_err_list = [] 

        self.gt_floor_height_list = [] 
        self.floor_height_list = [] 
        
        self.gt_foot_sliding_jnts_list = []
        self.foot_sliding_jnts_list = [] 

        self.gt_contact_percent_list = [] 
        self.contact_percent_list = []
        self.contact_precision_list = [] 
        self.contact_recall_list = [] 
        self.contact_acc_list = [] 
        self.contact_f1_score_list = []

        self.obj_rot_dist_list = []
        self.obj_com_pos_err_list = [] 

        self.start_obj_com_pos_err_list = [] 
        self.end_obj_com_pos_err_list = [] 
        self.waypoints_xy_pos_err_list = []

        self.gt_penetration_list = []
        self.penetration_list = [] 

        self.gt_hand_penetration_list = []
        self.hand_penetration_list = [] 

    def prep_res_folders(self):
        res_root_folder = self.save_res_folder 
        # Prepare folder for saving npz results 
        dest_res_for_eval_npz_folder = os.path.join(res_root_folder, "res_npz_files")
        # Prepare folder for evaluation metrics 
        dest_metric_root_folder = os.path.join(res_root_folder, "evaluation_metrics_json")
        # Prepare folder for visualization 
        dest_out_vis_root_folder = os.path.join(res_root_folder, "single_window_cmp_settings")
        # Prepare folder for saving .obj files 
        dest_out_obj_root_folder = os.path.join(res_root_folder, "objs_single_window_cmp_settings")
       
        if self.test_unseen_objects:
            dest_res_for_eval_npz_folder += "_unseen_obj"
            dest_metric_root_folder += "_unseen_obj"
            dest_out_vis_root_folder += "_unseen_obj"
            dest_out_obj_root_folder += "_unseen_obj"

        # Prepare folder for saving text json files 
        dest_out_text_json_folder = os.path.join(dest_out_vis_root_folder, "text_json_files")

        if self.use_guidance_in_denoising:
            dest_res_for_eval_npz_folder = os.path.join(dest_res_for_eval_npz_folder, "chois")
            dest_metric_folder = os.path.join(dest_metric_root_folder, "chois")
            dest_out_vis_folder = os.path.join(dest_out_vis_root_folder, "chois")
            dest_out_obj_folder = os.path.join(dest_out_obj_root_folder, "chois")
        else:
            if self.use_object_keypoints:
                dest_res_for_eval_npz_folder = os.path.join(dest_res_for_eval_npz_folder, "chois_wo_guidance")
                dest_metric_folder = os.path.join(dest_metric_root_folder, "chois_wo_guidance") 
                dest_out_vis_folder = os.path.join(dest_out_vis_root_folder, "chois_wo_guidance") 
                dest_out_obj_folder = os.path.join(dest_out_obj_root_folder, "chois_wo_guidance")
            else:
                dest_res_for_eval_npz_folder = os.path.join(dest_res_for_eval_npz_folder, "chois_wo_l_geo")  
                dest_metric_folder = os.path.join(dest_metric_root_folder, "chois_wo_l_geo")  
                dest_out_vis_folder = os.path.join(dest_out_vis_root_folder, "chois_wo_l_geo")           
                dest_out_obj_folder = os.path.join(dest_out_obj_root_folder, "chois_wo_l_geo")    
        
      
        # Create folders 
        if not os.path.exists(dest_metric_folder):
            os.makedirs(dest_metric_folder) 
        if not os.path.exists(dest_out_vis_folder):
            os.makedirs(dest_out_vis_folder) 
        if not os.path.exists(dest_res_for_eval_npz_folder):
            os.makedirs(dest_res_for_eval_npz_folder)
        if not os.path.exists(dest_out_obj_folder):
            os.makedirs(dest_out_obj_folder) 
        if not os.path.exists(dest_out_text_json_folder):
            os.makedirs(dest_out_text_json_folder)

        dest_out_gt_vis_folder = os.path.join(dest_out_vis_root_folder, "0_gt")
        if not os.path.exists(dest_out_gt_vis_folder):
            os.makedirs(dest_out_gt_vis_folder) 

        return dest_res_for_eval_npz_folder, dest_metric_folder, dest_out_vis_folder, \
            dest_out_gt_vis_folder, dest_out_obj_folder, dest_out_text_json_folder

    def cond_sample_res(self):
        if self.opt.pretrained_model == "":
            weights = os.listdir(self.results_folder)
            weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
            weight_path = max(weights_paths, key=os.path.getctime)
    
            print(f"Loaded weight: {weight_path}")

            milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "") 
            # milestone = "9" # 9, 10
            
            self.load(milestone)
        else:
            milestone = "10" # 9, 10
            self.load(milestone, pretrained_path=self.opt.pretrained_model) 

        self.ema.ema_model.eval()

        if self.test_on_train:
            test_loader = torch.utils.data.DataLoader(
                self.ds, batch_size=1, shuffle=False,
                num_workers=1, pin_memory=True, drop_last=False) 
        else:
            if self.test_unseen_objects:
                test_loader = torch.utils.data.DataLoader(
                    self.unseen_seq_ds, batch_size=1, shuffle=False,
                    num_workers=1, pin_memory=True, drop_last=False) 
            else:
                test_loader = torch.utils.data.DataLoader(
                    self.val_ds, batch_size=1, shuffle=False,
                    num_workers=1, pin_memory=True, drop_last=False) 

        self.prep_evaluation_metrics_list()

        dest_res_for_eval_npz_folder, dest_metric_folder, dest_out_vis_folder, \
        dest_out_gt_vis_folder, dest_out_obj_folder, dest_out_text_json_folder = self.prep_res_folders() 

        for s_idx, val_data_dict in enumerate(test_loader):

            seq_name_list = val_data_dict['seq_name']
            object_name_list = val_data_dict['obj_name']
            start_frame_idx_list = val_data_dict['s_idx']
            end_frame_idx_list = val_data_dict['e_idx'] 

            val_human_data = val_data_dict['motion'].cuda() 
            val_obj_data = val_data_dict['obj_motion'].cuda()

            obj_bps_data = val_data_dict['input_obj_bps'].cuda().reshape(-1, 1, 1024*3)
            ori_data_cond = obj_bps_data # BS X 1 X (1024*3) 

            rest_human_offsets = val_data_dict['rest_human_offsets'].cuda() # BS X 24 X 3 
            
            if "contact_labels" in val_data_dict:
                contact_labels = val_data_dict['contact_labels'].cuda() # BS X T X 4 
            else:
                contact_labels = None 

            # Generate padding mask 
            actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
            tmp_mask = torch.arange(self.window+1).expand(val_obj_data.shape[0], \
            self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
            # BS X max_timesteps
            padding_mask = tmp_mask[:, None, :].to(val_obj_data.device)

            end_pos_cond_mask = self.prep_start_end_condition_mask_pos_only(val_obj_data, val_data_dict['seq_len'])
            cond_mask = self.prep_mimic_A_star_path_condition_mask_pos_xy_only(val_obj_data, val_data_dict['seq_len'])
            cond_mask = end_pos_cond_mask * cond_mask 

            human_cond_mask = torch.ones_like(val_human_data).to(val_human_data.device)
            if self.input_first_human_pose:
                human_cond_mask[:, 0, :] = 0 
                
            cond_mask = torch.cat((cond_mask, human_cond_mask), dim=-1) # BS X T X (3+6+24*3+22*6)

            if self.use_guidance_in_denoising:
                # Load current sequence's object SDF
                self.object_sdf, self.object_sdf_centroid, self.object_sdf_extents = \
                self.load_object_sdf_data(val_data_dict['obj_name'][0])

                guidance_fn = self.apply_different_guidance_loss 
            else:
                guidance_fn = None 

            if self.compute_metrics:
                num_samples_per_seq = 1 
            else:
                num_samples_per_seq = 1

            val_obj_data = val_obj_data.repeat(num_samples_per_seq, 1, 1) # BS X T X D 
            val_human_data = val_human_data.repeat(num_samples_per_seq, 1, 1) 
            cond_mask = cond_mask.repeat(num_samples_per_seq, 1, 1) 
            padding_mask = padding_mask.repeat(num_samples_per_seq, 1, 1) # BS X 1 X 121 
            ori_data_cond = ori_data_cond.repeat(num_samples_per_seq, 1, 1)  
            rest_human_offsets = rest_human_offsets.repeat(num_samples_per_seq, 1, 1)
           
            contact_data = torch.zeros(val_obj_data.shape[0], val_obj_data.shape[1], 4).to(val_obj_data.device) 
            data = torch.cat((val_obj_data, val_human_data, contact_data), dim=-1) 
            cond_mask = torch.cat((cond_mask, \
                    torch.ones_like(contact_data).to(cond_mask.device)), dim=-1) 
           
            if self.add_language_condition:
                text_anno_data = val_data_dict['text']
                language_input = self.encode_text(text_anno_data) # BS X 512 
                language_input = language_input.to(data.device)
                language_input = language_input.repeat(num_samples_per_seq, 1) 
                all_res_list = self.ema.ema_model.sample(data, ori_data_cond, cond_mask, padding_mask, \
                            language_input=language_input, \
                            rest_human_offsets=rest_human_offsets, guidance_fn=guidance_fn, \
                            data_dict=val_data_dict)
            else:
                all_res_list = self.ema.ema_model.sample(data, ori_data_cond, \
                        cond_mask, padding_mask, \
                        rest_human_offsets=rest_human_offsets, \
                        guidance_fn=guidance_fn, \
                        data_dict=val_data_dict)

            for_vis_gt_data = torch.cat((val_obj_data, val_human_data), dim=-1)

            sample_idx = 0
            vis_tag = str(milestone)+"_sidx_"+str(s_idx)+"_sample_cnt_"+str(sample_idx)

            if self.use_guidance_in_denoising:
                vis_tag = vis_tag + "_w_guidance"

            if self.test_on_train:
                vis_tag = vis_tag + "_on_train"

            if self.test_unseen_objects:
                vis_tag = vis_tag + "_on_unseen_objects"

            curr_seq_name_tag = seq_name_list[0] + "_" + object_name_list[0]+ "_sidx_" + \
                        str(start_frame_idx_list[0].detach().cpu().numpy()) +\
                        "_eidx_" + str(end_frame_idx_list[0].detach().cpu().numpy()) + \
                        "_sample_cnt_" + str(sample_idx)

            dest_text_json_path = os.path.join(dest_out_text_json_folder, curr_seq_name_tag+".json")
            dest_text_json_dict = {}
            dest_text_json_dict['text'] = val_data_dict['text'][0]
            if not os.path.exists(dest_text_json_path):
                json.dump(dest_text_json_dict, open(dest_text_json_path, 'w'))

            curr_dest_out_mesh_folder = os.path.join(dest_out_obj_folder, curr_seq_name_tag) 
            curr_dest_out_vid_path = os.path.join(dest_out_vis_folder, curr_seq_name_tag+".mp4")
            curr_dest_out_gt_vid_path = os.path.join(dest_out_gt_vis_folder, curr_seq_name_tag+".mp4")

            if self.use_object_keypoints:
                all_res_list = all_res_list[:, :, :-4] 

            gt_human_verts_list, gt_human_jnts_list, gt_human_trans_list, gt_human_rot_list, \
            gt_obj_com_pos_list, gt_obj_rot_mat_list, gt_obj_verts_list, human_faces_list, obj_faces_list, _ = \
            self.gen_vis_res_generic(for_vis_gt_data, val_data_dict, milestone, cond_mask, vis_gt=True, \
            curr_object_name=object_name_list[0], vis_tag=vis_tag, \
            dest_out_vid_path=curr_dest_out_gt_vid_path, \
            dest_mesh_vis_folder=curr_dest_out_mesh_folder) 
           
            pred_human_verts_list, pred_human_jnts_list, pred_human_trans_list, pred_human_rot_list, \
            pred_obj_com_pos_list, pred_obj_rot_mat_list, pred_obj_verts_list, _, _, _ = \
            self.gen_vis_res_generic(all_res_list, val_data_dict, milestone, cond_mask, \
            curr_object_name=object_name_list[0], vis_tag=vis_tag, \
            dest_out_vid_path=curr_dest_out_vid_path, dest_mesh_vis_folder=curr_dest_out_mesh_folder)

            # Save results to npz files
            # Save global joint positions to npz files for evaluation (R_precition, FID, etc)
            # tmp_bs = val_obj_data.shape[0]
            tmp_bs = len(seq_name_list) 
            for tmp_bs_idx in range(tmp_bs):
                tmp_seq_name = seq_name_list[tmp_bs_idx]
                tmp_obj_name = object_name_list[tmp_bs_idx]
            
                curr_pred_global_jpos = pred_human_jnts_list[0].detach().cpu().numpy()
                if self.test_unseen_objects:
                    curr_seq_dest_res_npz_path = os.path.join(dest_res_for_eval_npz_folder, \
                                            tmp_seq_name+"_"+tmp_obj_name+".npz")
                else:
                    curr_seq_dest_res_npz_path = os.path.join(dest_res_for_eval_npz_folder, \
                                                        tmp_seq_name+".npz")
                np.savez(curr_seq_dest_res_npz_path, seq_name=tmp_seq_name, \
                        global_jpos=curr_pred_global_jpos) # T X 24 X 3 

            for tmp_s_idx in range(num_samples_per_seq):
                # Compute evaluation metrics 
                lhand_jpe, rhand_jpe, hand_jpe, mpvpe, mpjpe, rot_dist, trans_err, \
                gt_contact_percent, contact_percent, \
                gt_foot_sliding_jnts, foot_sliding_jnts, \
                contact_precision, contact_recall, contact_acc, contact_f1_score, \
                obj_rot_dist, obj_com_pos_err, start_obj_com_pos_err, end_obj_com_pos_err, waypoints_xy_pos_err, \
                gt_floor_height, pred_floor_height = \
                        compute_metrics(gt_human_verts_list[tmp_s_idx], pred_human_verts_list[tmp_s_idx], \
                        gt_human_jnts_list[tmp_s_idx], pred_human_jnts_list[tmp_s_idx], \
                        human_faces_list[tmp_s_idx], \
                        gt_human_trans_list[tmp_s_idx], pred_human_trans_list[tmp_s_idx], \
                        gt_human_rot_list[tmp_s_idx], pred_human_rot_list[tmp_s_idx], \
                        gt_obj_com_pos_list[tmp_s_idx], pred_obj_com_pos_list[tmp_s_idx], \
                        gt_obj_rot_mat_list[tmp_s_idx], pred_obj_rot_mat_list[tmp_s_idx], \
                        gt_obj_verts_list[tmp_s_idx], pred_obj_verts_list[tmp_s_idx], \
                        obj_faces_list[tmp_s_idx], val_data_dict['seq_len'])

                pred_hand_penetration_score = self.compute_hand_penetration_metric(object_name_list[0], \
                                    pred_human_verts_list[tmp_s_idx], \
                                    pred_obj_com_pos_list[tmp_s_idx], pred_obj_rot_mat_list[tmp_s_idx])
                gt_hand_penetration_score = self.compute_hand_penetration_metric(object_name_list[0], \
                                    gt_human_verts_list[tmp_s_idx], \
                                    gt_obj_com_pos_list[tmp_s_idx], gt_obj_rot_mat_list[tmp_s_idx])

                pred_penetration_score = self.compute_hand_penetration_metric(object_name_list[0], \
                                    pred_human_verts_list[tmp_s_idx], \
                                    pred_obj_com_pos_list[tmp_s_idx], pred_obj_rot_mat_list[tmp_s_idx], eval_fullbody=True)
                gt_penetration_score = self.compute_hand_penetration_metric(object_name_list[0], \
                                    gt_human_verts_list[tmp_s_idx], \
                                    gt_obj_com_pos_list[tmp_s_idx], gt_obj_rot_mat_list[tmp_s_idx], eval_fullbody=True)
               
                self.append_new_value_to_metrics_list(lhand_jpe, rhand_jpe, hand_jpe, mpvpe, mpjpe, rot_dist, trans_err, \
                gt_contact_percent, contact_percent, gt_foot_sliding_jnts, foot_sliding_jnts, \
                contact_precision, contact_recall, contact_acc, contact_f1_score, \
                obj_rot_dist, obj_com_pos_err, \
                start_obj_com_pos_err, end_obj_com_pos_err, waypoints_xy_pos_err, \
                gt_penetration_score, pred_penetration_score, \
                gt_hand_penetration_score, pred_hand_penetration_score, \
                gt_floor_height, pred_floor_height) 

                # Print current seq's evaluation metrics. 
                curr_seq_name_tag = seq_name_list[0] + "_" + object_name_list[0]+ "_sidx_" + str(start_frame_idx_list[0].detach().cpu().numpy()) +\
                        "_eidx_" + str(end_frame_idx_list[0].detach().cpu().numpy()) + "_sample_cnt_" + str(tmp_s_idx)
                print("Current Sequence name:{0}".format(curr_seq_name_tag))
                self.print_evaluation_metrics([lhand_jpe], [rhand_jpe], [hand_jpe], [mpvpe], [mpjpe], \
                [rot_dist], [trans_err], \
                [gt_contact_percent], [contact_percent], \
                [gt_foot_sliding_jnts], [foot_sliding_jnts], \
                [contact_precision], [contact_recall], [contact_acc], [contact_f1_score], \
                [obj_rot_dist], [obj_com_pos_err], \
                [start_obj_com_pos_err], [end_obj_com_pos_err], [waypoints_xy_pos_err], \
                [gt_penetration_score], [pred_penetration_score], \
                [gt_hand_penetration_score], [pred_hand_penetration_score], \
                [gt_floor_height], [pred_floor_height], \
                dest_metric_folder, curr_seq_name_tag) # Assume batch size = 1 

            torch.cuda.empty_cache()

        self.print_evaluation_metrics(self.lhand_jpe_list, self.rhand_jpe_list, self.hand_jpe_list, self.mpvpe_list, self.mpjpe_list, \
            self.rot_dist_list, self.trans_err_list, self.gt_contact_percent_list, self.contact_percent_list, \
            self.gt_foot_sliding_jnts_list, self.foot_sliding_jnts_list, \
            self.contact_precision_list, self.contact_recall_list, \
            self.contact_acc_list, self.contact_f1_score_list, \
            self.obj_rot_dist_list, self.obj_com_pos_err_list, \
            self.start_obj_com_pos_err_list, self.end_obj_com_pos_err_list, self.waypoints_xy_pos_err_list, \
            self.gt_penetration_list, self.penetration_list, self.gt_hand_penetration_list, self.hand_penetration_list, \
            self.gt_floor_height_list, self.floor_height_list, \
            dest_metric_folder)   
    
    def gen_longest_waypoints_for_seq(self, root_trans, obj_com_pos):
        # root_trans: T X 3 
        # obj_com_pos: T X 3 
        # Test the longest sequence the model can generate that follows the waypoints.
        init_root_dir = obj_com_pos[0:1] - root_trans[0:1] # 1 X 3 
        init_root_dir[:, -1] = 0 # zero out z(height)
        init_root_dir = F.normalize(init_root_dir) # 1 X 3 

        max_seq_len = 120*1

        new_obj_com_pos = torch.zeros(max_seq_len, 3).float().to(obj_com_pos.device)
        
        new_obj_com_pos[0:1] = obj_com_pos[0:1].clone() # Copy the initial position 

        # Sample waypoints along the straight line of the initial facing direction 
        interval_dist = 0.5  
        for idx in range(1, max_seq_len-1):
            new_obj_com_pos[idx:idx+1, :] = obj_com_pos[0:1] + \
                    init_root_dir * interval_dist * (idx-1)  
        
        new_obj_com_pos[-1:, :] = obj_com_pos[0:1] + init_root_dir * interval_dist * (max_seq_len-3) 

        # Use the initial frame's height for end frame's height to put the object on the floor. 
        new_obj_com_pos[-1:, -1] = obj_com_pos[0:1, -1]

        return new_obj_com_pos 
                
    def load_end_frame_height_heuristics(self, action_name, object_name):
        heuristic_dict = {}

        heuristic_dict['push'] = {}
        heuristic_dict['pull'] = {}
        heuristic_dict['lift'] = {}
        heuristic_dict['kick'] = {} 
        
        # 1. Floorlamp type 
        heuristic_dict['push']['floorlamp'] = [0.8, 0.9] # not used!
        heuristic_dict['pull']['floorlamp'] = [0.8, 0.9]
        heuristic_dict['lift']['floorlamp'] = [1.0, 1.3]
        heuristic_dict['kick']['floorlamp'] = [0.8, 0.9] 

        heuristic_dict['push']['clothesstand'] = [0.4, 0.55] # not used! 
        heuristic_dict['pull']['clothesstand'] = [0.4, 0.55]
        heuristic_dict['lift']['clothesstand'] = [0.8, 0.9]
        heuristic_dict['kick']['clothesstand'] = [0.4, 0.55]

        heuristic_dict['push']['tripod'] = [0.4, 0.55] # not used!
        heuristic_dict['pull']['tripod'] = [0.4, 0.55]
        heuristic_dict['lift']['tripod'] = [0.8, 1.1]
        heuristic_dict['kick']['tripod'] = [0.4, 0.55]
        
        # 2. Table type 
        heuristic_dict['push']['largetable'] = [0.35, 0.37]
        heuristic_dict['pull']['largetable'] = [0.35, 0.4]
        heuristic_dict['lift']['largetable'] = [0.8, 1.1] 
        heuristic_dict['kick']['largetable'] = [0.35, 0.37]

        heuristic_dict['push']['smalltable'] = [0.26, 0.27]
        heuristic_dict['pull']['smalltable'] = [0.26, 0.31]
        heuristic_dict['lift']['smalltable'] = [0.8, 1.1]
        heuristic_dict['kick']['smalltable'] = [0.26, 0.27]

        # 3. Chair type 
        heuristic_dict['push']['woodchair'] = [0.44, 0.45]
        heuristic_dict['pull']['woodchair'] = [0.48, 0.50]
        heuristic_dict['lift']['woodchair'] = [0.6, 1.1]
        heuristic_dict['kick']['woodchair'] = [0.44, 0.45]

        heuristic_dict['push']['whitechair'] = [0.46, 0.47]
        heuristic_dict['pull']['whitechair'] = [0.50, 0.52]
        heuristic_dict['lift']['whitechair'] = [0.6, 1.0]
        heuristic_dict['kick']['whitechair'] = [0.46, 0.47]

        # 4. Box type 
        heuristic_dict['push']['smallbox'] = [0.062, 0.068]
        heuristic_dict['pull']['smallbox'] = [0.062, 0.068]
        heuristic_dict['lift']['smallbox'] = [1.0, 1.2]
        heuristic_dict['kick']['smallbox'] = [0.062, 0.068]

        heuristic_dict['push']['largebox'] = [0.155, 0.16]
        heuristic_dict['pull']['largebox'] = [0.155, 0.16]
        heuristic_dict['lift']['largebox'] = [1.1, 1.15]
        heuristic_dict['kick']['largebox'] = [0.155, 0.16]

        heuristic_dict['push']['plasticbox'] = [0.13, 0.14]
        heuristic_dict['pull']['plasticbox'] = [0.13, 0.14]
        heuristic_dict['lift']['plasticbox'] = [0.9, 1.5]
        heuristic_dict['kick']['plasticbox'] = [0.13, 0.14]

        heuristic_dict['push']['suitcase'] = [0.322, 0.324]
        heuristic_dict['pull']['suitcase'] = [0.322, 0.324]
        heuristic_dict['lift']['suitcase'] = [1.1, 1.5]
        heuristic_dict['kick']['suitcase'] = [0.322, 0.324]

        # 5. Monitor type 
        heuristic_dict['push']['monitor'] = [0.23, 0.25] # not used!
        heuristic_dict['pull']['monitor'] = [0.23, 0.28] 
        heuristic_dict['lift']['monitor'] = [0.8, 1.2]

        # 6. Trashcan type 
        heuristic_dict['push']['trashcan'] = [0.15, 0.155]
        heuristic_dict['pull']['trashcan'] = [0.15, 0.155]
        heuristic_dict['lift']['trashcan'] = [0.85, 1.25]
        heuristic_dict['kick']['trashcan'] = [0.15, 0.155]

        return heuristic_dict[action_name][object_name] 

    def prepare_text_for_same_waypoints(self):
        chair_mapping_dict = {
            0: "Facing the back of the chair, lift the chair, move the chair, and then place the chair on the floor.",  
            1: "Lift the chair, move the chair, and put down the chair.", 
            2: "Grab the top of the chair, swing the chair, and put down the chair.",
            3: "Lift the chair over your head, walk and then place the chair on the floor.", 
            4: "Put your hand on the back of the chair, pull the chair, and set it back down.", 
            5: "Lift the chair, rotate the chair, and set it back down.", 
            6: "Use the foot to scoot the chair to change its orientation.", 
            7: "Push the chair, release the hands, then drag the chair, and set it back down.", 
            8: "Hold the chair and turn it around to face a diffferent orientation.", 
            9: "Grab the chair's legs, tilt the chair.", 
            10: "Kick the chair, and set it back down.", 
        }

        table_mapping_dict = {
            0: "Pull the table, and set it back down.",
            1: "Lift the table, move the table and put down the table.",
            2: "Lift the table above your head, spin it and put the table down.",
            3: "Lift the table above your head, walk, and put the table down.",
            4: "Lift the table by one of the edges, rotate it, drag the table and place the legs back down.",
            5: "Push the table, and set it back down.",
            6: "Kick the table, and set it back down.",
            7: "Push the table, release the hands, then drag the table, and set it back down.", 
            8: "Lift the table, so only two legs are off the floor. Slide your feet and rotate the table as you slide. Lower the table with your hands.",
        }

        box_mapping_dict = {
            0: "Push the box, and set it back down.",
            1: "Lift the box, move the box, and put down the box.",
            2: "Lift the box, rotate the box, and set it back down.",
            3: "Kick the box, lift the box, move the box, and put down the box.",
            4: "Kick the box, and set it back down.",
            5: "Pull the box, and set it back down.",
            6: "Push the box, release the hands, then pull the box, and set it back down.",
        }
            
        monitor_mapping_dict = {
            0: "Lift the monitor, move the monitor, and put down the monitor.",
            1: "Lift the monitor, put down the monitor, and then rotate the monitor to adjust its orientation..",
            2: "Lift the monitor, rotate the monitor, and put down the monitor.",
            3: "Grasp the sides of the monitor, tilt it, pull the monitor, and set it back down.", 
            4: "Lift the monitor, put down the monitor while rotating it.",
        }

        floorlamp_mapping_dict = {
            0: "Lift the floorlamp, move the floorlamp, and put down the floorlamp.",
            1: "Pull the floorlamp, and set it back down.",
            2: "Kick the base of the floorlamp, and set it back down.",
            3: "Lift the floorlamp, adjust the orientation of the floorlamp, and set it back down.",
        }
            
        tripod_mapping_dict = {
            0: "Lift the tripod, move the tripod, and put down the tripod.",
            1: "Pull the tripod, and set it back down.",
            2: "Put the tripod horizontally down, then pick up the fallen tripod.",
            3: "Put the tripod horizontally down.",
            4: "Pick up the fallen tripod.",
            5: "Hold and turn the tripod around to a different orientation.",
            6: "Kick the tripod, and set it back down.",
        }

        object_mapping_dict = {
            "woodchair": chair_mapping_dict, 
            "smalltable": table_mapping_dict,
            "largetable": table_mapping_dict, 
            "largebox": box_mapping_dict, 
            "plasticbox": box_mapping_dict, 
            "suitcase": box_mapping_dict, 
            "floorlamp": floorlamp_mapping_dict, 
            "monitor": monitor_mapping_dict, 
            # "clothesstand": tripod_mapping_dict, 
        }

        return object_mapping_dict 

    def get_long_planned_path_names_new(self):
        # Remove the target object name from the text since the model is trained using text without specifying the target object name. 
        # This mapping can also be done using LLM. 
        # Not sure how this will affect the model's performance, never tested. 
        text_mapping_dict = {
            "Pick up floorlamp, move floorlamp to be close to the sofa.": "Lift the floorlamp, move the floorlamp, and put down the floorlamp.", 
            "Pick up floorlamp, move floorlamp to be close to the shelf.": "Lift the floorlamp, move the floorlamp, and put down the floorlamp.", 
            "Pull the floorlamp to be close to the sofa.": "Pull the floorlamp, and set it back down.", 
            "Pull the floorlamp to be close to the shelf.": "Pull the floorlamp, and set it back down.", 
            "Lift the table, move the table to be close to the table.": "Lift the table, move the table and put down the table.",
            "Push the table to be close to the shelf.": "Push the table, and set it back down.", 
            "Pull the table to be close to the countertop.": "Pull the table, and set it back down.",
            "Lift the chair, move the chair to be close to the table.": "Lift the chair, move the chair, and put down the chair.", 
            "Pull the chair to be close to the table.": "Put your hand on the back of the chair, pull the chair, and set it back down.", 
            "Lift a box, move the box and put down on the table.": "Lift the box, move the box, and put down the box.", 
            "Lift a box, move the box and put down on the countertop.": "Lift the box, move the box, and put down the box.", 
            "Lift a box, move the box and place it under the table.": "Lift the box, move the box, and put down the box.", 
            "Push the box to be next to the countertop.": "Push the box, and set it back down.", 
            "Lift a monitor, move the monitor and put down on the table.": "Lift the monitor, move the monitor, and put down the monitor.",
            "Lift a monitor, move the monitor and put down on the floor in front of the tv-screen.": "Lift the monitor, move the monitor, and put down the monitor.",
            "Lift a trashcan, move the trashcan to be close to the shelf.": "Lift the trashcan, move the trashcan, and put down the trashcan.", 
            "Lift a trashcan, move the trashcan to be close to the refrigerator.": "Lift the trashcan, move the trashcan, and put down the trashcan.",
        }

        object2category_dict = {"floorlamp": "floorlamp", "clothesstand": "floorlamp", "tripod": "floorlamp", \
            "largetable": "table", "smalltable": "table", \
            "woodchair": "chair", "whitechair": "chair", \
            "smallbox": "box", "largebox": "box", "plasticbox": "box", "suitcase": "box", \
            "monitor": "monitor", \
            "trashcan": "trashcan"}

        height_json_path = os.path.join(self.data_root_folder, "replica_processed/scene_floor_height.json")
        scene_floor_dict = json.load(open(height_json_path, 'r'))

        text_idx_json_path = "utils/create_eval_dataset/eval_dataset_response_text_idx.json"
        text_dict = json.load(open(text_idx_json_path, 'r'))

        seq_json_path = "utils/create_eval_dataset/selected_long_seq_names.json"
        json_data = json.load(open(seq_json_path, 'r'))
        npy_root_folder = os.path.join(self.data_root_folder, "replica_processed/replica_single_object_long_seq_data_selected")

        object_name_data_dict = {} 

        total_cnt = 0
        for k in json_data:
            npy_name_tag = json_data[k]
            npy_path = os.path.join(npy_root_folder, npy_name_tag) 

            text_idx = npy_path.split("/")[-2]
            if int(text_idx) == 14:
                continue  
            # Remove text_idx = 14 since the parsing is wrong! the target object position is not right. 
         
            # When text_idx = 9, 10, 13, this is related to sample point on a top surface. Nedd to use the point extracted from the scene. 
            # Otherwise, can just use floor height as the target position. 
            # The 9,10,13 target 3d position is the real table height position, shouldn't use the relative height wrt the first frame. The first frame's height is not equal to floor height. 

            npy_data = np.load(npy_path)
            if npy_data.shape[0] < 9: # Discard single window sequence, we want to generate long sequence here 
                continue 

            curr_scene_name = npy_name_tag.split("/")[0]

            if curr_scene_name not in self.test_scene_name:
                continue 

            curr_object_name = npy_name_tag.split("/")[1]


            if curr_object_name not in object_name_data_dict:
               
                object_name_data_dict[curr_object_name] = {}
                object_name_data_dict[curr_object_name]['scene_list'] = []
                object_name_data_dict[curr_object_name]['text_list'] = []
                object_name_data_dict[curr_object_name]['height_range'] = []
                object_name_data_dict[curr_object_name]['npy_list'] = []

         
            curr_text_idx = npy_name_tag.split("/")[-2]
            curr_seq_text = text_dict[curr_text_idx]['text']

            if "Lift" in curr_seq_text or "lift" in curr_seq_text or "pick" in curr_seq_text or "Pick" in curr_seq_text:
                action_name = "lift"
            elif "Push" in curr_seq_text or "push" in curr_seq_text:
                action_name = "push"
            elif "Pull" in curr_seq_text or "pull" in curr_seq_text:
                action_name = "pull" 
            elif "Kick" in curr_seq_text or "kick" in curr_seq_text:
                action_name = "kick"

            curr_height_range = self.load_end_frame_height_heuristics(action_name, curr_object_name) 

            curr_object_type = object2category_dict[curr_object_name]
            curr_converted_text = text_mapping_dict[curr_seq_text].replace(curr_object_type, curr_object_name)

            object_name_data_dict[curr_object_name]['scene_list'].append(curr_scene_name)
            object_name_data_dict[curr_object_name]['text_list'].append(curr_converted_text) 
            object_name_data_dict[curr_object_name]['height_range'].append(curr_height_range) 
            object_name_data_dict[curr_object_name]['npy_list'].append(npy_path) 

            total_cnt += 1

        print("Total numebr of sequecens for testing on 3D scene:{0}".format(total_cnt))

        return object_name_data_dict 

    def process_long_path_for_unseen_objects(self, object_name_data_dict):
        unseen_object_corr_dict = {
            "largetable": ["0a2378ae-4042-4411-91cc-5e9d868ec63b", "6e31af89-62e3-4ec5-9718-4448bcba6557", \
                        "343dad12-3fba-4f75-8fa3-19f7aa3d5871", ], \
            "smalltable": ["0c51740b-7fd0-4b11-91fe-fb35553d6b4e", "0f95f670-3100-4cbd-85bd-e2d2167dd450"], \
            "woodchair": ["0acc134e-65bc-4b68-b836-c591935bdec6", "0bd81869-830e-4673-bfeb-e4d1e5c60302", \
                        "0cd722cb-4cf5-4026-9906-187e51d78c67", "3a431666-c294-41c8-85ca-2247a19e3671"], \
            "floorlamp": ["0c15e144-0244-4c05-957b-ed5d0a96e38e", "0d7421d4-2656-435b-9eb1-8884b4b3dcb3", \
                        "6f3f7829-e4a6-470f-b59b-98e97921de1d"], \
            "largebox": ["0efd1942-d8aa-4749-a3f7-fb10ed93e1c4", "1d9698ef-2edf-4398-946c-e24de4d33c1f"], \
            "smallbox": ["3a392911-1020-4c27-8d1e-2b39e3e69c22", "8dcdf4ef-6969-41af-b203-0bc7c22acb35", \
                        "111030f2-a815-4538-8ef4-be506bdbcf01"], \
        }

        new_object_name_data_dict = {}
        for ori_obj_name in object_name_data_dict:
            if ori_obj_name in unseen_object_corr_dict:
                curr_object_name_list = unseen_object_corr_dict[ori_obj_name]
                for curr_obj_name in curr_object_name_list:
                    new_object_name_data_dict[curr_obj_name] = object_name_data_dict[ori_obj_name]

        return new_object_name_data_dict

    def canonizalize_planned_path(self, planned_points):
        # We want to canonicalize the direction of the init planned path to align with human's forward direction. 
        # planned_points: T X 3
        num_steps = planned_points.shape[0]

        forward = normalize(planned_points[2, :].data.cpu().numpy()-planned_points[1, :].data.cpu().numpy()) 
        yrot = quat_normalize(quat_between(forward, np.array([1, 0, 0]))) # 4-dim, from current direction to canonicalized direction
        # yrot = quat_normalize(quat_between(forward, np.array([0, -1, 0]))) # 4-dim, from current direction to canonicalized direction
        cano_quat = torch.from_numpy(yrot).float()[None, :].repeat(num_steps, 1) # T X 4 

        # Apply rotation to the original path. 
        canonicalized_path_pts = transforms.quaternion_apply(cano_quat, planned_points) # T X 3 
        
        return cano_quat, canonicalized_path_pts 

    def load_planned_path_as_waypoints_new(self, long_seq_path, \
                                use_canonicalization=True, return_scene_names=False):
       
        selected_npy_path = long_seq_path 

        npy_data = np.load(selected_npy_path) # K X 3 (xyz, y represents the floor in Habitat) 

        planned_data = torch.from_numpy(npy_data).float() # T X 3  
        
        if use_canonicalization:
            cano_quat, cano_planned_data = self.canonizalize_planned_path(planned_data) # T X 3 
        else:
            cano_planned_data = planned_data

        if return_scene_names and use_canonicalization:
            return cano_quat, cano_planned_data, selected_npy_path
        elif return_scene_names: # not canonicalization 
            return cano_planned_data, selected_npy_path
        else:
            return cano_planned_data

    def sample_dense_waypoints(self, waypoints, distance_range=(0.6, 0.8), remainder=1):
        # Validate the remainder value
        assert remainder in [0, 1, 2, 3], "Remainder must be one of [0, 1, 2, 3]."
        
        # Compute the distances between each consecutive pair of waypoints
        segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=-1)
        
        # For each segment, compute the number of intermediate points to insert
        num_points_per_segment = np.ceil(segment_lengths / distance_range[0]) - 1
        num_points_per_segment = np.maximum(num_points_per_segment, 
                                            np.floor(segment_lengths / distance_range[1]) - 1).astype(int)

        # Create dense waypoints by interpolating for each segment
        dense_waypoints = [waypoints[0]]
        for i, num_points in enumerate(num_points_per_segment):
            for j in range(num_points):
                t = (j + 1) / (num_points + 1)
                interpolated_point = (1 - t) * waypoints[i] + t * waypoints[i+1]
                dense_waypoints.append(interpolated_point)
            dense_waypoints.append(waypoints[i+1])

        # Adjust number of waypoints to ensure it's of the desired form
        while len(dense_waypoints) % 4 != remainder:
            # For simplicity, remove the point closest to its neighbor
            min_dist_idx = np.argmin(np.linalg.norm(np.diff(dense_waypoints, axis=0), axis=-1)) + 1
            del dense_waypoints[min_dist_idx]

        return np.array(dense_waypoints)
    
    def load_planned_path_as_waypoints(self, long_seq_path, load_long_seq=True, \
                                use_canonicalization=True, return_scene_names=False, \
                                load_for_nav=False, start_waypoint=None):
        if load_long_seq:
            # selected_npy_path = self.get_long_planned_path_names() 
            selected_npy_path = long_seq_path 

        npy_data = np.load(selected_npy_path) # K X 3 (xyz, y represents the floor in Habitat) 
        npy_data = np.unique(npy_data, axis=0) # Remove redunant waypoints xy. 

        x_data = torch.from_numpy(npy_data[:, 0].copy()).float()[:, None] # K X 1
        y_data = -torch.from_numpy(npy_data[:, 2].copy()).float()[:, None] # K X 1
        z_data = torch.from_numpy(npy_data[:, 1].copy()).float()[:, None] # K X 1 

        xy_data = torch.cat((x_data, y_data), dim=-1) # K X 2  


        if load_for_nav:
            # For navigation sequence 
            # 1. Use previous interaction sequence's end pose to replace 1st waypoint. 
            # 2. Remove the last waypoint since next interaction sequence use it as first object com. 
            xy_data[0:1, :] = start_waypoint 
            dense_xy_data = self.sample_dense_waypoints(xy_data.detach().cpu().numpy(), distance_range=[0.7, 0.9], \
                        remainder=2)
            dense_xy_data = torch.from_numpy(dense_xy_data).float()
            dense_xy_data = dense_xy_data[:-1, :] # Need to ensure after removing the last value, the number of waypoints is 4*n+1. 
        
        else:
            # For interaction sequence
            # 1st interaction sequence: start waypoint, end waypoint should be repeated once. 
            # Since at the begining, need to approach the object. At the end, need to release the object. 
            dense_xy_data = self.sample_dense_waypoints(xy_data.detach().cpu().numpy(), distance_range=[0.6, 0.8], \
                        remainder=3)
            dense_xy_data = torch.from_numpy(dense_xy_data).float()
            dense_xy_data = torch.cat((dense_xy_data[0:1, :], dense_xy_data, dense_xy_data[-1:, :]), dim=0) 

        # new_x_data = torch.from_numpy(dense_xy_data[:, 0:1]).float()
        # new_y_data = torch.from_numpy(dense_xy_data[:, 1:2]).float() 

        new_x_data = dense_xy_data[:, 0:1]
        new_y_data = dense_xy_data[:, 1:2]
        # Note that in navigation, waypoints represent the root translaton xy.
        # In interaction, waypoints represent the object com xy. 
        # So during 

        # Tmp manually assign a floor height value
        new_z_data = z_data[0:1].repeat(new_x_data.shape[0], 1)
        new_z_data = torch.ones_like(new_z_data)
        new_z_data *= -1.6 
        new_z_data = new_z_data.to(z_data.device)  

        # Assign original target height to the end frame. 
        new_z_data[-1] = z_data[-1].clone() 

        planned_data = torch.cat((new_x_data, new_y_data, new_z_data), dim=-1) # T X 3  

        if use_canonicalization:
            cano_quat, cano_planned_data = self.canonizalize_planned_path(planned_data) # T X 3 
        else:
            cano_planned_data = planned_data

        if return_scene_names and use_canonicalization:
            return cano_quat, cano_planned_data, selected_npy_path
        elif return_scene_names: # not canonicalization 
            return cano_planned_data, selected_npy_path
        else:
            return cano_planned_data

    def gen_contact_label_for_long_seq(self, num_steps):
        # Option 1.
        # The first 1 second and last 1 second should not be in contact. The middle frame should be in contact. 
        # contact: 1, non-contact: 0. 
        start_num_frames = 30 
        end_num_frames = 30 

        contact_labels = torch.ones(num_steps)

        contact_labels[:start_num_frames] = 0
        contact_labels[-end_num_frames:] = 0

        # Option 2: 
        # Test whether the model can release at the middle. 
        # contact_labels[120:120]

        return contact_labels 

    def gen_language_for_long_seq(self, num_windows, text):
        text_list = []
        for w_idx in range(num_windows):
            text_list.append(text) 

        text_clip_feats_list = []
        for window_text in text_list:
            language_input = self.encode_text(window_text) # 1 X 512 
            text_clip_feats_list.append(language_input)

        return text_clip_feats_list  

    def get_object_mesh_from_prediction(self, all_res_list, data_dict, ds, \
            curr_window_ref_obj_rot_mat=None):
        num_seq = all_res_list.shape[0]

        pred_normalized_obj_trans = all_res_list[:, :, :3] # N X T X 3 
       
        if self.use_random_frame_bps:
            pred_obj_rel_rot_mat = all_res_list[:, :, 3:3+9].reshape(num_seq, -1, 3, 3) # N X T X 3 X 30
            if curr_window_ref_obj_rot_mat is not None:
                pred_obj_rot_mat = ds.rel_rot_to_seq(pred_obj_rel_rot_mat, curr_window_ref_obj_rot_mat)
            else:
                pred_obj_rot_mat = ds.rel_rot_to_seq(pred_obj_rel_rot_mat, data_dict['reference_obj_rot_mat']) 
                # ??? In some cases, this is not the first window!!! Bug? Ok for single-window generation. 
        else:
            pred_obj_rot_mat = all_res_list[:, :, 3:3+9].reshape(num_seq, -1, 3, 3) # N X T X 3 X 3

        pred_seq_com_pos = ds.de_normalize_obj_pos_min_max(pred_normalized_obj_trans)
        
        num_joints = 24
    
        normalized_global_jpos = all_res_list[:, :, 3+9:3+9+num_joints*3].reshape(num_seq, -1, num_joints, 3)
        global_jpos = ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, num_joints, 3))
        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3) # N X T X 22 X 3 
        global_root_jpos = global_jpos[:, :, 0, :].clone() # N X T X 3 

        global_rot_6d = all_res_list[:, :, 3+9+num_joints*3:3+9+num_joints*3+22*6].reshape(num_seq, -1, 22, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # N X T X 22 X 3 X 3 

        trans2joint = data_dict['trans2joint'].to(all_res_list.device) # N X 3

        seq_len = data_dict['seq_len'].detach().cpu().numpy() # BS 

        if trans2joint.shape[0] != all_res_list.shape[0]:
            trans2joint = trans2joint.repeat(num_seq, 1) 
            seq_len = seq_len.repeat(num_seq) 

        human_mesh_verts_list = []
        human_mesh_jnts_list = []
        object_mesh_verts_list = []
        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx] # T X 22 X 3 X 3 
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat) # T X 22 X 3 X 3 
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 22 X 3 
            
            curr_global_root_jpos = global_root_jpos[idx] # T X 3
     
            curr_trans2joint = trans2joint[idx:idx+1].clone()
            
            root_trans = curr_global_root_jpos + curr_trans2joint.to(curr_global_root_jpos.device) # T X 3 
         
            # Generate global joint position 
            bs = 1
            betas = data_dict['betas'][0]
            gender = data_dict['gender'][0]
            
            curr_gt_obj_rot_mat = data_dict['obj_rot_mat'][0]
            curr_gt_obj_trans = data_dict['obj_com_pos'][0]
            
            curr_obj_rot_mat = pred_obj_rot_mat[idx] # T X 3 X 3 
            curr_obj_quat = transforms.matrix_to_quaternion(curr_obj_rot_mat)
            curr_obj_rot_mat = transforms.quaternion_to_matrix(curr_obj_quat) # Potentially avoid some prediction not satisfying rotation matrix requirements.
            
            curr_obj_trans = pred_seq_com_pos[idx] # T X 3 
            # curr_obj_scale = data_dict['obj_scale'][idx]
            curr_seq_name = data_dict['seq_name'][0]
            # object_name = curr_seq_name.split("_")[1]
            object_name = data_dict['obj_name'][0]
          
            # Get human verts 
            mesh_jnts, mesh_verts, mesh_faces = \
                run_smplx_model(root_trans[None].cuda(), curr_local_rot_aa_rep[None].cuda(), \
                betas.cuda(), [gender], ds.bm_dict, return_joints24=True)

            # For generating all the vertices of the object 
            obj_rest_verts, obj_mesh_faces = ds.load_rest_pose_object_geometry(object_name) 
            obj_rest_verts = torch.from_numpy(obj_rest_verts).float().to(pred_seq_com_pos.device)

            obj_mesh_verts = ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat.cuda(), \
                        pred_seq_com_pos[idx], obj_rest_verts) # T X Nv X 3 

            # For generating object keypoints 
            # num_steps = pred_seq_com_pos[idx].shape[0]
            # rest_pose_obj_kpts = data_dict['rest_pose_obj_pts'].cuda()[0] # K X 3 
            # pred_seq_obj_kpts = torch.matmul(curr_obj_rot_mat[:, None, :, :].repeat(1, \
            #         rest_pose_obj_kpts.shape[0], 1, 1), \
            #         rest_pose_obj_kpts[None, :, :, None].repeat(num_steps, 1, 1, 1)) + \
            #         pred_seq_com_pos[idx][:, None, :, None] # T X K X 3 X 1  

            # pred_seq_obj_kpts = pred_seq_obj_kpts.squeeze(-1) # T X K X 3  

            human_mesh_verts_list.append(mesh_verts)
            human_mesh_jnts_list.append(mesh_jnts)

            object_mesh_verts_list.append(obj_mesh_verts)
            # object_mesh_verts_list.append(pred_seq_obj_kpts) 

        human_mesh_verts_list = torch.stack(human_mesh_verts_list)
        human_mesh_jnts_list = torch.stack(human_mesh_jnts_list)

        object_mesh_verts_list = torch.stack(object_mesh_verts_list)

        return human_mesh_verts_list, human_mesh_jnts_list, mesh_faces, object_mesh_verts_list, obj_mesh_faces 

    def rotation_matrix_from_two_vectors(self, vec1, vec2):
        # Find the rotation matrix that aligns vec1 to vec2 
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1-c) / (s**2))

        return rotation_matrix

    def apply_feet_floor_contact_guidance(self, pred_clean_x, rest_human_offsets, data_dict, \
        contact_labels=None, curr_window_ref_obj_rot_mat=None, \
        prev_window_cano_rot_mat=None, prev_window_init_root_trans=None): 
        # pred_clean_x: BS X T X D 
        # x_pose_cond: BS X T X D 
        # cond_mask: BS X T X D, 1 represents missing regions.  

        num_seq = pred_clean_x.shape[0]
        
        if self.test_unseen_objects:
            human_verts, human_jnts, human_faces, obj_verts, obj_faces = \
            self.get_object_mesh_from_prediction(pred_clean_x, data_dict, ds=self.unseen_seq_ds, \
            curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat) 
        else:
            human_verts, human_jnts, human_faces, obj_verts, obj_faces = \
            self.get_object_mesh_from_prediction(pred_clean_x, data_dict, ds=self.val_ds, \
            curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat) 
        # BS X 1 X T X Nv X 3, BS X 1 X T X 24 X 3, BS X T X Nv' X 3 

        left_toe_idx = 10
        right_toe_idx = 11 
        l_toe_height = human_jnts[:, 0, :, left_toe_idx, 2:] # BS X T X 1 
        r_toe_height = human_jnts[:, 0, :, right_toe_idx, 2:] # BS X T X 1   
        support_foot_height = torch.minimum(l_toe_height, r_toe_height)

        loss_feet_floor_contact = F.mse_loss(support_foot_height, torch.ones_like(support_foot_height)*0.02) 
        # print("Fett-Floor contact loss: {0}".format(loss_feet_floor_contact))

        loss = num_seq * loss_feet_floor_contact

        return loss 

    def compute_scene_penetration_score(self, converted_pred_jpos):
        # converted_pred_jpos: BS X (T*24) X 3, BS X N X 3  
        bs = converted_pred_jpos.shape[0]

        pred_jpos_in_scene = converted_pred_jpos + self.move_to_planned_path_in_scene # BS(1) X (T*24) X 3 

        # cano_quat: K X 4 
        cano_quat_for_human = transforms.quaternion_invert(self.cano_quat_in_scene[0:1].repeat(\
                        pred_jpos_in_scene.shape[1], 1)) # (T*24) X 4 
        pred_jpos_in_scene = transforms.quaternion_apply(cano_quat_for_human.to(pred_jpos_in_scene.device), \
                pred_jpos_in_scene[0])
        pred_jpos_in_scene = pred_jpos_in_scene[:, None, :] # (T*24) X Nv(1) X 3 
       
        signed_dists = compute_signed_distances(self.scene_sdf, self.scene_sdf_centroid, \
            self.scene_sdf_extents, pred_jpos_in_scene)

        scene_pen_score = torch.minimum(signed_dists, torch.zeros_like(signed_dists)).abs().mean() 

        return bs * scene_pen_score 

    def compute_vertex_normals(self, vertices, faces):
        """
        Compute vertex normals for a batch of 3D meshes.
        vertices: Tensor of shape (BS, T, N_o, 3) - Batch of 3D mesh vertices
        faces: Tensor of shape (Nf, 3) - Triangle definitions
        Returns: Tensor of shape (BS, T, N_o, 3) - Vertex normals
        """
        
        # Get the shape values
        BS, T, N_o, _ = vertices.shape

        # Compute the triangle vectors
        v1 = vertices[:, :, faces[:, 0], :]
        v2 = vertices[:, :, faces[:, 1], :]
        v3 = vertices[:, :, faces[:, 2], :]
        
        # Compute the two edge vectors of the triangles
        edge1 = v2 - v1
        edge2 = v3 - v1

        # Compute triangle normals using cross product
        triangle_normals = torch.cross(edge1, edge2)
        
        # Normalize triangle normals
        triangle_normals = triangle_normals / (triangle_normals.norm(dim=-1, keepdim=True) + 1e-8)

        # Initialize vertex normals tensor to zeros
        vertex_normals = torch.zeros_like(vertices)

        # Accumulate triangle normals to their vertices
        for i in range(3):
            vertex_normals.index_add_(2, faces[:, i], triangle_normals)

        # Normalize vertex normals
        vertex_normals = vertex_normals / (vertex_normals.norm(dim=-1, keepdim=True) + 1e-8)

        return vertex_normals

    def apply_hand_object_interaction_guidance_loss(self, pred_clean_x, rest_human_offsets, data_dict, \
        contact_labels=None, curr_window_ref_obj_rot_mat=None, \
        prev_window_cano_rot_mat=None, prev_window_init_root_trans=None):
        # prev_window_cano_rot_mat: BS X 3 X 3 
        # prev_window_init_root_trans: BS X 1 X 3 
        # pass 
        # pred_clean_x = torch.cat((data_dict['obj_motion'], data_dict['motion'], data_dict['feet_contact']), dim=-1).to(pred_clean_x.device)  

        num_seq = pred_clean_x.shape[0]

        if not self.use_long_planned_path:
            # SIngle window generation 
            pred_clean_x = pred_clean_x[:, :data_dict['seq_len'][0]]

        if self.test_unseen_objects:
            human_verts, human_jnts, human_faces, obj_verts, obj_faces = \
            self.get_object_mesh_from_prediction(pred_clean_x, data_dict, ds=self.unseen_seq_ds, \
            curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat) 
        else:
            human_verts, human_jnts, human_faces, obj_verts, obj_faces = \
            self.get_object_mesh_from_prediction(pred_clean_x, data_dict, ds=self.val_ds, \
            curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat) 
        # # BS X 1 X T X Nv X 3, BS X 1 X T X 24 X 3, BS X T X Nv' X 3 ]

        # Need to downsample object vertices sometimes. 
        num_obj_verts = obj_verts.shape[2]
        if num_obj_verts > 30000:
            downsample_rate = num_obj_verts//30000 + 1 
            obj_verts = obj_verts[:, :, ::downsample_rate, :] 

        # 1. Compute penetration loss between hand vertices and object vertices. 
        # hand_verts = human_verts.squeeze(1)[:, :, self.hand_vertex_idxs, :] # BS X T X N_hand X 3 
        pred_normalized_obj_trans = pred_clean_x[:, :, :3] # N X T X 3 

        if self.use_random_frame_bps:
            pred_obj_rel_rot_mat = pred_clean_x[:, :, 3:3+9].reshape(num_seq, -1, 3, 3) # N X T X 3 X 30
            if curr_window_ref_obj_rot_mat is not None:
                pred_obj_rot_mat = self.ds.rel_rot_to_seq(pred_obj_rel_rot_mat, curr_window_ref_obj_rot_mat)
            else:
                pred_obj_rot_mat = self.ds.rel_rot_to_seq(pred_obj_rel_rot_mat, \
                    data_dict['obj_rot_mat'].repeat(num_seq, 1, 1, 1)) # Bug? Since for the windows except the first one, the reference obj mat is not the originbal one in data? 
        else:
            pred_obj_rot_mat = pred_clean_x[:, :, 3:3+9].reshape(num_seq, -1, 3, 3) # N X T X 3 X 3

        pred_seq_com_pos = self.ds.de_normalize_obj_pos_min_max(pred_normalized_obj_trans) # N X T X 3

        num_steps = pred_clean_x.shape[1] 

        # 2. Compute contact loss, minimize the distance between hand vertices and nearest neugbor points on the object mesh. 
        l_palm_idx = 22 
        r_palm_idx = 23 
        left_palm_jpos = human_jnts.squeeze(1)[:, :, l_palm_idx, :] # BS X T X 3 
        right_palm_jpos = human_jnts.squeeze(1)[:, :, r_palm_idx, :] # BS X T X 3
        
        contact_points = torch.cat((left_palm_jpos[:, :, None, :], \
                    right_palm_jpos[:, :, None, :]), dim=2) # BS X T X 2 X 3
        bs, seq_len, _, _ = contact_points.shape  
      
        # print("Object # vertices:{0}".format(obj_verts.shape)) 
        dists = torch.cdist(contact_points.reshape(bs*seq_len, 2, 3)[:, :, :], \
                    obj_verts.reshape(bs*seq_len, -1, 3)) # (BS*T) X 2 X N_object 
        dists, _ = torch.min(dists, 2) # (BS*T) X 2

        pred_contact_semantic = pred_clean_x[:, :, -4:-2] # BS X T X 2
        contact_labels = pred_contact_semantic > 0.95 

        contact_labels = contact_labels.reshape(bs*seq_len, -1)[:, :2].detach() # (BS*T) X 2

        zero_target = torch.zeros_like(dists).to(dists.device)
        contact_threshold = 0.02 

        loss_contact = F.l1_loss(torch.maximum(dists*contact_labels[:, :2]-contact_threshold, zero_target), \
                zero_target) 
       
        # Compute temporal consistency loss. 
        left_palm_to_obj_com = left_palm_jpos - pred_seq_com_pos.detach() # BS X T X 3 
        right_palm_to_obj_com = right_palm_jpos - pred_seq_com_pos.detach() 
        relative_left_palm_jpos = torch.matmul(pred_obj_rot_mat.detach().transpose(2, 3), \
                        left_palm_to_obj_com[:, :, :, None]).squeeze(-1) # BS X T X 3 
        relative_right_palm_jpos = torch.matmul(pred_obj_rot_mat.detach().transpose(2, 3), \
                        right_palm_to_obj_com[:, :, :, None]).squeeze(-1)  

        contact_labels = contact_labels.reshape(num_seq, num_steps, -1) # BS X T X 2 
       
        # Expand dimensions of contact_labels for multiplication
        left_contact_labels_expanded = contact_labels[:, :, 0:1]
        left_contact_mask = left_contact_labels_expanded * left_contact_labels_expanded.transpose(-1, -2)

        right_contact_labels_expanded = contact_labels[:, :, 1:2]
        right_contact_mask = right_contact_labels_expanded * right_contact_labels_expanded.transpose(-1, -2) # BS X T X T 
        
        left_norms = torch.norm(relative_left_palm_jpos, dim=-1, keepdim=True)
        left_normalized = relative_left_palm_jpos / left_norms
        left_similarity = torch.matmul(left_normalized, left_normalized.transpose(-1, -2))

        right_norms = torch.norm(relative_right_palm_jpos, dim=-1, keepdim=True)
        right_normalized = relative_right_palm_jpos / right_norms
        right_similarity = torch.matmul(right_normalized, right_normalized.transpose(-1, -2)) # BS X T X T 

        loss_consistency = 1 - torch.mean(left_similarity * left_contact_mask) + \
                    1 - torch.mean(right_similarity * right_contact_mask) # GT: 0.11 
      
        # Add floor-object penetration loss 
        loss_floor_object = torch.minimum(obj_verts[:, :, :, -1], \
                    torch.zeros_like(obj_verts[:, :, :, -1])).abs().mean()

        loss = bs * (loss_contact + loss_consistency + loss_floor_object * 100)

        return loss

    def apply_different_guidance_loss(self, noise_level, pred_clean_x, x_pose_cond, cond_mask, \
        rest_human_offsets, data_dict, \
        contact_labels=None, curr_window_ref_obj_rot_mat=None, \
        prev_window_cano_rot_mat=None, prev_window_init_root_trans=None):
        # Combine all the guidance we need during denoising step. 

        # Feet (the one that is supporting each frame) and floor should be in contact. 
        loss_feet_floor_contact = self.apply_feet_floor_contact_guidance(pred_clean_x, rest_human_offsets, data_dict, \
                        contact_labels=contact_labels, curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat, \
                        prev_window_cano_rot_mat=prev_window_cano_rot_mat, \
                        prev_window_init_root_trans=prev_window_init_root_trans)

        # Hand and object should be in contact, not penetrating, temporal consistent.
        loss_hand_object_interaction = self.apply_hand_object_interaction_guidance_loss(pred_clean_x, rest_human_offsets, data_dict, \
                        contact_labels=contact_labels, curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat, \
                        prev_window_cano_rot_mat=prev_window_cano_rot_mat, \
                        prev_window_init_root_trans=prev_window_init_root_trans)

        loss = loss_hand_object_interaction * 10000 + loss_feet_floor_contact * 100000 * 3

        return loss

    def prep_evaluation_metrics_list_for_long_seq(self):
        
        self.foot_sliding_jnts_list_long_seq = [] 

        self.floor_height_list_long_seq = [] 

        self.contact_percent_list_long_seq = []

        self.start_obj_com_pos_err_list_long_seq = [] 
        self.end_obj_com_pos_err_list_long_seq = [] 
        self.waypoints_xy_pos_err_list_long_seq = []

        self.penetration_list_long_seq = [] 
        self.hand_penetration_list_long_seq = [] 

        self.scene_human_penetration_list_long_seq = []
        self.scene_object_penetration_list_long_seq = [] 

    def append_new_value_to_metrics_list_for_long_seq(self, foot_sliding_jnts, \
            floor_height, contact_percent, start_obj_com_pos_err, end_obj_com_pos_err, \
            waypoints_xy_pos_err, penetration_score, hand_penetration_score, \
            scene_human_penetration, scene_object_penetration): 
        # Append new sequence's value to list. 
        self.foot_sliding_jnts_list_long_seq.append(foot_sliding_jnts)

        self.floor_height_list_long_seq.append(floor_height)

        self.contact_percent_list_long_seq.append(contact_percent)
        
        self.start_obj_com_pos_err_list_long_seq.append(start_obj_com_pos_err)
        self.end_obj_com_pos_err_list_long_seq.append(end_obj_com_pos_err)
        self.waypoints_xy_pos_err_list_long_seq.append(waypoints_xy_pos_err)

        self.penetration_list_long_seq.append(penetration_score) 
        self.hand_penetration_list_long_seq.append(hand_penetration_score) 

        self.scene_human_penetration_list_long_seq.append(scene_human_penetration)
        self.scene_object_penetration_list_long_seq.append(scene_object_penetration) 

    def cond_sample_res_w_long_planned_path(self):
        if self.opt.pretrained_model == "":
            weights = os.listdir(self.results_folder)
            weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
            weight_path = max(weights_paths, key=os.path.getctime)
    
            print(f"Loaded weight: {weight_path}")

            milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
            # milestone = "9"

            self.load(milestone)
        else:
            milestone = "10"
            self.load(milestone, pretrained_path=self.opt.pretrained_model) 

        self.ema.ema_model.eval()
        
        if self.test_unseen_objects:
            test_loader = torch.utils.data.DataLoader(
                self.unseen_seq_ds, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False) 
        else:
            test_loader = torch.utils.data.DataLoader(
                self.whole_seq_ds, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False) 
            
        overlap_frame_num = 10

        self.prep_evaluation_metrics_list_for_long_seq()

        dest_res_for_eval_npz_folder, dest_metric_folder, dest_out_vis_folder, \
        dest_out_gt_vis_folder, dest_out_obj_folder, dest_out_text_json_folder = self.prep_res_folders() 

        object_test_seq_dict = self.get_long_planned_path_names_new() 
        if self.test_unseen_objects:
            object_test_seq_dict = self.process_long_path_for_unseen_objects(object_test_seq_dict)

        # Load scene floor height
        scene_floor_json_path = os.path.join(self.data_root_folder, "replica_processed/scene_floor_height.json")
        scene_floor_h_dict = json.load(open(scene_floor_json_path, 'r'))
        # scene_floor_h_dict = {} 
        # scene_floor_h_dict['frl_apartment_0'] = -1.58 
        # # scene_floor_h_dict['frl_apartment_1'] = -1.48 
        # scene_floor_h_dict['frl_apartment_4'] = -1.5 
        # # scene_floor_h_dict['apartment_1'] = -1.42


        # scene_floor_h_dict['frl_apartment_0'] = 0  
        # scene_floor_h_dict['frl_apartment_1'] = -1.48 
        # scene_floor_h_dict['frl_apartment_4'] = 0 
        # scene_floor_h_dict['apartment_1'] = -1.42

        ori_object_com_height_dict = {
            "largetable": 0.35, 
            "smalltable": 0.26, 
            "woodchair": 0.44, 
            "floorlamp": 0.9,
            "largebox": 0.155, 
            "smallbox": 0.06, 
            "whitechair": 0.46,
            "plasticbox": 0.13,
            "suitcase": 0.32,
            "trashcan": 0.15,
        }

        tmp_selected_seq_name_list = ["frl_apartment_4_sub16_clothesstand_004_clothesstand_pidx_1_sample_cnt_0", \
            "frl_apartment_4_sub16_largetable_006_largetable_pidx_0_sample_cnt_0", \
            "frl_apartment_4_sub16_largetable_006_largetable_pidx_2_sample_cnt_0", \
            "frl_apartment_4_sub16_largetable_006_largetable_pidx_6_sample_cnt_0", \
            "frl_apartment_4_sub16_trashcan_000_trashcan_pidx_0_sample_cnt_0", \
            "frl_apartment_4_sub16_trashcan_000_trashcan_pidx_13_sample_cnt_0", \
            "frl_apartment_4_sub16_trashcan_000_trashcan_pidx_2_sample_cnt_0", \
            "frl_apartment_4_sub16_trashcan_000_trashcan_pidx_5_sample_cnt_0", \
            "frl_apartment_4_sub16_trashcan_000_trashcan_pidx_9_sample_cnt_0", \
            "frl_apartment_4_sub17_floorlamp_001_floorlamp_pidx_1_sample_cnt_0", \
            "frl_apartment_4_sub17_floorlamp_001_floorlamp_pidx_4_sample_cnt_0", \
            "frl_apartment_4_sub17_floorlamp_001_floorlamp_pidx_5_sample_cnt_0", \
            "frl_apartment_4_sub17_smallbox_018_smallbox_pidx_10_sample_cnt_0", \
            "frl_apartment_4_sub17_smallbox_018_smallbox_pidx_11_sample_cnt_0", \
            "frl_apartment_4_sub17_smallbox_018_smallbox_pidx_8_sample_cnt_0"]

        # with torch.no_grad():
        for s_idx, val_data_dict in enumerate(test_loader):

            seq_name_list = val_data_dict['seq_name']
            object_name_list = val_data_dict['obj_name'] 

            if object_name_list[0] not in object_test_seq_dict:
                continue 

            # planned_paths_list, text_list, end_frame_height_range_list = \
            #         self.get_long_planned_path_names() 
            planned_paths_list = object_test_seq_dict[object_name_list[0]]['npy_list']
            text_list = object_test_seq_dict[object_name_list[0]]['text_list']
            end_frame_height_range_list = object_test_seq_dict[object_name_list[0]]['height_range']

            num_planned_path = len(planned_paths_list)

            for p_idx in range(num_planned_path):

                video_paths = [] 

                curr_seq_name_tag = self.test_scene_name + "_" + seq_name_list[0] + \
                "_" + object_name_list[0]+ "_pidx_" + str(p_idx) + "_sample_cnt_0"

                if ("lift" not in text_list[p_idx]) and ("Lift" not in text_list[p_idx]):
                    continue 

                # Use distance heuristics to determine the waypoints at frame 30, 60, 90. 
                cano_quat, planned_obj_path, planned_scene_names = \
                        self.load_planned_path_as_waypoints_new(planned_paths_list[p_idx], \
                        use_canonicalization=True, return_scene_names=True) 

                # planned_obj_path: K X 3
                # To convert the planned path back to the original one, need to apply inverse(cano_quat). 

                rest_human_offsets = val_data_dict['rest_human_offsets'].cuda() # BS X 24 X 3 

                # planned_path_floor_height = planned_obj_path[0, -1] # In visualization, put the interaction from floor z = 0 to this value. 
                planned_path_floor_height = scene_floor_h_dict[self.test_scene_name] 

                tmp_text_idx = planned_scene_names.split("/")[-2]
                if int(tmp_text_idx) in [9, 10, 13]: # sample from a surface, this is accurate. 
                    planned_obj_path[:-2, 2] = planned_path_floor_height 
                else:
                    planned_obj_path[:, 2] = planned_path_floor_height 

                start_obj_pos_on_planned_path = planned_obj_path[0:1, :] # 1 X 3 
                waypoints2start_trans = planned_obj_path[1:, :] - planned_obj_path[0:1, :] # (K-1) X 3 
                end2start_trans = planned_obj_path[-1:, :] - planned_obj_path[0:1, :] # 1 X 3 

                val_human_data = val_data_dict['motion'].cuda() 
                val_normalized_obj_data = val_data_dict['obj_motion'].cuda() # Only need the first frame. 
                val_ori_obj_data = val_data_dict['ori_obj_motion'].cuda() # BS X T X (3+9)

                start_obj_com_pos = val_ori_obj_data[:, 0:1, :3] # BS X 1 X 3 
                move2aligned_planned_path = start_obj_pos_on_planned_path[None].to(start_obj_com_pos.device) - \
                        start_obj_com_pos # BS X 1 X 3 
                move2aligned_planned_path[:, :, 2] = planned_path_floor_height 

                self.move_to_planned_path_in_scene = move2aligned_planned_path.clone() 
                self.cano_quat_in_scene = cano_quat.clone()  

                end_obj_com_pos = start_obj_com_pos + end2start_trans[None, :, :].cuda() # BS X 1 X 3

                seq_obj_com_pos = torch.zeros(start_obj_com_pos.shape[0], (planned_obj_path.shape[0]-1)*30, 3).cuda() 
                seq_obj_com_pos[:, 0:1, :] = start_obj_com_pos.clone() 
                
                waypoints_com_pos = start_obj_com_pos + waypoints2start_trans[None, :, :].cuda() # BS X (K-1) X 3

                waypoints_com_pos_for_vis = waypoints_com_pos.clone()

                num_pts = waypoints_com_pos.shape[1]
                
                window_cnt = 0
                for tmp_p_idx in range(num_pts):
                    if (tmp_p_idx+1)*30 % self.window == 0: # The last 30 frame in each window 
                        t_idx = (tmp_p_idx+1)*30-1-overlap_frame_num*window_cnt
                        seq_obj_com_pos[:, t_idx, :2] = waypoints_com_pos[:, tmp_p_idx, :2]
                    
                        # Tmp set a height for the sequence: lift, carry and walk, put down. 
                        if tmp_p_idx == num_pts - 1 or tmp_p_idx == num_pts - 2:
                            seq_obj_com_pos[:, t_idx, 2] = waypoints_com_pos[:, tmp_p_idx, 2] # Put down on the floor. 
                            # seq_obj_com_pos[:, t_idx, 2] = seq_obj_com_pos[:, 0, 2].clone() 
                        else:
                            # seq_obj_com_pos[:, t_idx, 2] = waypoints_com_pos[:, 0, 2] + 0.05 # For floorlamp case. 
                            # seq_obj_com_pos[:, t_idx, 2] = 1.2 # For lift, move and put down. tried[0.9, 1.2]
                            # seq_obj_com_pos[:, t_idx, 2] =  waypoints_com_pos[:, p_idx, 2] # For pushing all the way. 
                            if self.test_unseen_objects:
                                curr_obj_type_name = seq_name_list[0].split("_")[1]
                                ori_obj_com_height = ori_object_com_height_dict[curr_obj_type_name]
                                end_frame_height_offset = random.uniform(end_frame_height_range_list[p_idx][0], \
                                                    end_frame_height_range_list[p_idx][1]) - ori_obj_com_height 
                                seq_obj_com_pos[:, t_idx, 2] = start_obj_com_pos[:, 0, 2] + end_frame_height_offset 
                            else:
                                seq_obj_com_pos[:, t_idx, 2] = random.uniform(end_frame_height_range_list[p_idx][0], \
                                                    end_frame_height_range_list[p_idx][1])

                            waypoints_com_pos_for_vis[:, tmp_p_idx, 2] = seq_obj_com_pos[:, t_idx, 2].clone() 

                        window_cnt += 1 
                    else:
                        t_idx = (tmp_p_idx+1)*30-1-overlap_frame_num*window_cnt
                        seq_obj_com_pos[:, t_idx, :2] = waypoints_com_pos[:, tmp_p_idx, :2]

                actual_num_frames = window_cnt * self.window - (window_cnt-1) * overlap_frame_num
                seq_obj_com_pos = seq_obj_com_pos[:, :actual_num_frames] 
       
                if self.add_language_condition:
                    text_clip_feats_list = self.gen_language_for_long_seq(window_cnt, text_list[p_idx])

                seq_obj_com_pos = self.val_ds.normalize_obj_pos_min_max(seq_obj_com_pos) # BS X T X 3 
                val_obj_data = torch.cat((seq_obj_com_pos, torch.zeros(seq_obj_com_pos.shape[0], \
                            seq_obj_com_pos.shape[1], 9).to(seq_obj_com_pos.device)), dim=-1) # BS X T X (3+9) 
                # Reaplce the first frame's object rotation. 
                val_obj_data[:, 0:1, 3:] = val_normalized_obj_data[:, 0:1, 3:] 

                obj_bps_data = val_data_dict['input_obj_bps'].cuda().reshape(-1, 1, 1024*3)
             
                ori_data_cond = obj_bps_data 

                # Manually define contact labels, the heuristic is that the start and end frames are not in contact,
                # the middle frames are in contact. 
                contact_labels = self.gen_contact_label_for_long_seq(actual_num_frames) # T
                contact_labels = contact_labels[None].repeat(seq_obj_com_pos.shape[0], 1).to(seq_obj_com_pos.device) # BS X T  
             
                # Generate padding mask 
                actual_seq_len = torch.ones(val_obj_data.shape[0], self.window+1) * self.window + 1
                tmp_mask = torch.arange(self.window+1).expand(val_human_data.shape[0], \
                        self.window+1) < actual_seq_len
                        # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].cuda() # 1 X 1 X 121 

                # padding_mask = None 

                bs_window_len = torch.zeros(val_obj_data.shape[0])
                bs_window_len[:] = self.window 
               
                end_pos_cond_mask = self.prep_start_end_condition_mask_pos_only(val_obj_data[:, :self.window], bs_window_len)
                cond_mask = self.prep_mimic_A_star_path_condition_mask_pos_xy_only(val_obj_data[:, :self.window], bs_window_len)
                cond_mask = end_pos_cond_mask * cond_mask 
               
                # human_cond_mask = torch.ones_like(val_human_data).to(val_human_data.device)
                human_cond_mask = torch.ones(cond_mask.shape[0], cond_mask.shape[1], val_human_data.shape[-1]).to(val_human_data.device)
                if self.input_first_human_pose:
                    human_cond_mask[:, 0, :] = 0 
                cond_mask = torch.cat((cond_mask, human_cond_mask), dim=-1) # BS X T X (3+6+24*3+22*6)

                if self.use_guidance_in_denoising:
                    # Actually used. 
                    self.object_sdf, self.object_sdf_centroid, self.object_sdf_extents = \
                    self.load_object_sdf_data(val_data_dict['obj_name'][0])
                    guidance_fn = self.apply_different_guidance_loss
                else:
                    guidance_fn = None 

                num_samples_per_seq = 1
                for sample_idx in range(num_samples_per_seq):
                    # data = torch.cat((val_obj_data, val_human_data), dim=-1)
                    tmp_val_human_data = torch.cat((val_human_data[:, 0:1, :], torch.zeros(val_human_data.shape[0], \
                                val_obj_data.shape[1]-1, val_human_data.shape[-1]).to(val_obj_data.device)), dim=1)
                    # data = torch.cat((val_obj_data, torch.zeros(val_human_data.shape[0], \
                                # val_obj_data.shape[1], val_human_data.shape[-1]).to(val_obj_data.device)), dim=-1)
                

                    if self.use_object_keypoints:
                        contact_data = torch.zeros(val_obj_data.shape[0], \
                                val_obj_data.shape[1], 4).to(val_obj_data.device) 
                        data = torch.cat((val_obj_data, tmp_val_human_data, contact_data), dim=-1) 
                        cond_mask = torch.cat((cond_mask, \
                                torch.ones(cond_mask.shape[0], cond_mask.shape[1], \
                                4).to(cond_mask.device)), dim=-1) 
                    else:
                        data = torch.cat((val_obj_data, tmp_val_human_data), dim=-1)


                    if self.add_language_condition: # Not ready yet. 
                        if self.test_unseen_objects:
                            input_ds = self.unseen_seq_ds
                        else:
                            input_ds = self.ds 
                        all_res_list = self.ema.ema_model.sample_sliding_window_w_canonical(input_ds, \
                            val_data_dict['obj_name'], val_data_dict['trans2joint'], \
                            data, ori_data_cond, cond_mask, padding_mask, overlap_frame_num, \
                            input_waypoints=True, language_input=text_clip_feats_list, \
                            contact_labels=contact_labels, \
                            rest_human_offsets=rest_human_offsets, guidance_fn=guidance_fn, \
                            data_dict=val_data_dict)
                    
                    # vis_tag = str(milestone)+"_final_long_seq_w_planned_waypoints_"+"_sidx_"+str(s_idx)+"_sample_cnt_"+str(sample_idx)
                    
                    vis_tag = str(milestone)+"_"+self.test_scene_name+"_sidx_"+str(s_idx)+"_long_seq_"+"_pidx_"+str(p_idx)+"_sample_cnt_"+str(sample_idx)
                    if self.test_on_train:
                        vis_tag = vis_tag + "_on_train"

                    if self.use_guidance_in_denoising:
                        vis_tag = vis_tag + "_all_guidance"

                    if self.test_unseen_objects:
                        vis_tag = vis_tag + "_unseen_object"

                    if self.use_object_keypoints:
                        all_res_list = all_res_list[:, :, :-4]


                    curr_seq_name_tag = self.test_scene_name + "_" + seq_name_list[0] + "_" + object_name_list[0]+ "_pidx_" + str(p_idx) + "_sample_cnt_" + str(sample_idx)

                    dest_text_json_path = os.path.join(dest_out_text_json_folder, curr_seq_name_tag+".json")
                    dest_text_json_dict = {}
                    dest_text_json_dict['text'] = text_list[p_idx]
                    if not os.path.exists(dest_text_json_path):
                        json.dump(dest_text_json_dict, open(dest_text_json_path, 'w'))

                    curr_dest_out_mesh_folder = os.path.join(dest_out_obj_folder, curr_seq_name_tag)
                    curr_dest_out_mesh_topview_folder = os.path.join(dest_out_obj_folder, curr_seq_name_tag+"_topview")

                    curr_dest_out_vid_path = os.path.join(dest_out_vis_folder, curr_seq_name_tag+".mp4")
                    curr_dest_out_vid_topview_path = os.path.join(dest_out_vis_folder, curr_seq_name_tag+"_topview.mp4")

                    # For visualization on 3D scene. 
                    if not self.compute_metrics:
                        self.gen_vis_res_generic(all_res_list, val_data_dict, milestone, cond_mask, \
                                    vis_tag=vis_tag, planned_end_obj_com=end_obj_com_pos+move2aligned_planned_path, \
                                    move_to_planned_path=move2aligned_planned_path, \
                                    planned_waypoints_pos=waypoints_com_pos+move2aligned_planned_path, \
                                    planned_scene_names=planned_scene_names, \
                                    planned_path_floor_height=planned_path_floor_height, \
                                    cano_quat=cano_quat, dest_out_vid_path=curr_dest_out_vid_topview_path, \
                                    dest_mesh_vis_folder=curr_dest_out_mesh_topview_folder, save_obj_only=True) 

                    # For visualization on empty floor. 
                    pred_human_verts_list, pred_human_jnts_list, pred_human_trans_list, pred_human_rot_list, \
                    pred_obj_com_pos_list, pred_obj_rot_mat_list, pred_obj_verts_list, \
                    _, _, _ = self.gen_vis_res_generic(all_res_list, val_data_dict, milestone, cond_mask, \
                            vis_tag=vis_tag, planned_end_obj_com=end_obj_com_pos, \
                            planned_waypoints_pos=waypoints_com_pos_for_vis, \
                            vis_wo_scene=True, gen_long_seq=True, dest_out_vid_path=curr_dest_out_vid_path, \
                            dest_mesh_vis_folder=curr_dest_out_mesh_folder, save_obj_only=False)  

                    video_paths.append(curr_dest_out_vid_path)

                    mesh_save_folders_str = "&".join([curr_dest_out_mesh_folder])
                    # initial_obj_paths = "&".join(initial_obj_paths)
                    use_guidance_str = "1" if self.use_guidance_in_denoising else "0"
                    interaction_epoch = milestone
                    video_save_dir_name = os.path.join(self.save_res_folder, "long_seq_res_videos")
                    if not os.path.exists(video_save_dir_name):
                        os.makedirs(video_save_dir_name) 
                    # video_save_dir_name = os.path.join("visualizer_results", opt.vis_wdir)

                    ori_seq_obj_com_pos = self.val_ds.de_normalize_obj_pos_min_max(seq_obj_com_pos)
                    foot_sliding_jnts, floor_height, contact_percent, \
                    start_obj_com_pos_err, end_obj_com_pos_err, waypoints_xy_pos_err = \
                            compute_metrics_long_seq(pred_human_jnts_list[0], \
                            pred_obj_com_pos_list[0], pred_obj_rot_mat_list[0], \
                            pred_obj_verts_list[0], \
                            ori_seq_obj_com_pos[0], cond_mask)

                    pred_hand_penetration_score = self.compute_hand_penetration_metric(object_name_list[0], \
                                    pred_human_verts_list[0], \
                                    pred_obj_com_pos_list[0], pred_obj_rot_mat_list[0])
               
                    pred_penetration_score = self.compute_hand_penetration_metric(object_name_list[0], \
                                        pred_human_verts_list[0], \
                                        pred_obj_com_pos_list[0], pred_obj_rot_mat_list[0], \
                                        eval_fullbody=True)

                    # (Pdb) pred_human_verts_list[0].shape
                    # torch.Size([230, 10475, 3])
                    # pred_obj_verts_list[0].shape  T X Nv X 3 
                    pred_human_verts_sampled = pred_human_verts_list[0][:, ::10, :].reshape(-1, 3)[None] # 1 X (T*Nv) X 3 
                    pred_obj_verts_sampled = pred_obj_verts_list[0].reshape(-1, 3)[None] # 1 X (T*No) X 3 
                    scene_human_penetration_score = self.compute_scene_penetration_score(pred_human_verts_sampled.detach()).detach().cpu().numpy() 
                    scene_object_penetration_score = self.compute_scene_penetration_score(pred_obj_verts_sampled.detach()).detach().cpu().numpy()

                    self.append_new_value_to_metrics_list_for_long_seq(foot_sliding_jnts, \
                        floor_height, contact_percent, \
                        start_obj_com_pos_err, end_obj_com_pos_err, waypoints_xy_pos_err, \
                        pred_penetration_score, pred_hand_penetration_score, \
                        scene_human_penetration_score, scene_object_penetration_score) 

                    # curr_seq_name_tag = seq_name_list[0] + "_" + object_name_list[0] + "_sample_cnt_" + str(sample_idx)
                    print("Current Sequence name:{0}".format(curr_seq_name_tag))
                    self.print_evaluation_metrics_for_long_seq([foot_sliding_jnts], \
                    [floor_height], [contact_percent], \
                    [start_obj_com_pos_err], [end_obj_com_pos_err], [waypoints_xy_pos_err], \
                    [pred_penetration_score], [pred_hand_penetration_score], \
                    [scene_human_penetration_score], [scene_object_penetration_score], \
                    dest_metric_folder, curr_seq_name_tag)

        self.print_evaluation_metrics_for_long_seq(self.foot_sliding_jnts_list_long_seq, \
                self.floor_height_list_long_seq, \
                self.contact_percent_list_long_seq, \
                self.start_obj_com_pos_err_list_long_seq, \
                self.end_obj_com_pos_err_list_long_seq, \
                self.waypoints_xy_pos_err_list_long_seq, \
                self.penetration_list_long_seq, self.hand_penetration_list_long_seq, \
                self.scene_human_penetration_list_long_seq, self.scene_object_penetration_list_long_seq, \
                dest_metric_folder)  

    def create_ball_mesh(self, center_pos, ball_mesh_path):
        # center_pos: K X 3  
        ball_color = np.asarray([22, 173, 100]) # green 

        num_mesh = center_pos.shape[0]
        for idx in range(num_mesh):
            ball_mesh = trimesh.primitives.Sphere(radius=0.05, center=center_pos[idx])
            
            dest_ball_mesh = trimesh.Trimesh(
                vertices=ball_mesh.vertices,
                faces=ball_mesh.faces,
                vertex_colors=ball_color,
                process=False)

            result = trimesh.exchange.ply.export_ply(dest_ball_mesh, encoding='ascii')
            output_file = open(ball_mesh_path.replace(".ply", "_"+str(idx)+".ply"), "wb+")
            output_file.write(result)
            output_file.close()
    
    def export_to_mesh(self, mesh_verts, mesh_faces, mesh_path):
        dest_mesh = trimesh.Trimesh(
            vertices=mesh_verts,
            faces=mesh_faces,
            process=False)

        result = trimesh.exchange.ply.export_ply(dest_mesh, encoding='ascii')
        output_file = open(mesh_path, "wb+")
        output_file.write(result)
        output_file.close()

    def plot_arr(self, t_vec, pred_val, gt_val, dest_path):
        plt.plot(t_vec, gt_val, color='green', label="gt")
        plt.plot(t_vec, pred_val, color='red', label="pred")
        plt.legend(["gt", "pred"])
        plt.savefig(dest_path)
        plt.clf()
    
    def gen_vis_res_generic(self, all_res_list, data_dict, step, cond_mask, vis_gt=False, vis_tag=None, \
                planned_end_obj_com=None, move_to_planned_path=None, planned_waypoints_pos=None, \
                vis_long_seq=False, overlap_frame_num=10, planned_scene_names=None, \
                planned_path_floor_height=None, vis_wo_scene=False, text_anno=None, cano_quat=None, \
                gen_long_seq=False, curr_object_name=None, dest_out_vid_path=None, dest_mesh_vis_folder=None, \
                save_obj_only=False):

        # Prepare list used for evaluation. 
        human_jnts_list = []
        human_verts_list = [] 
        obj_verts_list = [] 
        trans_list = []
        human_mesh_faces_list = []
        obj_mesh_faces_list = [] 

        # all_res_list: N X T X (3+9) 
        num_seq = all_res_list.shape[0]

        pred_normalized_obj_trans = all_res_list[:, :, :3] # N X T X 3 
        pred_seq_com_pos = self.ds.de_normalize_obj_pos_min_max(pred_normalized_obj_trans)

        if self.use_random_frame_bps:
            reference_obj_rot_mat = data_dict['reference_obj_rot_mat'] # N X 1 X 3 X 3 

            pred_obj_rel_rot_mat = all_res_list[:, :, 3:3+9].reshape(num_seq, -1, 3, 3) # N X T X 3 X 3
            pred_obj_rot_mat = self.ds.rel_rot_to_seq(pred_obj_rel_rot_mat, reference_obj_rot_mat)

        num_joints = 24
    
        normalized_global_jpos = all_res_list[:, :, 3+9:3+9+num_joints*3].reshape(num_seq, -1, num_joints, 3)
        global_jpos = self.ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, num_joints, 3))
        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3) # N X T X 22 X 3 

        # For putting human into 3D scene 
        if move_to_planned_path is not None:
            pred_seq_com_pos = pred_seq_com_pos + move_to_planned_path
            global_jpos = global_jpos + move_to_planned_path[:, :, None, :]

        global_root_jpos = global_jpos[:, :, 0, :].clone() # N X T X 3 

        global_rot_6d = all_res_list[:, :, 3+9+24*3:3+9+24*3+22*6].reshape(num_seq, -1, 22, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # N X T X 22 X 3 X 3 

        trans2joint = data_dict['trans2joint'].to(all_res_list.device).squeeze(1) # BS X  3 
        seq_len = data_dict['seq_len'] # BS, should only be used during for single window generation. 
        if all_res_list.shape[0] != trans2joint.shape[0]:
            trans2joint = trans2joint.repeat(num_seq, 1, 1) # N X 24 X 3 
            seq_len = seq_len.repeat(num_seq) # N 
        seq_len = seq_len.detach().cpu().numpy() # N 

        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx] # T X 22 X 3 X 3 
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat) # T X 22 X 3 X 3 
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 22 X 3 
            
            curr_global_root_jpos = global_root_jpos[idx] # T X 3
     
            curr_trans2joint = trans2joint[idx:idx+1].clone() # 1 X 3 
            
            root_trans = curr_global_root_jpos + curr_trans2joint.to(curr_global_root_jpos.device) # T X 3 

            # Generate global joint position 
            bs = 1
            betas = data_dict['betas'][0]
            gender = data_dict['gender'][0]
            
            curr_gt_obj_rot_mat = data_dict['obj_rot_mat'][0] # T X 3 X 3
            curr_gt_obj_com_pos = data_dict['obj_com_pos'][0] # T X 3 
            
            curr_obj_rot_mat = pred_obj_rot_mat[idx] # T X 3 X 3 
            curr_obj_quat = transforms.matrix_to_quaternion(curr_obj_rot_mat)
            curr_obj_rot_mat = transforms.quaternion_to_matrix(curr_obj_quat) # Potentially avoid some prediction not satisfying rotation matrix requirements.

            if curr_object_name is not None: 
                object_name = curr_object_name 
            else:
                curr_seq_name = data_dict['seq_name'][0]
                object_name = data_dict['obj_name'][0]
          
            # Get human verts 
            mesh_jnts, mesh_verts, mesh_faces = \
                run_smplx_model(root_trans[None].cuda(), curr_local_rot_aa_rep[None].cuda(), \
                betas.cuda(), [gender], self.ds.bm_dict, return_joints24=True)

            if self.test_unseen_objects:
                # Get object verts 
                obj_rest_verts, obj_mesh_faces = self.unseen_seq_ds.load_rest_pose_object_geometry(object_name)
                obj_rest_verts = torch.from_numpy(obj_rest_verts)

                gt_obj_mesh_verts = self.unseen_seq_ds.load_object_geometry_w_rest_geo(curr_gt_obj_rot_mat, \
                            curr_gt_obj_com_pos, obj_rest_verts.float())

                obj_mesh_verts = self.unseen_seq_ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat, \
                            pred_seq_com_pos[idx], obj_rest_verts.float().to(pred_seq_com_pos.device))
            else:
                # Get object verts 
                obj_rest_verts, obj_mesh_faces = self.ds.load_rest_pose_object_geometry(object_name)
                obj_rest_verts = torch.from_numpy(obj_rest_verts)

                gt_obj_mesh_verts = self.ds.load_object_geometry_w_rest_geo(curr_gt_obj_rot_mat, \
                            curr_gt_obj_com_pos, obj_rest_verts.float())
                obj_mesh_verts = self.ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat, \
                            pred_seq_com_pos[idx], obj_rest_verts.float().to(pred_seq_com_pos.device))

            actual_len = seq_len[idx]

            human_jnts_list.append(mesh_jnts[0])
            human_verts_list.append(mesh_verts[0]) 
            obj_verts_list.append(obj_mesh_verts)
            trans_list.append(root_trans) 

            human_mesh_faces_list.append(mesh_faces)
            obj_mesh_faces_list.append(obj_mesh_faces) 

            if self.compute_metrics:
                continue 


            if dest_mesh_vis_folder is None:
                if vis_tag is None:
                    dest_mesh_vis_folder = os.path.join(self.vis_folder, "blender_mesh_vis", str(step))
                else:
                    dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag, str(step))
            
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)

            if vis_gt:
                ball_mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "ball_objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt.mp4")
            else:
                ball_mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "ball_objs_step_"+str(step)+"_bs_idx_"+str(idx))
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_sideview_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "sideview_imgs_step_"+str(step)+"_bs_idx_"+str(idx))
                
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+".mp4")
                out_sideview_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "sideview_vid_step_"+str(step)+"_bs_idx_"+str(idx)+".mp4")
                
                if vis_wo_scene:
                    ball_mesh_save_folder = ball_mesh_save_folder + "_vis_no_scene"
                    mesh_save_folder = mesh_save_folder + "_vis_no_scene"
                    out_rendered_img_folder = out_rendered_img_folder + "_vis_no_scene"
                    out_vid_file_path = out_vid_file_path.replace(".mp4", "_vis_no_scene.mp4") 

            if text_anno is not None:
                out_vid_file_path.replace(".mp4", "_"+text_anno.replace(" ", "_")+".mp4")

            start_obj_com_pos = data_dict['ori_obj_motion'][0, 0:1, :3] # 1 X 3 
            if planned_end_obj_com is None:
                end_obj_com_pos = data_dict['ori_obj_motion'][0, actual_len-1:actual_len, :3] # 1 X 3 
            else:
                end_obj_com_pos = planned_end_obj_com[idx].to(start_obj_com_pos.device) # 1 X 3 
            start_object_mesh = gt_obj_mesh_verts[0] # Nv X 3 
            if move_to_planned_path is not None:
                start_object_mesh += move_to_planned_path[idx].to(start_object_mesh.device) 
            end_object_mesh = gt_obj_mesh_verts[actual_len-1] # Nv X 3 
            if not os.path.exists(ball_mesh_save_folder):
                os.makedirs(ball_mesh_save_folder)
            ball_mesh_path = os.path.join(ball_mesh_save_folder, "conditions.ply")
            start_mesh_path = os.path.join(ball_mesh_save_folder, "start_object.ply")
            end_mesh_path = os.path.join(ball_mesh_save_folder, "end_object.ply") 
            self.export_to_mesh(start_object_mesh, obj_mesh_faces, start_mesh_path)
           
            if planned_waypoints_pos is not None:
                if planned_path_floor_height is None:
                    num_waypoints = planned_waypoints_pos[idx].shape[0]
                    for tmp_idx in range(num_waypoints):
                        if (tmp_idx+1) % 4 != 0:
                            planned_waypoints_pos[idx, tmp_idx, 2] = 0.05 
                else:
                    planned_waypoints_pos[idx, :, 2] = planned_path_floor_height + 0.05 

                if move_to_planned_path is None:
                    ball_for_vis_data = torch.cat((start_obj_com_pos, \
                                    planned_waypoints_pos[idx].to(end_obj_com_pos.device), end_obj_com_pos), dim=0) 
                else:
                    ball_for_vis_data = torch.cat((start_obj_com_pos+move_to_planned_path[idx].to(start_obj_com_pos.device), \
                                    planned_waypoints_pos[idx].to(end_obj_com_pos.device), end_obj_com_pos), dim=0) 
                # ball_for_vis_data: K X 3 
                #  
                if cano_quat is not None:
                    cano_quat_for_ball = transforms.quaternion_invert(cano_quat[0:1].repeat(ball_for_vis_data.shape[0], \
                                                        1)) # K X 4 
                    ball_for_vis_data = transforms.quaternion_apply(cano_quat_for_ball.to(ball_for_vis_data.device), ball_for_vis_data)
    
                self.create_ball_mesh(ball_for_vis_data, ball_mesh_path)
            else:
                curr_cond_mask = cond_mask[idx, :, 0] # T 
                waypoints_list = [start_obj_com_pos]
                end_obj_com_pos_xy = end_obj_com_pos.clone()
                # end_obj_com_pos_xy[:, 2] = 0.05
                waypoints_list.append(end_obj_com_pos_xy)
                curr_timesteps = curr_cond_mask.shape[0]
                for t_idx in range(curr_timesteps):
                    if curr_cond_mask[t_idx] == 0 and t_idx != 0:
                        selected_waypoint = data_dict['ori_obj_motion'][idx, t_idx:t_idx+1, :3]
                        selected_waypoint[:, 2] = 0.05
                        waypoints_list.append(selected_waypoint)

                ball_for_vis_data = torch.cat(waypoints_list, dim=0) # K X 3 
                self.create_ball_mesh(ball_for_vis_data, ball_mesh_path)

            # For faster debug visualization!!
            # mesh_verts = mesh_verts[:, ::30, :, :] # 1 X T X Nv X 3
            # obj_mesh_verts = obj_mesh_verts[::30, :, :] # T X Nv X 3 

            if cano_quat is not None:
                # mesh_verts: 1 X T X Nv X 3 
                # obj_mesh_verts: T X Nv' X 3 
                # cano_quat: K X 4 
                cano_quat_for_human = transforms.quaternion_invert(cano_quat[0:1][None].repeat(mesh_verts.shape[1], \
                                                            mesh_verts.shape[2], 1)) # T X Nv X 4 
                cano_quat_for_obj = transforms.quaternion_invert(cano_quat[0:1][None].repeat(obj_mesh_verts.shape[0], \
                                                            obj_mesh_verts.shape[1], 1)) # T X Nv X 4
                mesh_verts = transforms.quaternion_apply(cano_quat_for_human.to(mesh_verts.device), mesh_verts[0])
                obj_mesh_verts = transforms.quaternion_apply(cano_quat_for_obj.to(obj_mesh_verts.device), obj_mesh_verts) 

                save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy(), mesh_faces.detach().cpu().numpy(), \
                        obj_mesh_verts.detach().cpu().numpy(), obj_mesh_faces, mesh_save_folder)
            else:
                if gen_long_seq:
                    save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy()[0], \
                            mesh_faces.detach().cpu().numpy(), \
                            obj_mesh_verts.detach().cpu().numpy(), obj_mesh_faces, mesh_save_folder)
                else: # For single window
                    save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy()[0][:seq_len[idx]], \
                            mesh_faces.detach().cpu().numpy(), \
                            obj_mesh_verts.detach().cpu().numpy()[:seq_len[idx]], obj_mesh_faces, mesh_save_folder)

            # continue 
            if move_to_planned_path is not None:
                curr_scene_name = planned_scene_names.split("/")[-4]
                root_blend_file_folder = "/move/u/jiamanli/datasets/FullBodyManipCapture/processed_manip_data/replica_blender_files"
                
                # Top-down view visualization 
                curr_scene_blend_path = os.path.join(root_blend_file_folder, self.test_scene_name+"_topview.blend")
                # if not os.path.exists(dest_out_vid_path):
                if not save_obj_only:
                    run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, out_vid_file_path, \
                            condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                            scene_blend_path=curr_scene_blend_path) 
                
            else:
                floor_blend_path = os.path.join(self.data_root_folder, "blender_files/floor_colorful_mat.blend")
                if planned_end_obj_com is not None:
                    if dest_out_vid_path is None:
                        dest_out_vid_path = out_vid_file_path.replace(".mp4", "_wo_scene.mp4")

                    if not os.path.exists(dest_out_vid_path):
                        if not save_obj_only:
                            run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, dest_out_vid_path, \
                                condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                                scene_blend_path=floor_blend_path)
                    
                else:
                    if dest_out_vid_path is None:
                        dest_out_vid_path = out_vid_file_path.replace(".mp4", "_wo_scene.mp4")
                    if not os.path.exists(dest_out_vid_path):
                        if not vis_gt: # Skip GT visualiation 
                            if not save_obj_only:
                                run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, dest_out_vid_path, \
                                        condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                                        scene_blend_path=floor_blend_path)

                    if vis_gt: 
                        if not save_obj_only:
                            run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, dest_out_vid_path, \
                                    condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                                    scene_blend_path=floor_blend_path)
                    

            if idx > 1:
                break 

        return human_verts_list, human_jnts_list, trans_list, global_rot_mat, pred_seq_com_pos, pred_obj_rot_mat, \
        obj_verts_list, human_mesh_faces_list, obj_mesh_faces_list, dest_out_vid_path  

    def gen_vis_res(self, all_res_list, data_dict, step, cond_mask, vis_gt=False, vis_tag=None, \
                curr_object_name=None, dest_out_vid_path=None, dest_mesh_vis_folder=None, \
                save_obj_only=False):

        # Prepare list used for evaluation. 
        human_jnts_list = []
        human_verts_list = [] 
        obj_verts_list = [] 
        trans_list = []
        human_mesh_faces_list = []
        obj_mesh_faces_list = [] 

        # all_res_list: N X T X (3+9) 
        num_seq = all_res_list.shape[0]

        pred_normalized_obj_trans = all_res_list[:, :, :3] # N X T X 3 
        pred_seq_com_pos = self.ds.de_normalize_obj_pos_min_max(pred_normalized_obj_trans)

        if self.use_random_frame_bps:
            reference_obj_rot_mat = data_dict['reference_obj_rot_mat'] # N X 1 X 3 X 3 

            pred_obj_rel_rot_mat = all_res_list[:, :, 3:3+9].reshape(num_seq, -1, 3, 3) # N X T X 3 X 3
            pred_obj_rot_mat = self.ds.rel_rot_to_seq(pred_obj_rel_rot_mat, reference_obj_rot_mat)

        num_joints = 24
    
        normalized_global_jpos = all_res_list[:, :, 3+9:3+9+num_joints*3].reshape(num_seq, -1, num_joints, 3)
        global_jpos = self.ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, num_joints, 3))
        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3) # N X T X 22 X 3 

        global_root_jpos = global_jpos[:, :, 0, :].clone() # N X T X 3 

        global_rot_6d = all_res_list[:, :, 3+9+24*3:3+9+24*3+22*6].reshape(num_seq, -1, 22, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # N X T X 22 X 3 X 3 

        trans2joint = data_dict['trans2joint'].to(all_res_list.device).squeeze(1) # BS X  3 
        seq_len = data_dict['seq_len'] # BS, should only be used during for single window generation. 
        if all_res_list.shape[0] != trans2joint.shape[0]:
            trans2joint = trans2joint.repeat(num_seq, 1, 1) # N X 24 X 3 
            seq_len = seq_len.repeat(num_seq) # N 
        seq_len = seq_len.detach().cpu().numpy() # N 

        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx] # T X 22 X 3 X 3 
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat) # T X 22 X 3 X 3 
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 22 X 3 
            
            curr_global_root_jpos = global_root_jpos[idx] # T X 3
     
            curr_trans2joint = trans2joint[idx:idx+1].clone() # 1 X 3 
            
            root_trans = curr_global_root_jpos + curr_trans2joint.to(curr_global_root_jpos.device) # T X 3 

            # Generate global joint position 
            betas = data_dict['betas'][idx]
            gender = data_dict['gender'][idx]
            
            curr_gt_obj_rot_mat = data_dict['obj_rot_mat'][idx] # T X 3 X 3
            curr_gt_obj_com_pos = data_dict['obj_com_pos'][idx] # T X 3 
          
            curr_obj_rot_mat = pred_obj_rot_mat[idx] # T X 3 X 3 
            curr_obj_quat = transforms.matrix_to_quaternion(curr_obj_rot_mat)
            curr_obj_rot_mat = transforms.quaternion_to_matrix(curr_obj_quat) # Potentially avoid some prediction not satisfying rotation matrix requirements.

            if curr_object_name is not None: 
                object_name = curr_object_name 
            else:
                curr_seq_name = data_dict['seq_name'][idx]
                object_name = data_dict['obj_name'][idx]
          
            # Get human verts 
            mesh_jnts, mesh_verts, mesh_faces = \
                run_smplx_model(root_trans[None].cuda(), curr_local_rot_aa_rep[None].cuda(), \
                betas.cuda(), [gender], self.ds.bm_dict, return_joints24=True)

            # Get object verts 
            obj_rest_verts, obj_mesh_faces = self.ds.load_rest_pose_object_geometry(object_name)
            obj_rest_verts = torch.from_numpy(obj_rest_verts)

            gt_obj_mesh_verts = self.ds.load_object_geometry_w_rest_geo(curr_gt_obj_rot_mat.to(pred_seq_com_pos.device), \
                        curr_gt_obj_com_pos.to(pred_seq_com_pos.device), obj_rest_verts.float().to(pred_seq_com_pos.device))
            obj_mesh_verts = self.ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat, \
                        pred_seq_com_pos[idx], obj_rest_verts.float().to(pred_seq_com_pos.device))

            actual_len = seq_len[idx]

            human_jnts_list.append(mesh_jnts[0])
            human_verts_list.append(mesh_verts[0]) 
            obj_verts_list.append(obj_mesh_verts)
            trans_list.append(root_trans) 

            human_mesh_faces_list.append(mesh_faces)
            obj_mesh_faces_list.append(obj_mesh_faces) 

            if self.compute_metrics:
                continue 

            if dest_mesh_vis_folder is None:
                if vis_tag is None:
                    dest_mesh_vis_folder = os.path.join(self.vis_folder, "blender_mesh_vis", str(step))
                else:
                    dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag, str(step))
            
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)

            if vis_gt:
                ball_mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "ball_objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt.mp4")
            else:
                ball_mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "ball_objs_step_"+str(step)+"_bs_idx_"+str(idx))
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_sideview_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "sideview_imgs_step_"+str(step)+"_bs_idx_"+str(idx))
                
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+".mp4")
                out_sideview_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "sideview_vid_step_"+str(step)+"_bs_idx_"+str(idx)+".mp4")
            

            start_obj_com_pos = data_dict['ori_obj_motion'][0, 0:1, :3] # 1 X 3 
            end_obj_com_pos = data_dict['ori_obj_motion'][0, actual_len-1:actual_len, :3] # 1 X 3 
           
            start_object_mesh = gt_obj_mesh_verts[0].cpu() # Nv X 3 
           
            end_object_mesh = gt_obj_mesh_verts[actual_len-1] # Nv X 3 
            if not os.path.exists(ball_mesh_save_folder):
                os.makedirs(ball_mesh_save_folder)
            ball_mesh_path = os.path.join(ball_mesh_save_folder, "conditions.ply")
            start_mesh_path = os.path.join(ball_mesh_save_folder, "start_object.ply")
            end_mesh_path = os.path.join(ball_mesh_save_folder, "end_object.ply") 
            self.export_to_mesh(start_object_mesh, obj_mesh_faces, start_mesh_path)
           
            curr_cond_mask = cond_mask[idx, :, 0] # T 
            waypoints_list = [start_obj_com_pos]
            end_obj_com_pos_xy = end_obj_com_pos.clone()
            # end_obj_com_pos_xy[:, 2] = 0.05
            waypoints_list.append(end_obj_com_pos_xy)
            curr_timesteps = curr_cond_mask.shape[0]
            for t_idx in range(curr_timesteps):
                if curr_cond_mask[t_idx] == 0 and t_idx != 0:
                    selected_waypoint = data_dict['ori_obj_motion'][idx, t_idx:t_idx+1, :3]
                    selected_waypoint[:, 2] = 0.05
                    waypoints_list.append(selected_waypoint)

            ball_for_vis_data = torch.cat(waypoints_list, dim=0) # K X 3 
            self.create_ball_mesh(ball_for_vis_data, ball_mesh_path)

            # For faster debug visualization!!
            # mesh_verts = mesh_verts[:, ::30, :, :] # 1 X T X Nv X 3
            # obj_mesh_verts = obj_mesh_verts[::30, :, :] # T X Nv X 3 

            save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy()[0][:seq_len[idx]], \
                    mesh_faces.detach().cpu().numpy(), \
                    obj_mesh_verts.detach().cpu().numpy()[:seq_len[idx]], obj_mesh_faces, mesh_save_folder)

            if dest_out_vid_path is None:
                dest_out_vid_path = out_vid_file_path

            floor_blend_path = os.path.join(self.data_root_folder, "blender_files/floor_colorful_mat.blend")
            if vis_gt: 
                run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, dest_out_vid_path, \
                        condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                        scene_blend_path=floor_blend_path)
            else:
                run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, dest_out_vid_path, \
                        condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                        scene_blend_path=floor_blend_path)
            
            if idx >= 1:
                break 

        return human_verts_list, human_jnts_list, trans_list, global_rot_mat, pred_seq_com_pos, pred_obj_rot_mat, \
        obj_verts_list, human_mesh_faces_list, obj_mesh_faces_list, dest_out_vid_path  

def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model  
    repr_dim = 3 + 9 # Object relative translation (3) and relative rotation matrix (9)  

    repr_dim += 24 * 3 + 22 * 6 # Global human joint positions and rotation 6D representation 

    if opt.use_object_keypoints:
        repr_dim += 4 

    loss_type = "l1"

    diffusion_model = ObjectCondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                input_first_human_pose=opt.input_first_human_pose, \
                use_object_keypoints=opt.use_object_keypoints) 
   
    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=400000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
    )
    trainer.train()

    torch.cuda.empty_cache()

def run_sample(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # Define model     
    repr_dim = 3 + 9 

    repr_dim += 24 * 3 + 22 * 6 

    if opt.use_object_keypoints:
        repr_dim += 4 
   
    loss_type = "l1"

    diffusion_model = ObjectCondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                input_first_human_pose=opt.input_first_human_pose, \
                use_object_keypoints=opt.use_object_keypoints)

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=8000000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False 
    )
   
    if opt.use_long_planned_path:
        trainer.cond_sample_res_w_long_planned_path() 
    else:
        trainer.cond_sample_res()

    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--wandb_pj_name', type=str, default='chois_projects', help='project name')
    parser.add_argument('--entity', default='', help='W&B entity')
    parser.add_argument('--exp_name', default='chois', help='save to project/name')

    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--window', type=int, default=120, help='horizon')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--pretrained_model', type=str, default="", help='checkpoint')

    parser.add_argument('--data_root_folder', type=str, default="", help='data root folder')

    parser.add_argument('--save_res_folder', type=str, default="", help='save res folder')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")

    # For testing sampled results w planned path 
    parser.add_argument("--use_long_planned_path", action="store_true")

    # For testing sampled results on training dataset 
    parser.add_argument("--test_on_train", action="store_true")

    # For quantitative evaluation. 
    parser.add_argument("--for_quant_eval", action="store_true")

    # Train and test on different objects. 
    parser.add_argument("--use_object_split", action="store_true")

    # Add language conditions. 
    parser.add_argument("--add_language_condition", action="store_true")

    # Input the first human pose, maybe can connect the windows better.  
    parser.add_argument("--input_first_human_pose", action="store_true")

    parser.add_argument("--use_guidance_in_denoising", action="store_true")

    parser.add_argument("--compute_metrics", action="store_true")

    # Add rest offsets for body shape information. 
    parser.add_argument("--use_random_frame_bps", action="store_true")

    parser.add_argument('--test_object_name', type=str, default="", help='object name for long sequence generation testing')
    parser.add_argument('--test_scene_name', type=str, default="", help='scene name for long sequence generation testing')

    parser.add_argument("--use_object_keypoints", action="store_true")

    parser.add_argument('--loss_w_feet', type=float, default=1, help='the loss weight for feet contact loss')
    parser.add_argument('--loss_w_fk', type=float, default=1, help='the loss weight for fk loss')
    parser.add_argument('--loss_w_obj_pts', type=float, default=1, help='the loss weight for fk loss')

    parser.add_argument("--add_semantic_contact_labels", action="store_true")

    parser.add_argument("--test_unseen_objects", action="store_true")

   
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    else:
        run_train(opt, device)
