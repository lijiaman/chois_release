import sys
sys.path.append("/viscam/u/jiamanli/github/scene_aware_manip")
sys.path.append("../../")

import os
import numpy as np
import joblib 
import trimesh  
import json 

import random 

import codecs as cs

import torch
from torch.utils.data import Dataset

import pytorch3d.transforms as transforms 

from human_body_prior.body_model.body_model import BodyModel

from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)

def rotate(points, R):
    shape = list(points.shape)
    points = to_tensor(points)
    R = to_tensor(R)
    if len(shape)>3:
        points = points.squeeze()
    if len(shape)<3:
        points = points.unsqueeze(dim=1)
    if R.shape[0] > shape[0]:
        shape[0] = R.shape[0]
    r_points = torch.matmul(points, R.transpose(1,2))
    return r_points.reshape(shape)

def get_smpl_parents(use_joints24=True):
    smplh_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh_amass"
    bm_path = os.path.join(smplh_path, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 

    if use_joints24:
        parents = ori_kintree_table[0, :23] # 23 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.

        parents_list = parents.tolist()
        parents_list.append(ori_kintree_table[0][37])
        parents = np.asarray(parents_list) # 24 
    else:
        parents = ori_kintree_table[0, :22] # 22 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.
    
    return parents

def local2global_pose(local_pose):
    # local_pose: T X J X 3 X 3 
    kintree = get_smpl_parents(use_joints24=False) 

    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose # T X J X 3 X 3 

def quat_ik_torch(grot_mat):
    # grot: T X J X 3 X 3 
    parents = get_smpl_parents(use_joints24=False) 

    grot = transforms.matrix_to_quaternion(grot_mat) # T X J X 4 

    res = torch.cat(
            [
                grot[..., :1, :],
                transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), \
                grot[..., 1:, :]),
            ],
            dim=-2) # T X J X 4 

    res_mat = transforms.quaternion_to_matrix(res) # T X J X 3 X 3 

    return res_mat 

def quat_fk_torch(lrot_mat, lpos, use_joints24=True):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J/(J+2) X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    if use_joints24:
        parents = get_smpl_parents(use_joints24=True)
    else:
        parents = get_smpl_parents() 

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        if i < lrot.shape[-2]:
            gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res

class CanoObjectTrajDataset(Dataset):
    def __init__(
        self,
        train,
        data_root_folder,
        window=120,
        word_vectorizer=None, 
        use_object_splits=False,
        input_language_condition=True,
        return_dict=True, 
    ):
        self.train = train
        
        self.window = window 

        self.w_vectorizer = word_vectorizer 

        self.return_dict = return_dict 

        self.use_object_splits = use_object_splits 
        self.train_objects = ["largetable", "woodchair", "plasticbox", "largebox", "smallbox", "trashcan", "monitor", \
                    "floorlamp", "clothesstand"] # 10 objects 
        self.test_objects = ["smalltable", "whitechair", "suitcase", "tripod"]

        self.input_language_condition = input_language_condition 

        self.parents = get_smpl_parents() # 24/22 

        self.data_root_folder = data_root_folder 
        self.language_anno_folder = os.path.join(self.data_root_folder, "omomo_text_anno_txt_data") 
        
        train_subjects = []
        test_subjects = []
        num_subjects = 17 
        for s_idx in range(1, num_subjects+1):
            if s_idx >= 16:
                test_subjects.append("sub"+str(s_idx))
            else:
                train_subjects.append("sub"+str(s_idx))

        keep_same_len_window = False 
        self.keep_same_len_window = keep_same_len_window 

        if keep_same_len_window:
            if self.train:
                seq_data_path = os.path.join(data_root_folder, "train_diffusion_manip_seq_joints24.p")  
                processed_data_path = os.path.join(data_root_folder, "cano_train_diffusion_manip_window_"+str(self.window)+"_joints24_same_len_window.p")   
            else:    
                seq_data_path = os.path.join(data_root_folder, "test_diffusion_manip_seq_joints24.p")
                processed_data_path = os.path.join(data_root_folder, "cano_test_diffusion_manip_window_"+str(self.window)+"_joints24_same_len_window.p")

            min_max_mean_std_data_path = os.path.join(data_root_folder, "cano_min_max_mean_std_data_window_"+str(self.window)+"_joints24_same_len_window.p")
        else:
            if self.train:
                seq_data_path = os.path.join(data_root_folder, "train_diffusion_manip_seq_joints24.p")  
                processed_data_path = os.path.join(data_root_folder, "cano_train_diffusion_manip_window_"+str(self.window)+"_joints24.p")   
            else:    
                seq_data_path = os.path.join(data_root_folder, "test_diffusion_manip_seq_joints24.p")
                processed_data_path = os.path.join(data_root_folder, "cano_test_diffusion_manip_window_"+str(self.window)+"_joints24.p")

            min_max_mean_std_data_path = os.path.join(data_root_folder, "cano_min_max_mean_std_data_window_"+str(self.window)+"_joints24.p")

        if os.path.exists(processed_data_path):
            self.window_data_dict = joblib.load(processed_data_path)          

        if os.path.exists(min_max_mean_std_data_path):
            min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
           
        self.global_jpos_min = torch.from_numpy(min_max_mean_std_jpos_data['global_jpos_min']).float().reshape(24, 3)[None]
        self.global_jpos_max = torch.from_numpy(min_max_mean_std_jpos_data['global_jpos_max']).float().reshape(24, 3)[None]

        # For jpos mean, std
        mean_std_jpos_path = os.path.join(data_root_folder, "t2m_mean_std_jpos.p")
        if os.path.exists(mean_std_jpos_path):
            mean_std_dict = joblib.load(mean_std_jpos_path)
            jpos_mean, jpos_std = mean_std_dict['jpos_mean'], mean_std_dict['jpos_std'] 
        else:
            jpos_mean, jpos_std = self.compute_mean_std() 

            mean_std_dict = {} 
            mean_std_dict['jpos_mean'] = jpos_mean 
            mean_std_dict['jpos_std'] = jpos_std 

            joblib.dump(mean_std_dict, mean_std_jpos_path) 

        self.mean_jpos = torch.from_numpy(jpos_mean).float() # 72  
        self.std_jpos = torch.from_numpy(jpos_std).float() # 72 

        self.window_data_dict = self.filter_out_short_sequences() 
        if self.input_language_condition:
            self.window_data_dict = self.filter_out_seq_wo_text() 

        # Get train and validation statistics. 
        if self.train:
            print("Total number of windows for training:{0}".format(len(self.window_data_dict))) # all, Total number of windows for training:28859
        else:
            print("Total number of windows for validation:{0}".format(len(self.window_data_dict))) # all, 3224 

        # Prepare SMPLX model 
        soma_work_base_dir = os.path.join(self.data_root_folder, 'smpl_all_models')
        support_base_dir = soma_work_base_dir 
        surface_model_type = "smplx"
        # surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "male", 'model.npz')
        # surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "female", 'model.npz')
        surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_MALE.npz")
        surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_FEMALE.npz")
        dmpl_fname = None
        num_dmpls = None 
        num_expressions = None
        num_betas = 16 

        self.male_bm = BodyModel(bm_fname=surface_model_male_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname)
        self.female_bm = BodyModel(bm_fname=surface_model_female_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname)

        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False 

        self.male_bm = self.male_bm.cuda()
        self.female_bm = self.female_bm.cuda()
        
        self.bm_dict = {'male' : self.male_bm, 'female' : self.female_bm}

    def compute_mean_std(self):
        # compute joint position mean, std. 
        jpos_list = []
        for k in self.window_data_dict:
            curr_jpos = self.window_data_dict[k]['motion'][:, :24*3] # T X 72(24*3) 
            jpos_list.append(curr_jpos)

        jpos_list = np.vstack(jpos_list) # (N*T) X 72 

        jpos_mean = np.mean(jpos_list, axis=0) # 72 
        jpos_std = np.std(jpos_list, axis=0) # 72 

        return jpos_mean, jpos_std 

    def load_language_annotation(self, seq_name):
        # seq_name: sub16_clothesstand_000, etc. 
        # json_path = os.path.join(self.language_anno_folder, seq_name+".json")
        # json_data = json.load(open(json_path, 'r'))
        
        # text_anno = json_data[seq_name]

        # Load .txt file 
        txt_path = os.path.join(self.language_anno_folder, seq_name+".txt")
        with cs.open(txt_path) as f:
            for line in f.readlines():
                text_dict = {}
                line_split = line.strip().split('#')
                caption = line_split[0]
                tokens = line_split[1].split(' ')
               
                text_dict['caption'] = caption
                text_dict['tokens'] = tokens
                
        f.close() 

        return text_dict 

    def filter_out_short_sequences(self):
        # Remove some sequences from window_data_dict such that we have some unseen objects during testing. 
        new_cnt = 0
        new_window_data_dict = {}
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]
            seq_name = window_data['seq_name']
            object_name = seq_name.split("_")[1]
           
            curr_seq_len = window_data['motion'].shape[0]

            if curr_seq_len < self.window:
                continue 

            # if self.window_data_dict[k]['start_t_idx'] != 0:
            #     continue 

            new_window_data_dict[new_cnt] = self.window_data_dict[k]
            if "ori_w_idx" in self.window_data_dict[k]:
                new_window_data_dict[new_cnt]['ori_w_idx'] = self.window_data_dict[k]['ori_w_idx']
            else:
                new_window_data_dict[new_cnt]['ori_w_idx'] = k 
            
            new_cnt += 1

        return new_window_data_dict

    def filter_out_object_split(self):
        # Remove some sequences from window_data_dict such that we have some unseen objects during testing. 
        new_cnt = 0
        new_window_data_dict = {}
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]
            seq_name = window_data['seq_name']
            object_name = seq_name.split("_")[1]
            if self.train and object_name in self.train_objects:
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                new_window_data_dict[new_cnt]['ori_w_idx'] = k 
                new_cnt += 1

            if (not self.train) and object_name in self.test_objects:
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                new_window_data_dict[new_cnt]['ori_w_idx'] = k 
                new_cnt += 1

        return new_window_data_dict

    def filter_out_seq_wo_text(self):
        new_cnt = 0
        new_window_data_dict = {}
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]
            seq_name = window_data['seq_name']
            text_json_path = os.path.join(self.language_anno_folder, seq_name+".txt")
            if os.path.exists(text_json_path):
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                if "ori_w_idx" in self.window_data_dict[k]: # Based on filtered results split by objects. 
                    new_window_data_dict[new_cnt]['ori_w_idx'] = self.window_data_dict[k]['ori_w_idx']
                else: # Based on the original window_daia_dict. 
                    new_window_data_dict[new_cnt]['ori_w_idx'] = k 
                new_cnt += 1

        return new_window_data_dict

    def normalize_jpos_min_max(self, ori_jpos):
        # ori_jpos: T X 22/24 X 3 
        # or BS X T X J X 3 
        if ori_jpos.dim() == 4:
            normalized_jpos = (ori_jpos - self.global_jpos_min.to(ori_jpos.device)[None])/(self.global_jpos_max.to(ori_jpos.device)[None] \
            -self.global_jpos_min.to(ori_jpos.device)[None])
        else:
            normalized_jpos = (ori_jpos - self.global_jpos_min.to(ori_jpos.device))/(self.global_jpos_max.to(ori_jpos.device)\
            -self.global_jpos_min.to(ori_jpos.device))
        normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 

        return normalized_jpos # (BS X) T X 22/24 X 3 

    def normalize_specific_jpos_min_max(self, ori_jpos, j_idx):
        # ori_jpos: T X 3 
        # or BS X T X 3 
        if ori_jpos.dim() == 3:
            normalized_jpos = (ori_jpos - self.global_jpos_min[:, j_idx, :].to(ori_jpos.device)[None])/(self.global_jpos_max[:, j_idx, :].to(ori_jpos.device)[None] \
            -self.global_jpos_min[:, j_idx, :].to(ori_jpos.device)[None])
        else:
            normalized_jpos = (ori_jpos - self.global_jpos_min[:, j_idx, :].to(ori_jpos.device))/(self.global_jpos_max[:, j_idx, :].to(ori_jpos.device)\
            -self.global_jpos_min[:, j_idx, :].to(ori_jpos.device))
        normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 

        return normalized_jpos # (BS X) T X 22/24 X 3 

    def de_normalize_jpos_min_max(self, normalized_jpos):
        # normalized_jpos: T X 22/24 X 3 
        # or BS X T X J X 3 
        normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range
        
        if normalized_jpos.dim() == 4:
            de_jpos = normalized_jpos * (self.global_jpos_max.to(normalized_jpos.device)[None]-\
            self.global_jpos_min.to(normalized_jpos.device)[None]) + self.global_jpos_min.to(normalized_jpos.device)[None]
        else:
            de_jpos = normalized_jpos * (self.global_jpos_max.to(normalized_jpos.device)-\
            self.global_jpos_min.to(normalized_jpos.device)) + self.global_jpos_min.to(normalized_jpos.device)

        return de_jpos # (BS X) T X 22/24 X 3
    
    def de_normalize_specific_jpos_min_max(self, normalized_jpos, j_idx):
        # normalized_jpos: T X 3 
        # or BS X T X 3 
        normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range
        
        if normalized_jpos.dim() == 3:
            de_jpos = normalized_jpos * (self.global_jpos_max[:, j_idx, :].to(normalized_jpos.device)[None]-\
            self.global_jpos_min[:, j_idx, :].to(normalized_jpos.device)[None]) + \
            self.global_jpos_min[:, j_idx, :].to(normalized_jpos.device)[None]
        else:
            de_jpos = normalized_jpos * (self.global_jpos_max[:, j_idx, :].to(normalized_jpos.device)-\
            self.global_jpos_min[:, j_idx, :].to(normalized_jpos.device)) + \
            self.global_jpos_min[:, j_idx, :].to(normalized_jpos.device)

        return de_jpos # (BS X) T X 3

    def __len__(self):
        return len(self.window_data_dict)
    
    def normalize_jpos_mean_std(self, ori_jpos):
        # ori_jpos: T X 72/BS X T X 72
        if ori_jpos.dim() == 3: # BS X T X 72 
            norm_jpos = (ori_jpos - self.mean_jpos[None, None, :])/self.std_jpos[None, None, :] 
        else: # T X 72 
            norm_jpos = (ori_jpos - self.mean_jpos[None, :])/self.std_jpos[None, :] 

        return norm_jpos 
    
    def de_normalize_jpos_mean_std(self, norm_jpos):
        # norma_jpos: T X 72/BS X T X 72 
        if norm_jpos.dim() == 3:
            ori_jpos = norm_jpos * self.std_jpos[None, None, :] + self.mean_jpos[None, None, :]
        else:
            ori_jpos = norm_jpos * self.std_jpos[None, :] + self.mean_jpos[None, :]

        return ori_jpos 

    def __getitem__(self, index):
        # index = 0 # For debug 
        # data_input = self.window_data_dict[index]['motion']
        
        ori_jpos = self.window_data_dict[index]['motion'][:, :24*3] # T X 72 
        ori_jpos = torch.from_numpy(ori_jpos).float() # T X 72 
        data_input = self.normalize_jpos_mean_std(ori_jpos) # T X 72 

        seq_name = self.window_data_dict[index]['seq_name'] 
        object_name = seq_name.split("_")[1]
        
        window_s_idx = self.window_data_dict[index]['start_t_idx']
        window_e_idx = self.window_data_dict[index]['end_t_idx']

        trans2joint = self.window_data_dict[index]['trans2joint'] 
        rest_human_offsets = self.window_data_dict[index]['rest_human_offsets']  

        # num_joints = 24         
        # normalized_jpos = self.normalize_jpos_min_max(data_input[:, :num_joints*3].reshape(-1, num_joints, 3)) # T X 22 X 3 
       
        # global_joint_rot = data_input[:, 2*num_joints*3:] # T X (22*6)

        # new_data_input = torch.cat((normalized_jpos.reshape(-1, num_joints*3), global_joint_rot), dim=1)
        # ori_data_input = torch.cat((data_input[:, :num_joints*3], global_joint_rot), dim=1)
      
        new_data_input = data_input 

        # Add padding. 
        actual_steps = new_data_input.shape[0]
        if actual_steps < self.window:
            paded_new_data_input = torch.cat((new_data_input, torch.zeros(self.window-actual_steps, new_data_input.shape[-1])), dim=0)
            # paded_ori_data_input = torch.cat((ori_data_input, torch.zeros(self.window-actual_steps, ori_data_input.shape[-1])), dim=0)  
        else:
            paded_new_data_input = new_data_input 
            # paded_ori_data_input = ori_data_input 

        data_input_dict = {}
        data_input_dict['motion'] = paded_new_data_input
        # data_input_dict['ori_motion'] = paded_ori_data_input 

        data_input_dict['betas'] = self.window_data_dict[index]['betas']
        data_input_dict['gender'] = str(self.window_data_dict[index]['gender'])
       
        data_input_dict['seq_name'] = seq_name
        data_input_dict['obj_name'] = seq_name.split("_")[1]

        data_input_dict['seq_len'] = actual_steps 

        data_input_dict['trans2joint'] = trans2joint 
        data_input_dict['rest_human_offsets'] = rest_human_offsets 

        data_input_dict['s_idx'] = window_s_idx
        data_input_dict['e_idx'] = window_e_idx 
        
        # Load language annotation 
        text_data = self.load_language_annotation(seq_name) 
        caption, tokens = text_data['caption'], text_data['tokens']

        max_text_len = 30 
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        data_input_dict['pos_one_hots'] = pos_one_hots 
        data_input_dict['word_embeddings'] = word_embeddings 

        if self.return_dict:
            return data_input_dict 
        else:
            return word_embeddings, pos_one_hots, caption, sent_len, \
            paded_new_data_input, actual_steps, '_'.join(tokens)

        # data_input_dict['motion']: T X (22*3+22*6) range [-1, 1]
        # data_input_dict['obj_bps]: T X N X 3 
