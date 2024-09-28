import sys
sys.path.append("../../")

import os
import numpy as np
import joblib 
import trimesh  
import json 

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import pytorch3d.transforms as transforms 

from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform
from bps_torch.tools import sample_uniform_cylinder

from human_body_prior.body_model.body_model import BodyModel

from manip.lafan1.utils import rotate_at_frame_w_obj 

from manip.data.cano_traj_dataset import get_smpl_parents, quat_fk_torch, quat_ik_torch, local2global_pose 


class UnseenCanoObjectTrajDataset(Dataset):
    def __init__(
        self,
        train,
        data_root_folder,
        window=120,
        use_object_splits=False,
        input_language_condition=False,
        use_first_frame_bps=False, 
        use_random_frame_bps=False, 
        test_long_seq=False, 
    ):
        
        self.window = window 

        self.use_object_splits = use_object_splits 
        self.train_objects = ["largetable", "woodchair", "plasticbox", "largebox", "smallbox", "trashcan", "monitor", \
                    "floorlamp", "clothesstand", "vacuum"] # 10 objects 
        self.test_objects = ["smalltable", "whitechair", "suitcase", "tripod", "mop"]

        object2seq_dict = {"clothesstand": "sub16_clothesstand_004", "largebox": "sub16_largebox_005", \
                "largetable": "sub16_largetable_006", "plasticbox": "sub16_plasticbox_011", "trashcan": "sub16_trashcan_000", \
                "whitechair": "sub16_whitechair_016", \
                "floorlamp": "sub17_floorlamp_001", "monitor": "sub17_monitor_011", \
                "smallbox": "sub17_smallbox_018", "smalltable": "sub17_smalltable_034", \
                "suitcase": "sub17_suitcase_006", \
                "tripod": "sub17_tripod_009", "woodchair": "sub17_woodchair_035"}
      
        self.selected_seq_names = []
        for k in object2seq_dict:
            self.selected_seq_names.append(object2seq_dict[k]) 

        self.test_long_seq = test_long_seq 

        self.input_language_condition = input_language_condition 

        self.use_first_frame_bps = use_first_frame_bps 

        self.use_random_frame_bps = use_random_frame_bps 

        self.parents = get_smpl_parents() # 24/22 

        self.data_root_folder = data_root_folder 
        self.obj_geo_root_folder = os.path.join(self.data_root_folder, "captured_objects")
        
        self.rest_object_geo_folder = "/move/u/jiamanli/datasets/semantic_manip/unseen_objects_data/selected_unseen_objects/selected_rotated_zeroed_obj_files"

        self.bps_path = "./bps.pt"

        self.language_anno_folder = os.path.join(self.data_root_folder, "omomo_text_anno_json_data") 
        
        self.contact_npy_folder = os.path.join(self.data_root_folder, "contact_labels_npy_files")

        train_subjects = []
        test_subjects = []
        num_subjects = 17 
        for s_idx in range(1, num_subjects+1):
            if s_idx >= 16:
                test_subjects.append("sub"+str(s_idx))
            else:
                train_subjects.append("sub"+str(s_idx))

        if self.test_long_seq:
            dest_obj_bps_npy_folder_for_test = os.path.join(self.data_root_folder, \
                            "unseen_objects_bps_npy_files_for_test_joints24_long_seq")
        else:
            dest_obj_bps_npy_folder_for_test = os.path.join(self.data_root_folder, \
                            "unseen_objects_bps_npy_files_for_test_joints24")

        if not os.path.exists(dest_obj_bps_npy_folder_for_test):
            os.makedirs(dest_obj_bps_npy_folder_for_test)

        self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder_for_test 

        seq_data_path = os.path.join(data_root_folder, "test_diffusion_manip_seq_joints24.p")
        # processed_data_path = os.path.join(data_root_folder, \
        #     "unseen_objects_test_diffusion_manip_window_joints24.p")
        
        processed_data_path = os.path.join(data_root_folder, \
        "cano_test_diffusion_manip_window_"+str(self.window)+"_joints24.p")
       
        min_max_mean_std_data_path = os.path.join(data_root_folder, \
                        "cano_min_max_mean_std_data_window_"+str(self.window)+"_joints24.p")
       
        self.prep_bps_data()

        if os.path.exists(processed_data_path):
            self.window_data_dict = joblib.load(processed_data_path)

        self.generate_data_for_unseen_objects()          

        min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
           
        self.global_jpos_min = torch.from_numpy(min_max_mean_std_jpos_data['global_jpos_min']).float().reshape(24, 3)[None]
        self.global_jpos_max = torch.from_numpy(min_max_mean_std_jpos_data['global_jpos_max']).float().reshape(24, 3)[None]

        self.obj_pos_min = torch.from_numpy(min_max_mean_std_jpos_data['obj_com_pos_min']).float().reshape(1, 3)
        self.obj_pos_max = torch.from_numpy(min_max_mean_std_jpos_data['obj_com_pos_max']).float().reshape(1, 3)

        # Get train and validation statistics. 
        print("Total number of windows for validation:{0}".format(len(self.new_window_data_dict)))

        # Prepare SMPLX model 
        soma_work_base_dir = os.path.join(self.data_root_folder, 'smpl_all_models')
        support_base_dir = soma_work_base_dir 
        surface_model_type = "smplx"
        surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_MALE.npz")
        surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_FEMALE.npz")
        surface_model_neutral_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_NEUTRAL.npz")
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
        self.neutral_bm = BodyModel(bm_fname=surface_model_neutral_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname)

        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False 
        for p in self.neutral_bm.parameters():
            p.requires_grad = False 

        self.male_bm = self.male_bm.cuda()
        self.female_bm = self.female_bm.cuda()
        self.neutral_bm = self.neutral_bm.cuda() 
        
        self.bm_dict = {'male' : self.male_bm, 'female' : self.female_bm, 'neutral': self.neutral_bm}

    def load_object_geometry_w_rest_geo(self, obj_rot, obj_com_pos, rest_verts):
        # obj_scale: T, obj_rot: T X 3 X 3, obj_com_pos: T X 3, rest_veerts: Nv X 3 
        # rest_verts = rest_verts[None].repeat(obj_rot.shape[0], 1, 1)
        # transformed_obj_verts = obj_scale.unsqueeze(-1).unsqueeze(-1) * \
        # obj_rot.bmm(rest_verts.transpose(1, 2)) + obj_com_pos[:, :, None]
        # transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

        rest_verts = rest_verts[None].repeat(obj_rot.shape[0], 1, 1)
        transformed_obj_verts = obj_rot.bmm(rest_verts.transpose(1, 2)) + obj_com_pos[:, :, None]
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

        return transformed_obj_verts 

    def compute_object_geo_bps(self, obj_verts, obj_trans):
        # obj_verts: T X Nv X 3, obj_trans: T X 3
        bps_object_geo = self.bps_torch.encode(x=obj_verts, \
                    feature_type=['deltas'], \
                    custom_basis=self.obj_bps.repeat(obj_trans.shape[0], \
                    1, 1)+obj_trans[:, None, :])['deltas'] # T X N X 3 

        return bps_object_geo

    def prep_bps_data(self):
        n_obj = 1024
        r_obj = 1.0 # Previous 0.6, cannot cover long objects. 
       
        self.bps = torch.load(self.bps_path)

        self.bps_torch = bps_torch()

        self.obj_bps = self.bps['obj'] 

    def generate_data_for_unseen_objects(self):
        unseen_object_geo_folder = os.path.join(self.data_root_folder, "unseen_objects_data/selected_rotated_zeroed_obj_files")
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

        obj_height_json_path = os.path.join(self.data_root_folder, "unseen_objects_data/selected_object_height.json")
        json_data = json.load(open(obj_height_json_path, 'r'))

        self.new_window_data_dict = {} # For unseen obejcts 
        new_cnt = 0
        for k in self.window_data_dict:
            seq_name = self.window_data_dict[k]['seq_name']
            object_name = seq_name.split("_")[1]

            curr_seq_com_pos = torch.from_numpy(self.window_data_dict[k]['window_obj_com_pos']).float() # T X 3  
            curr_seq_com_pos = curr_seq_com_pos[:self.window] # 120 X 3 

            if self.test_long_seq and (seq_name not in self.selected_seq_names):
                continue 

            if self.window_data_dict[k]['start_t_idx'] != 0:
                continue 

            if curr_seq_com_pos.shape[0] < self.window:
                continue 

            if object_name not in unseen_object_corr_dict:
                continue 

            text_json_path = os.path.join(self.language_anno_folder, seq_name+".json")
            if not os.path.exists(text_json_path):
                continue 

            unseen_object_list = unseen_object_corr_dict[object_name]
            for unseen_obj_name in unseen_object_list:
                # For debug 
                # if unseen_obj_name in tmp_debug_visited_object_dict:
                #     continue 

                unseen_obj_geo_path = os.path.join(unseen_object_geo_folder, unseen_obj_name+".ply")
                unseen_obj_mesh = trimesh.load_mesh(unseen_obj_geo_path)
                unseen_obj_verts = unseen_obj_mesh.vertices # Nv X 3 
                unseen_obj_verts = torch.from_numpy(unseen_obj_verts).float() 

                unseen_obj_com_on_floor = np.asarray(json_data[unseen_obj_name]['com']) # 3 
                unseen_obj_com_on_floor = torch.from_numpy(unseen_obj_com_on_floor).float() # 3

                unseen_obj_com_height = torch.zeros(self.window, 1).float() # 120 X 1 
                unseen_obj_com_height[0, 0] = unseen_obj_com_on_floor[2]
                unseen_obj_com_height[-1, 0] = unseen_obj_com_on_floor[2] 
                unseen_obj_com_pos = torch.cat((curr_seq_com_pos[:, :2], unseen_obj_com_height), dim=-1) # 120 X 3 

                # Compute BPS 
                dest_obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name+"_"+unseen_obj_name+".npy")

                if not os.path.exists(dest_obj_bps_npy_path):
                    center_verts = torch.zeros(1, 3).float() 
                    object_bps = self.compute_object_geo_bps(unseen_obj_verts[None], center_verts)
                    np.save(dest_obj_bps_npy_path, object_bps.data.cpu().numpy()) # 1 X 1024 X 3 

                self.new_window_data_dict[new_cnt] = {}
                # Copy some data directly from window_data_dict 
                self.new_window_data_dict[new_cnt]['motion'] = self.window_data_dict[k]['motion']
                self.new_window_data_dict[new_cnt]['betas'] = self.window_data_dict[k]['betas']
                self.new_window_data_dict[new_cnt]['gender'] = self.window_data_dict[k]['gender']
                self.new_window_data_dict[new_cnt]['trans2joint'] = self.window_data_dict[k]['trans2joint']
                self.new_window_data_dict[new_cnt]['rest_human_offsets'] = self.window_data_dict[k]['rest_human_offsets'] 

                self.new_window_data_dict[new_cnt]['seq_name'] = seq_name 

                # Add unseen object's data 
                self.new_window_data_dict[new_cnt]['obj_name'] = unseen_obj_name  
                self.new_window_data_dict[new_cnt]['window_obj_com_pos'] = unseen_obj_com_pos 
                self.new_window_data_dict[new_cnt]['obj_rot_mat'] = torch.eye(3)[None].repeat(self.window, 1, 1) # W X 3 X 3 

                new_cnt += 1 

                # For debug 
                # tmp_debug_visited_object_dict[unseen_obj_name] = 1 

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

    def normalize_obj_pos_min_max(self, ori_obj_pos):
        # ori_jpos: T X 3 
        if ori_obj_pos.dim() == 3: # BS X T X 3 
            normalized_jpos = (ori_obj_pos - self.obj_pos_min.to(ori_obj_pos.device)[None])/(self.obj_pos_max.to(ori_obj_pos.device)[None] \
            -self.obj_pos_min.to(ori_obj_pos.device)[None])
        else:
            normalized_jpos = (ori_obj_pos - self.obj_pos_min.to(ori_obj_pos.device))/(self.obj_pos_max.to(ori_obj_pos.device)\
            -self.obj_pos_min.to(ori_obj_pos.device))

        normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 

        return normalized_jpos # T X 3 /BS X T X 3

    def de_normalize_obj_pos_min_max(self, normalized_obj_pos):
        normalized_obj_pos = (normalized_obj_pos + 1) * 0.5 # [0, 1] range
        if normalized_obj_pos.dim() == 3:
            de_jpos = normalized_obj_pos * (self.obj_pos_max.to(normalized_obj_pos.device)[None]-\
            self.obj_pos_min.to(normalized_obj_pos.device)[None]) + self.obj_pos_min.to(normalized_obj_pos.device)[None]
        else:
            de_jpos = normalized_obj_pos * (self.obj_pos_max.to(normalized_obj_pos.device)-\
            self.obj_pos_min.to(normalized_obj_pos.device)) + self.obj_pos_min.to(normalized_obj_pos.device)

        return de_jpos # T X 3

    def __len__(self):
        return len(self.new_window_data_dict)

    def load_rest_pose_object_geometry(self, object_name):
        rest_obj_path = os.path.join(self.rest_object_geo_folder, object_name+".ply")
        
        mesh = trimesh.load_mesh(rest_obj_path)
        rest_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3

        return rest_verts, obj_mesh_faces 

    def load_language_annotation(self, seq_name):
        # seq_name: sub16_clothesstand_000, etc. 
        json_path = os.path.join(self.language_anno_folder, seq_name+".json")
        json_data = json.load(open(json_path, 'r'))
        
        text_anno = json_data[seq_name]

        return text_anno 
    
    def prep_rel_obj_rot_mat(self, obj_rot_mat):
        # obj_rot_mat: T X 3 X 3 
        if obj_rot_mat.dim() == 4:
            timesteps = obj_rot_mat.shape[1]

            init_obj_rot_mat = obj_rot_mat[:, 0:1].repeat(1, timesteps, 1, 1) # BS X T X 3 X 3
            rel_rot_mat = torch.matmul(obj_rot_mat, init_obj_rot_mat.transpose(2, 3)) # BS X T X 3 X 3
        else:
            timesteps = obj_rot_mat.shape[0]

            # Compute relative rotation matrix with respect to the first frame's object geometry. 
            init_obj_rot_mat = obj_rot_mat[0:1].repeat(timesteps, 1, 1) # T X 3 X 3
            rel_rot_mat = torch.matmul(obj_rot_mat, init_obj_rot_mat.transpose(1, 2)) # T X 3 X 3

        return rel_rot_mat 
    
    def prep_rel_obj_rot_mat_w_reference_mat(self, obj_rot_mat, ref_rot_mat):
        # obj_rot_mat: T X 3 X 3 / BS X T X 3 X 3 
        # ref_rot_mat: BS X 1 X 3 X 3/ 1 X 3 X 3 
        if obj_rot_mat.dim() == 4:
            timesteps = obj_rot_mat.shape[1]

            init_obj_rot_mat = ref_rot_mat.repeat(1, timesteps, 1, 1) # BS X T X 3 X 3
            rel_rot_mat = torch.matmul(obj_rot_mat, init_obj_rot_mat.transpose(2, 3)) # BS X T X 3 X 3
        else:
            timesteps = obj_rot_mat.shape[0]

            # Compute relative rotation matrix with respect to the first frame's object geometry. 
            init_obj_rot_mat = ref_rot_mat.repeat(timesteps, 1, 1) # T X 3 X 3
            rel_rot_mat = torch.matmul(obj_rot_mat, init_obj_rot_mat.transpose(1, 2)) # T X 3 X 3

        return rel_rot_mat 

    def rel_rot_to_seq(self, rel_rot_mat, obj_rot_mat):
        # rel_rot_mat: BS X T X 3 X 3 
        # obj_rot_mat: BS X T X 3 X 3 (only use the first frame's rotation)
        timesteps = rel_rot_mat.shape[1]

        # Compute relative rotation matrix with respect to the first frame's object geometry. 
        init_obj_rot_mat = obj_rot_mat[:, 0:1].repeat(1, timesteps, 1, 1) # BS X T X 3 X 3
        obj_rot_mat = torch.matmul(rel_rot_mat, init_obj_rot_mat.to(rel_rot_mat.device)) 

        return obj_rot_mat 

    def __getitem__(self, index):
        # index = 0 # For debug 
        data_input = self.new_window_data_dict[index]['motion']
        data_input = torch.from_numpy(data_input).float()

        seq_name = self.new_window_data_dict[index]['seq_name'] 
        object_name = self.new_window_data_dict[index]['obj_name']
        
        trans2joint = self.new_window_data_dict[index]['trans2joint'] 

        rest_human_offsets = self.new_window_data_dict[index]['rest_human_offsets'] 
            
        obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name+"_"+object_name+".npy") 
      
        obj_bps_data = np.load(obj_bps_npy_path) # T X N X 3 
        obj_bps_data = torch.from_numpy(obj_bps_data) 

        obj_com_pos = self.new_window_data_dict[index]['window_obj_com_pos']
      
        normalized_obj_com_pos = self.normalize_obj_pos_min_max(obj_com_pos)

        # Prepare object motion information
        window_obj_rot_mat = self.new_window_data_dict[index]['obj_rot_mat']

        # Prepare relative rotation 
        window_rel_obj_rot_mat = self.prep_rel_obj_rot_mat(window_obj_rot_mat)

        num_joints = 24         
        normalized_jpos = self.normalize_jpos_min_max(data_input[:, :num_joints*3].reshape(-1, num_joints, 3)) # T X 22 X 3 
       
        global_joint_rot = data_input[:, 2*num_joints*3:] # T X (22*6)

        new_data_input = torch.cat((normalized_jpos.reshape(-1, num_joints*3), global_joint_rot), dim=1)
        ori_data_input = torch.cat((data_input[:, :num_joints*3], global_joint_rot), dim=1)
    
        # Add padding. 
        actual_steps = new_data_input.shape[0]
        
        paded_new_data_input = new_data_input 
        paded_ori_data_input = ori_data_input 

        paded_normalized_obj_com_pos = normalized_obj_com_pos
        paded_obj_com_pos = obj_com_pos 
        
        paded_obj_rot_mat = window_obj_rot_mat

        paded_rel_obj_rot_mat = window_rel_obj_rot_mat 

        data_input_dict = {}
        data_input_dict['motion'] = paded_new_data_input
        data_input_dict['ori_motion'] = paded_ori_data_input 
 
        data_input_dict['ori_obj_motion'] = torch.cat((paded_obj_com_pos, \
                                        paded_rel_obj_rot_mat.reshape(-1, 9)), dim=-1) # T X (3+9)
        data_input_dict['obj_motion'] = torch.cat((paded_normalized_obj_com_pos, \
                                            paded_rel_obj_rot_mat.reshape(-1, 9)), dim=-1) # T X (3+9)

        data_input_dict['input_obj_bps'] = obj_bps_data[0:1] # 1 X 1024 X 3 

        reference_obj_rot_mat = window_obj_rot_mat[0:1] 

        data_input_dict['obj_rot_mat'] = paded_obj_rot_mat # T X 3 X 3 
        data_input_dict['obj_com_pos'] = paded_obj_com_pos 
       
        data_input_dict['seq_name'] = seq_name
        data_input_dict['obj_name'] = object_name 

        data_input_dict['seq_len'] = actual_steps 

        if self.input_language_condition:
            # Load language annotation 
            seq_text_anno = self.load_language_annotation(seq_name) 
            data_input_dict['text'] = seq_text_anno # a string 
        
        # Use the same body shape for now! 
        data_input_dict['betas'] = self.new_window_data_dict[0]['betas']
        data_input_dict['gender'] = str(self.new_window_data_dict[0]['gender'])
        data_input_dict['trans2joint'] = self.new_window_data_dict[0]['trans2joint'] 
        data_input_dict['rest_human_offsets'] = self.new_window_data_dict[0]['rest_human_offsets'] 

        data_input_dict['reference_obj_rot_mat']= reference_obj_rot_mat

        data_input_dict['s_idx'] = 0
        data_input_dict['e_idx'] = 120 

        return data_input_dict 
        # data_input_dict['motion']: T X (22*3+22*6) range [-1, 1]
        # data_input_dict['obj_bps]: T X N X 3 
