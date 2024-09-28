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


class LongCanoObjectTrajDataset(Dataset):
    def __init__(
        self,
        train,
        data_root_folder,
        window=120,
        use_object_splits=False,
        input_language_condition=False,
        use_first_frame_bps=False, 
        use_random_frame_bps=False, 
        test_object_name="largebox",
    ):
        
        self.window = window 

        self.use_object_splits = use_object_splits 
        self.train_objects = ["largetable", "woodchair", "plasticbox", "largebox", "smallbox", "trashcan", "monitor", \
                    "floorlamp", "clothesstand"] 
        self.test_objects = ["smalltable", "whitechair", "suitcase", "tripod"]

        object2seq_dict = {"clothesstand": "sub16_clothesstand_004", "largebox": "sub16_largebox_005", \
                "largetable": "sub16_largetable_006", "plasticbox": "sub16_plasticbox_011", "trashcan": "sub16_trashcan_000", \
                "whitechair": "sub16_whitechair_016", "floorlamp": "sub17_floorlamp_001", "monitor": "sub17_monitor_011", \
                "smallbox": "sub17_smallbox_018", "smalltable": "sub17_smalltable_034", "suitcase": "sub17_suitcase_006", \
                "tripod": "sub17_tripod_009", "woodchair": "sub17_woodchair_035"}

        if test_object_name == "all":
            self.selected_seq_names = []
            for k in object2seq_dict:
                self.selected_seq_names.append(object2seq_dict[k]) 
        else:
            self.selected_seq_names = [object2seq_dict[test_object_name]] 

        self.input_language_condition = input_language_condition 

        self.use_first_frame_bps = use_first_frame_bps 

        self.use_random_frame_bps = use_random_frame_bps 

        self.parents = get_smpl_parents() # 24/22 

        self.data_root_folder = data_root_folder 
        self.obj_geo_root_folder = os.path.join(self.data_root_folder, "captured_objects")
        
        self.rest_object_geo_folder = os.path.join(self.data_root_folder, "rest_object_geo")
        if not os.path.exists(self.rest_object_geo_folder):
            os.makedirs(self.rest_object_geo_folder)

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

        dest_obj_bps_npy_folder_for_test = os.path.join(self.data_root_folder, \
                        "whole_seq_object_bps_npy_files_for_test_joints24"+"_"+test_object_name)

        if not os.path.exists(dest_obj_bps_npy_folder_for_test):
            os.makedirs(dest_obj_bps_npy_folder_for_test)

        self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder_for_test 

        seq_data_path = os.path.join(data_root_folder, "test_diffusion_manip_seq_joints24.p")
        processed_data_path = os.path.join(data_root_folder, "whole_seq_test_diffusion_manip_window_joints24_"+test_object_name+".p")

        min_max_mean_std_data_path = os.path.join(data_root_folder, \
                            "cano_min_max_mean_std_data_window_"+str(self.window)+"_joints24.p")
       
        self.prep_bps_data()

        if os.path.exists(processed_data_path):
            self.window_data_dict = joblib.load(processed_data_path)

            # if not self.train:
                # Mannually enable this. For testing data (discarded some testing sequences)
                # self.get_bps_from_window_data_dict()
        else:
            self.data_dict = joblib.load(seq_data_path)
            
            self.extract_rest_pose_object_geometry_and_rotation() 

            self.cal_normalize_data_input()
            joblib.dump(self.window_data_dict, processed_data_path)            
       
        min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
           
        self.global_jpos_min = torch.from_numpy(min_max_mean_std_jpos_data['global_jpos_min']).float().reshape(24, 3)[None]
        self.global_jpos_max = torch.from_numpy(min_max_mean_std_jpos_data['global_jpos_max']).float().reshape(24, 3)[None]

        self.obj_pos_min = torch.from_numpy(min_max_mean_std_jpos_data['obj_com_pos_min']).float().reshape(1, 3)
        self.obj_pos_max = torch.from_numpy(min_max_mean_std_jpos_data['obj_com_pos_max']).float().reshape(1, 3)

        if self.use_object_splits:
            self.window_data_dict = self.filter_out_object_split()

        if self.input_language_condition:
            self.window_data_dict = self.filter_out_seq_wo_text() 

        # Get train and validation statistics. 
        print("Total number of windows for validation:{0}".format(len(self.window_data_dict)))

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

    def load_language_annotation(self, seq_name):
        # seq_name: sub16_clothesstand_000, etc. 
        json_path = os.path.join(self.language_anno_folder, seq_name+".json")
        json_data = json.load(open(json_path, 'r'))
        
        text_anno = json_data[seq_name]

        return text_anno 

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
            text_json_path = os.path.join(self.language_anno_folder, seq_name+".json")
            if os.path.exists(text_json_path):
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                if "ori_w_idx" in self.window_data_dict[k]: # Based on filtered results split by objects. 
                    new_window_data_dict[new_cnt]['ori_w_idx'] = self.window_data_dict[k]['ori_w_idx']
                else: # Based on the original window_daia_dict. 
                    new_window_data_dict[new_cnt]['ori_w_idx'] = k 
                new_cnt += 1

        return new_window_data_dict

    def apply_transformation_to_obj_geometry(self, obj_mesh_path, obj_scale, obj_rot, obj_trans):
        mesh = trimesh.load_mesh(obj_mesh_path)
        obj_mesh_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3 

        ori_obj_verts = torch.from_numpy(obj_mesh_verts).float()[None].repeat(obj_trans.shape[0], 1, 1) # T X Nv X 3 
    
        if torch.is_tensor(obj_scale):
            seq_scale = obj_scale.float() 
        else:
            seq_scale = torch.from_numpy(obj_scale).float() # T 
        
        if torch.is_tensor(obj_rot):
            seq_rot_mat = obj_rot.float()
        else:
            seq_rot_mat = torch.from_numpy(obj_rot).float() # T X 3 X 3 
        
        if obj_trans.shape[-1] != 1:
            if torch.is_tensor(obj_trans):
                seq_trans = obj_trans.float()[:, :, None]
            else:
                seq_trans = torch.from_numpy(obj_trans).float()[:, :, None] # T X 3 X 1 
        else:
            if torch.is_tensor(obj_trans):
                seq_trans = obj_trans.float()
            else:
                seq_trans = torch.from_numpy(obj_trans).float() # T X 3 X 1 

        transformed_obj_verts = seq_scale.unsqueeze(-1).unsqueeze(-1) * \
        seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2).to(seq_trans.device)) + seq_trans
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

        return transformed_obj_verts, obj_mesh_faces  

    def convert_rest_pose_obj_geometry(self, object_name, obj_scale, obj_trans, obj_rot):
        # obj_scale: T, obj_trans: T X 3, obj_rot: T X 3 X 3
        # obj_mesh_verts: T X Nv X 3
        rest_obj_path = os.path.join(self.res_object_geo_folder, object_name+".ply")

        if os.path.exists(rest_obj_path):
            mesh = trimesh.load_mesh(rest_obj_path)
            rest_verts = np.asarray(mesh.vertices) # Nv X 3
            obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3

            rest_verts = torch.from_numpy(rest_verts).to(obj_scale.device)  
        else:
            obj_mesh_verts, obj_mesh_faces = self.load_object_geometry(object_name, obj_scale, obj_trans, obj_rot)
            com_pos = obj_mesh_verts[0].mean(dim=0)[None] # 1 X 3 
            tmp_verts = obj_mesh_verts[0] - com_pos # Nv X 3 
            tmp_verts = tmp_verts.to(obj_scale.device)

            rest_verts = torch.matmul(obj_rot[0:1].repeat(tmp_verts.shape[0], 1, 1).transpose(1, 2), \
                    tmp_verts[:, :, None]) # Nv X 3 X 1
            rest_verts = rest_verts.squeeze(-1) # Nv X 3 

            dest_mesh = trimesh.Trimesh(
            vertices=rest_verts.data.cpu().numpy(),
            faces=obj_mesh_faces,
            process=False)

            result = trimesh.exchange.ply.export_ply(dest_mesh, encoding='ascii')
            output_file = open(rest_obj_path, "wb+")
            output_file.write(result)
            output_file.close()

        return rest_verts, obj_mesh_faces 

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
    
    def load_object_geometry(self, object_name, obj_scale, obj_trans, obj_rot, \
        obj_bottom_scale=None, obj_bottom_trans=None, obj_bottom_rot=None):
        obj_mesh_path = os.path.join(self.obj_geo_root_folder, object_name+"_cleaned_simplified.obj")
        obj_mesh_verts, obj_mesh_faces =self.apply_transformation_to_obj_geometry(obj_mesh_path, \
        obj_scale, obj_rot, obj_trans) # T X Nv X 3 

        return obj_mesh_verts, obj_mesh_faces 

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

    def convert_rest_pose_obj_geometry(self, object_name, obj_scale, obj_trans, obj_rot):
        # obj_scale: T, obj_trans: T X 3, obj_rot: T X 3 X 3
        # obj_mesh_verts: T X Nv X 3
        rest_obj_path = os.path.join(self.rest_object_geo_folder, object_name+".ply")
        rest_obj_json_path = os.path.join(self.rest_object_geo_folder, object_name+".json")

        if os.path.exists(rest_obj_path):
            mesh = trimesh.load_mesh(rest_obj_path)
            rest_verts = np.asarray(mesh.vertices) # Nv X 3
            obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3

            rest_verts = torch.from_numpy(rest_verts) 

            json_data = json.load(open(rest_obj_json_path, 'r'))
            rest_pose_ori_obj_rot = np.asarray(json_data['rest_pose_ori_obj_rot']) # 3 X 3 
            rest_pose_ori_obj_com_pos = np.asarray(json_data['rest_pose_ori_com_pos']) # 1 X 3 
            obj_trans_to_com_pos = np.asarray(json_data['obj_trans_to_com_pos']) # 1 X 3 

        return rest_verts, obj_mesh_faces, rest_pose_ori_obj_rot, rest_pose_ori_obj_com_pos, obj_trans_to_com_pos  

    def extract_rest_pose_object_geometry_and_rotation(self):
        self.rest_pose_object_dict = {} 

        for seq_idx in self.data_dict:
            seq_name = self.data_dict[seq_idx]['seq_name']
            object_name = seq_name.split("_")[1]
            if object_name in ["vacuum", "mop"]:
                continue 

            if object_name not in self.rest_pose_object_dict:
                obj_trans = self.data_dict[seq_idx]['obj_trans'][:, :, 0] # T X 3
                obj_rot = self.data_dict[seq_idx]['obj_rot'] # T X 3 X 3 
                obj_scale = self.data_dict[seq_idx]['obj_scale'] # T  

                rest_verts, obj_mesh_faces, rest_pose_ori_rot, rest_pose_ori_com_pos, obj_trans_to_com_pos = \
                self.convert_rest_pose_obj_geometry(object_name, obj_scale, obj_trans, obj_rot)

                self.rest_pose_object_dict[object_name] = {}
                self.rest_pose_object_dict[object_name]['ori_rotation'] = rest_pose_ori_rot # 3 X 3 
                self.rest_pose_object_dict[object_name]['ori_trans'] = rest_pose_ori_com_pos # 1 X 3 
                self.rest_pose_object_dict[object_name]['obj_trans_to_com_pos'] = obj_trans_to_com_pos # 1 X 3 

    def cal_normalize_data_input(self):
        self.window_data_dict = {}
        s_idx = 0 
        for index in self.data_dict:
            seq_name = self.data_dict[index]['seq_name']

            if seq_name not in self.selected_seq_names:
                continue 
            
            object_name = seq_name.split("_")[1]

            # Skip vacuum, mop for now since they consist of two object parts. 
            if object_name in ["vacuum", "mop"]:
                continue 

            rest_pose_obj_data = self.rest_pose_object_dict[object_name]
            rest_pose_rot_mat = rest_pose_obj_data['ori_rotation'] # 3 X 3

            rest_obj_path = os.path.join(self.rest_object_geo_folder, object_name+".ply")
            mesh = trimesh.load_mesh(rest_obj_path)
            rest_verts = np.asarray(mesh.vertices) # Nv X 3
            obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3
            rest_verts = torch.from_numpy(rest_verts).float() # Nv X 3

            betas = self.data_dict[index]['betas'] # 1 X 16 
            gender = self.data_dict[index]['gender']

            seq_root_trans = self.data_dict[index]['trans'] # T X 3 
            seq_root_orient = self.data_dict[index]['root_orient'] # T X 3 
            seq_pose_body = self.data_dict[index]['pose_body'].reshape(-1, 21, 3) # T X 21 X 3

            rest_human_offsets = self.data_dict[index]['rest_offsets'] # 22 X 3/24 X 3
            trans2joint = self.data_dict[index]['trans2joint'] # 3 

            # Used in old version without defining rest object geometry. 
            seq_obj_trans = self.data_dict[index]['obj_trans'][:, :, 0] # T X 3
            seq_obj_rot = self.data_dict[index]['obj_rot'] # T X 3 X 3 
            seq_obj_scale = self.data_dict[index]['obj_scale'] # T  

            seq_obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, seq_obj_scale, \
                        seq_obj_trans, seq_obj_rot) # T X Nv X 3, tensor
            seq_obj_com_pos = seq_obj_verts.mean(dim=1) # T X 3 

            obj_trans = seq_obj_com_pos.clone().detach().cpu().numpy() 

            rest_pose_rot_mat_rep = torch.from_numpy(rest_pose_rot_mat).float()[None, :, :] # 1 X 3 X 3 
            obj_rot = torch.from_numpy(self.data_dict[index]['obj_rot']) # T X 3 X 3 
            obj_rot = torch.matmul(obj_rot, rest_pose_rot_mat_rep.repeat(obj_rot.shape[0], 1, 1).transpose(1, 2)) # T X 3 X 3  
            obj_rot = obj_rot.detach().cpu().numpy() 
           
            num_steps = seq_root_trans.shape[0]

            start_t_idx = 0
            end_t_idx = num_steps - 1 

            self.window_data_dict[s_idx] = {}

            joint_aa_rep = torch.cat((torch.from_numpy(seq_root_orient[start_t_idx:end_t_idx+1]).float()[:, None, :], \
                torch.from_numpy(seq_pose_body[start_t_idx:end_t_idx+1]).float()), dim=1) # T X J X 3 
            X = torch.from_numpy(rest_human_offsets).float()[None].repeat(joint_aa_rep.shape[0], 1, 1).detach().cpu().numpy() # T X J X 3 
            X[:, 0, :] = seq_root_trans[start_t_idx:end_t_idx+1] 
            local_rot_mat = transforms.axis_angle_to_matrix(joint_aa_rep) # T X J X 3 X 3 
            Q = transforms.matrix_to_quaternion(local_rot_mat).detach().cpu().numpy() # T X J X 4 

            obj_x = obj_trans[start_t_idx:end_t_idx+1].copy() # T X 3 
            obj_rot_mat = torch.from_numpy(obj_rot[start_t_idx:end_t_idx+1]).float()# T X 3 X 3 
            obj_q = transforms.matrix_to_quaternion(obj_rot_mat).detach().cpu().numpy() # T X 4 

            # Canonicalize based on the first human pose's orientation. 
            X, Q, new_obj_x, new_obj_q = rotate_at_frame_w_obj(X[np.newaxis], Q[np.newaxis], \
            obj_x[np.newaxis], obj_q[np.newaxis], \
            trans2joint[np.newaxis], self.parents, n_past=1, floor_z=True)
            # 1 X T X J X 3, 1 X T X J X 4, 1 X T X 3, 1 X T X 4 

            new_seq_root_trans = X[0, :, 0, :] # T X 3 
            new_local_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(Q[0]).float()) # T X J X 3 X 3 
            new_local_aa_rep = transforms.matrix_to_axis_angle(new_local_rot_mat) # T X J X 3 
            new_seq_root_orient = new_local_aa_rep[:, 0, :] # T X 3
            new_seq_pose_body = new_local_aa_rep[:, 1:, :] # T X 21 X 3 
            
            new_obj_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(new_obj_q[0]).float()) # T X 3 X 3 \
            
            cano_obj_mat = torch.matmul(new_obj_rot_mat[0], obj_rot_mat[0].transpose(0, 1)) # 3 X 3 

            obj_verts = self.load_object_geometry_w_rest_geo(new_obj_rot_mat, \
                        torch.from_numpy(new_obj_x[0]).float().to(new_obj_rot_mat.device), rest_verts)

            center_verts = obj_verts.mean(dim=1) # T X 3 

            query = self.process_window_data(rest_human_offsets, trans2joint, \
                new_seq_root_trans, new_seq_root_orient.detach().cpu().numpy(), \
                new_seq_pose_body.detach().cpu().numpy(),  \
                new_obj_x[0], new_obj_rot_mat.detach().cpu().numpy(), center_verts)

            # Compute BPS representation for this window
            # Save to numpy file 
            dest_obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name+"_"+str(s_idx)+".npy")

            if not os.path.exists(dest_obj_bps_npy_path):
                object_bps = self.compute_object_geo_bps(obj_verts[0:1], center_verts[0:1])
                np.save(dest_obj_bps_npy_path, object_bps.data.cpu().numpy()) # 1 X 1024 X 3 

            curr_global_jpos = query['global_jpos'].detach().cpu().numpy()
            curr_global_jvel = query['global_jvel'].detach().cpu().numpy()
            curr_global_rot_6d = query['global_rot_6d'].detach().cpu().numpy()

            self.window_data_dict[s_idx]['cano_obj_mat'] = cano_obj_mat.detach().cpu().numpy() 

            self.window_data_dict[s_idx]['motion'] = np.concatenate((curr_global_jpos.reshape(-1, 24*3), \
            curr_global_jvel.reshape(-1, 24*3), curr_global_rot_6d.reshape(-1, 22*6)), axis=1) # T X (24*3+24*3+22*6)
            
            self.window_data_dict[s_idx]['seq_name'] = seq_name
            self.window_data_dict[s_idx]['start_t_idx'] = start_t_idx
            self.window_data_dict[s_idx]['end_t_idx'] = end_t_idx 

            self.window_data_dict[s_idx]['betas'] = betas 
            self.window_data_dict[s_idx]['gender'] = gender

            self.window_data_dict[s_idx]['trans2joint'] = trans2joint 

            self.window_data_dict[s_idx]['obj_rot_mat'] = query['obj_rot_mat'].detach().cpu().numpy()

            self.window_data_dict[s_idx]['window_obj_com_pos'] = query['window_obj_com_pos'].detach().cpu().numpy() 

            self.window_data_dict[s_idx]['rest_human_offsets'] = rest_human_offsets 

            s_idx += 1 

            # if s_idx > 32:
            #     break 

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

    def process_window_data(self, rest_human_offsets, trans2joint, seq_root_trans, seq_root_orient, seq_pose_body, \
        obj_trans, obj_rot, center_verts):
        random_t_idx = 0 
        end_t_idx = seq_root_trans.shape[0] - 1

        window_root_trans = torch.from_numpy(seq_root_trans[random_t_idx:end_t_idx+1]).cuda()
        window_root_orient = torch.from_numpy(seq_root_orient[random_t_idx:end_t_idx+1]).float().cuda()
        window_pose_body  = torch.from_numpy(seq_pose_body[random_t_idx:end_t_idx+1]).float().cuda()

        # window_obj_scale = torch.from_numpy(obj_scale[random_t_idx:end_t_idx+1]).float().cuda() # T
        window_obj_rot_mat = torch.from_numpy(obj_rot[random_t_idx:end_t_idx+1]).float().cuda() # T X 3 X 3 
        window_obj_trans = torch.from_numpy(obj_trans[random_t_idx:end_t_idx+1]).float().cuda() # T X 3

        window_center_verts = center_verts[random_t_idx:end_t_idx+1].to(window_obj_trans.device)

        # Move thr first frame's human position to zero. 
        move_to_zero_trans = window_root_trans[0:1, :].clone() # 1 X 3 
        move_to_zero_trans[:, 2] = 0 

        # Move motion and object translation to make the initial pose trans 0. 
        window_root_trans = window_root_trans - move_to_zero_trans 
        window_obj_trans = window_obj_trans - move_to_zero_trans 
        window_center_verts = window_center_verts - move_to_zero_trans 

        window_root_rot_mat = transforms.axis_angle_to_matrix(window_root_orient) # T' X 3 X 3 
        window_pose_rot_mat = transforms.axis_angle_to_matrix(window_pose_body) # T' X 21 X 3 X 3 

        # Generate global joint rotation 
        local_joint_rot_mat = torch.cat((window_root_rot_mat[:, None, :, :], window_pose_rot_mat), dim=1) # T' X 22 X 3 X 3 
        global_joint_rot_mat = local2global_pose(local_joint_rot_mat) # T' X 22 X 3 X 3 

        curr_seq_pose_aa = torch.cat((window_root_orient[:, None, :], window_pose_body), dim=1) # T' X 22 X 3/T' X 24 X 3 
        rest_human_offsets = torch.from_numpy(rest_human_offsets).float()[None] 
        curr_seq_local_jpos = rest_human_offsets.repeat(curr_seq_pose_aa.shape[0], 1, 1).cuda() # T' X 22 X 3/T' X 24 X 3  
        curr_seq_local_jpos[:, 0, :] = window_root_trans - torch.from_numpy(trans2joint).cuda()[None] # T' X 22/24 X 3 

        local_joint_rot_mat = transforms.axis_angle_to_matrix(curr_seq_pose_aa)
        _, human_jnts = quat_fk_torch(local_joint_rot_mat, curr_seq_local_jpos)

        global_jpos = human_jnts # T' X 22/24 X 3 
        global_jvel = global_jpos[1:] - global_jpos[:-1] # (T'-1) X 22/24 X 3 

        global_joint_rot_mat = local2global_pose(local_joint_rot_mat) # T' X 22 X 3 X 3 

        local_rot_6d = transforms.matrix_to_rotation_6d(local_joint_rot_mat)
        global_rot_6d = transforms.matrix_to_rotation_6d(global_joint_rot_mat)

        query = {}

        query['local_rot_mat'] = local_joint_rot_mat # T' X 22 X 3 X 3 
        query['local_rot_6d'] = local_rot_6d # T' X 22 X 6

        query['global_jpos'] = global_jpos # T' X 22/24 X 3 
        query['global_jvel'] = torch.cat((global_jvel, \
            torch.zeros(1, global_jvel.shape[1], 3).to(global_jvel.device)), dim=0) # T' X 22/24 X 3 
        
        query['global_rot_mat'] = global_joint_rot_mat # T' X 22 X 3 X 3 
        query['global_rot_6d'] = global_rot_6d # T' X 22 X 6

        query['obj_trans'] = window_obj_trans # T' X 3 
        query['obj_rot_mat'] = window_obj_rot_mat # T' X 3 X 3 

        query['window_obj_com_pos'] = window_center_verts # T X 3 

        return query 

    def __len__(self):
        return len(self.window_data_dict)
    
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
        data_input = self.window_data_dict[index]['motion']
        data_input = torch.from_numpy(data_input).float()

        seq_name = self.window_data_dict[index]['seq_name'] 
        object_name = seq_name.split("_")[1]
        
        window_s_idx = self.window_data_dict[index]['start_t_idx']
        window_e_idx = self.window_data_dict[index]['end_t_idx']

        trans2joint = self.window_data_dict[index]['trans2joint'] 

        rest_human_offsets = self.window_data_dict[index]['rest_human_offsets'] 

        if self.use_first_frame_bps or self.use_random_frame_bps:
            if self.use_object_splits or self.input_language_condition:
                ori_w_idx = self.window_data_dict[index]['ori_w_idx']
                obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name+"_"+str(ori_w_idx)+".npy") 
            else:
                obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name+"_"+str(index)+".npy") 
        else:
            obj_bps_npy_path = os.path.join(self.rest_object_geo_folder, object_name+".npy")

        obj_bps_data = np.load(obj_bps_npy_path) # T X N X 3 
        obj_bps_data = torch.from_numpy(obj_bps_data) 

        obj_com_pos = torch.from_numpy(self.window_data_dict[index]['window_obj_com_pos']).float()
      
        normalized_obj_com_pos = self.normalize_obj_pos_min_max(obj_com_pos)

        # Prepare object motion information
        window_obj_rot_mat = torch.from_numpy(self.window_data_dict[index]['obj_rot_mat']).float()

        # Prepare relative rotation 
        window_rel_obj_rot_mat = self.prep_rel_obj_rot_mat(window_obj_rot_mat)

        # Prepare object keypoints for each frame. 
        
        # Load rest pose BPS and compute nn points on the object. 
        rest_obj_bps_npy_path = os.path.join(self.rest_object_geo_folder, object_name+".npy")
        rest_obj_bps_data = np.load(rest_obj_bps_npy_path) # 1 X 1024 X 3 
        nn_pts_on_mesh = self.obj_bps + torch.from_numpy(rest_obj_bps_data).float().to(self.obj_bps.device) # 1 X 1024 X 3 
        nn_pts_on_mesh = nn_pts_on_mesh.squeeze(0) # 1024 X 3 

        # Random sample 100 points used for training
        # sampled_vidxs = random.sample(list(range(1024)), 100) 
        # sampled_nn_pts_on_mesh = nn_pts_on_mesh[sampled_vidxs] # K X 3 

        # During inference, use all 1024 points for penetration loss, contact loss? 
        sampled_nn_pts_on_mesh = nn_pts_on_mesh # K X 3 

        rest_pose_obj_nn_pts = sampled_nn_pts_on_mesh.clone() 

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
        paded_obj_com_pos = torch.from_numpy(self.window_data_dict[index]['window_obj_com_pos']).float()
        
        paded_obj_rot_mat = window_obj_rot_mat

        paded_rel_obj_rot_mat = window_rel_obj_rot_mat 

        data_input_dict = {}
        data_input_dict['motion'] = paded_new_data_input
        data_input_dict['ori_motion'] = paded_ori_data_input 
 
        if self.use_first_frame_bps or self.use_random_frame_bps:
            data_input_dict['ori_obj_motion'] = torch.cat((paded_obj_com_pos, \
                                            paded_rel_obj_rot_mat.reshape(-1, 9)), dim=-1) # T X (3+9)
            data_input_dict['obj_motion'] = torch.cat((paded_normalized_obj_com_pos, \
                                                paded_rel_obj_rot_mat.reshape(-1, 9)), dim=-1) # T X (3+9)

            data_input_dict['input_obj_bps'] = obj_bps_data[0:1] # 1 X 1024 X 3 

            reference_obj_rot_mat = window_obj_rot_mat[0:1] 
        else:
            data_input_dict['ori_obj_motion'] = torch.cat((paded_obj_com_pos, \
                                                paded_obj_rot_mat.reshape(-1, 9)), dim=-1) # T X (3+9)
            data_input_dict['obj_motion'] = torch.cat((paded_normalized_obj_com_pos, \
                                                paded_obj_rot_mat.reshape(-1, 9)), dim=-1) # T X (3+9)
            data_input_dict['input_obj_bps'] = obj_bps_data[0:1] # 1 X 1024 X 3 

        data_input_dict['obj_rot_mat'] = paded_obj_rot_mat # T X 3 X 3 
        data_input_dict['obj_com_pos'] = paded_obj_com_pos 
       
        data_input_dict['seq_name'] = seq_name
        data_input_dict['obj_name'] = seq_name.split("_")[1]

        data_input_dict['seq_len'] = actual_steps 

        if self.input_language_condition:
            # Load language annotation 
            seq_text_anno = self.load_language_annotation(seq_name) 
            data_input_dict['text'] = seq_text_anno # a string 
        
        # Use the same body shape for now! 
        data_input_dict['betas'] = self.window_data_dict[0]['betas']
        data_input_dict['gender'] = str(self.window_data_dict[0]['gender'])
        data_input_dict['trans2joint'] = self.window_data_dict[0]['trans2joint'] 
        data_input_dict['rest_human_offsets'] = self.window_data_dict[0]['rest_human_offsets'] 

        data_input_dict['reference_obj_rot_mat']= reference_obj_rot_mat

        data_input_dict['rest_pose_obj_pts'] = rest_pose_obj_nn_pts # K X 3 

        return data_input_dict 
        # data_input_dict['motion']: T X (22*3+22*6) range [-1, 1]
        # data_input_dict['obj_bps]: T X N X 3 
