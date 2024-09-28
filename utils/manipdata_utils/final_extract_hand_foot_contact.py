import os 
import numpy as np 
import trimesh 
import torch 

import pickle as pkl

from human_body_prior.body_model.body_model import BodyModel

import igl 
from vis_all_captured_motion_and_object import get_contact_labels, run_smplx_model

def extract_part_vert_idx(part_obj_path, full_obj_path): 
    part_mesh = trimesh.load_mesh(part_obj_path)
    part_verts = np.asarray(part_mesh.vertices) # Nv X 3
    part_faces = np.asarray(part_mesh.faces) # Nf X 3 
    
    full_mesh = trimesh.load_mesh(full_obj_path) 
    full_verts = np.asarray(full_mesh.vertices) # Nv' X 3 
    full_faces = np.asarray(full_mesh.faces)

    contact_part_points, contact_labels, contact_full_verts = \
    get_contact_labels(part_verts, full_verts, full_faces)
    # Nv X 3, Nv, Nv X 3 

    # # For extracting human part vertex idxs that are in contact 
    human_c_idx_list = []
    for c_idx in range(contact_labels.shape[0]):
        if contact_labels[c_idx]:
            curr_contact_human_vt = torch.from_numpy(contact_full_verts[c_idx]).float()[None].repeat(full_verts.shape[0], 1)
            tmp_full_verts = torch.from_numpy(full_verts)
            vert_dist = ((tmp_full_verts - curr_contact_human_vt)**2).sum(dim=1) 
            
            human_c_idx = vert_dist.argmin()

            human_c_idx_list.append(human_c_idx)

            # all_seq_human_contact_idxs_list.append(human_c_idx)

    human_c_idx_list = np.asarray(human_c_idx_list) # Should be Nv ideally. 

    extracted_obj_verts = full_verts[human_c_idx_list]
   
    debug_obj_mesh = trimesh.Trimesh(
        vertices=extracted_obj_verts,
        faces=part_faces,
        process=False)

    dest_debug_obj_mesh_path = part_obj_path.replace(".ply", "_check_vids.obj")
    debug_obj_mesh.export(open(dest_debug_obj_mesh_path, 'w'), file_type='obj') 

    dest_part_idxs_folder = "./part_vert_ids"
    if not os.path.exists(dest_part_idxs_folder):
        os.makedirs(dest_part_idxs_folder)

    dest_vid_npy_path = os.path.join(dest_part_idxs_folder, part_obj_path.split("/")[-1].replace(".ply", "_vids.npy"))
    np.save(dest_vid_npy_path, human_c_idx_list) 

def load_object_geometry(object_vis_name, obj_scale, obj_rot, obj_trans):
    # Load object information 
    # obj_geo_root_folder = "/move/u/jiamanli/datasets/FullBodyManipCapture/captured_objects"
    # obj_mesh_path = os.path.join(obj_geo_root_folder, object_vis_name+"_cleaned_simplified.obj")
    obj_geo_root_folder = "/move/u/jiamanli/datasets/FullBodyManipCapture/object_pcs"
    obj_mesh_path = os.path.join(obj_geo_root_folder, object_vis_name+"_cleaned_simplified.ply")

    if object_vis_name == "vacuum" or object_vis_name == "mop":
        obj_mesh_path = os.path.join(obj_geo_root_folder, object_vis_name+"_cleaned_simplified_top.ply")
       
    rest_obj_pcs = trimesh.load_mesh(obj_mesh_path)
    rest_obj_points = np.asarray(rest_obj_pcs.vertices)

    obj_pcs = apply_transformation_to_obj_geometry(rest_obj_points, obj_scale, obj_rot, obj_trans) 

    return obj_pcs  

def apply_transformation_to_obj_geometry(obj_mesh_verts, obj_scale, obj_rot, obj_trans):
    ori_obj_verts = torch.from_numpy(obj_mesh_verts).float()[None].repeat(obj_trans.shape[0], 1, 1) # T X Nv X 3 

    seq_scale = torch.from_numpy(obj_scale).float() # T 
    seq_rot_mat = torch.from_numpy(obj_rot).float() # T X 3 X 3 
    seq_trans = torch.from_numpy(obj_trans).float() # T X 3 X 1 
    transformed_obj_verts = seq_scale.unsqueeze(-1).unsqueeze(-1) * \
    seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2)) + seq_trans
    transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

    return transformed_obj_verts.detach().cpu().numpy() 

def get_vertex_colors_in_contact(ori_obj_verts, contact_labels_list, color_list):
    vertex_colors = np.zeros_like(ori_obj_verts) # Nv X 3 
    for tmp_idx in range(vertex_colors.shape[0]):
        vertex_colors[tmp_idx] = np.asarray([238, 228, 228])

    num_parts = len(contact_labels_list)
    for p_idx in range(num_parts):
        curr_contact_label = contact_labels_list[p_idx]

        for tmp_c_idx in range(curr_contact_label.shape[0]):
            if curr_contact_label[tmp_c_idx]:
                vertex_colors[tmp_c_idx] = color_list[p_idx] # np.asarray([226, 135, 67])

    return vertex_colors

def sample_pcs_from_mesh():
    obj_geo_folder = "/move/u/jiamanli/datasets/FullBodyManipCapture/captured_objects" 
    dest_pcs_folder = "/move/u/jiamanli/datasets/FullBodyManipCapture/object_pcs"
    if not os.path.exists(dest_pcs_folder):
        os.makedirs(dest_pcs_folder) 

    ply_files = os.listdir(obj_geo_folder)
    for ply_name in ply_files:
        if "cleaned_simplified.obj" in ply_name or "cleaned_simplified_top.obj" in ply_name:
            ply_path = os.path.join(obj_geo_folder, ply_name)
            mesh = trimesh.load_mesh(ply_path)

            sampled_points, _ = trimesh.sample.sample_surface_even(mesh, count=1024)
            dest_ply_path = os.path.join(dest_pcs_folder, ply_name.replace(".obj", ".ply"))

            sampled_points = trimesh.PointCloud(sampled_points)
            sampled_points.export(dest_ply_path)

def compute_contact_label_w_semantics():
    # Prepare SMPLX model 
    soma_work_base_dir = '/move/u/jiamanli/datasets/mreaching_data'
    support_base_dir = os.path.join(soma_work_base_dir, 'support_files')
    surface_model_type = "smplx"
    surface_model_fname = os.path.join(support_base_dir, surface_model_type, "male", 'model.npz')
    dmpl_fname = None
    num_dmpls = None 
    num_expressions = None
    num_betas = 16 

    male_bm = BodyModel(bm_fname=surface_model_fname,
                    num_betas=num_betas,
                    num_expressions=num_expressions,
                    num_dmpls=num_dmpls,
                    dmpl_fname=dmpl_fname)
    female_bm = BodyModel(bm_fname=surface_model_fname.replace("male", "female"),
                    num_betas=num_betas,
                    num_expressions=num_expressions,
                    num_dmpls=num_dmpls,
                    dmpl_fname=dmpl_fname)
    bm_dict = {'male' : male_bm, 'female' : female_bm}

    motion_data_folder = "/move/u/jiamanli/datasets/FullBodyManipCapture/processed_manip_data/npz_files"

    part_vids_folder = "./part_vert_ids"
    left_hand_vids_path = os.path.join(part_vids_folder, "left_hand_vids.npy")
    right_hand_vids_path = os.path.join(part_vids_folder, "right_hand_vids.npy")
    left_foot_vids_path = os.path.join(part_vids_folder, "left_foot_vids.npy")
    right_foot_vids_path = os.path.join(part_vids_folder, "right_foot_vids.npy")

    left_hand_vids = np.load(left_hand_vids_path)
    right_hand_vids = np.load(right_hand_vids_path)
    left_foot_vids = np.load(left_foot_vids_path)
    right_foot_vids = np.load(right_foot_vids_path) 

    # Load part object geometry to get the faces 
    left_hand_geo_path = "./left_hand.ply"
    right_hand_geo_path = "./right_hand.ply"
    left_foot_geo_path = "./left_foot.ply"
    right_foot_geo_path = "./right_foot.ply"
    
    left_hand_mesh = trimesh.load_mesh(left_hand_geo_path)
    left_hand_faces = np.asarray(left_hand_mesh.faces) 
    
    right_hand_mesh = trimesh.load_mesh(right_hand_geo_path)
    right_hand_faces = np.asarray(right_hand_mesh.faces) 
    
    left_foot_mesh = trimesh.load_mesh(left_foot_geo_path)
    left_foot_faces = np.asarray(left_foot_mesh.faces) 
    
    right_foot_mesh = trimesh.load_mesh(right_foot_geo_path)
    right_foot_faces = np.asarray(right_foot_mesh.faces) 

    dest_contact_npz_folder = "/move/u/jiamanli/datasets/FullBodyManipCapture/processed_manip_data/final_hand_foot_contact_pkl_files"
    dest_contact_vis_obj_folder = "/move/u/jiamanli/datasets/FullBodyManipCapture/processed_manip_data/final_hand_foot_contact_vis_obj"

    if not os.path.exists(dest_contact_npz_folder):
        os.makedirs(dest_contact_npz_folder)

    if not os.path.exists(dest_contact_vis_obj_folder):
        os.makedirs(dest_contact_vis_obj_folder)

    subject_object_dict = {} 

    npz_files = os.listdir(motion_data_folder)
    npz_files.sort() 

    # block_idx = 0 
    # block_size = 700 
    # start_idx = block_idx * block_size 
    # end_idx = (block_idx+1) * block_size 

    for npz_name in npz_files:
        dest_contact_pkl_path = os.path.join(dest_contact_npz_folder, npz_name.replace(".npz", ".pkl"))
        if os.path.exists(dest_contact_pkl_path):
            continue 

        npz_path = os.path.join(motion_data_folder, npz_name) 
        npz_data = np.load(npz_path)

        subject_name = npz_name.split("_")[0]
        object_name = npz_name.split("_")[1]

        # tmp_k_name = subject_name + "_" + object_name 
        # if tmp_k_name in subject_object_dict:
        #     continue 

        root_trans = torch.from_numpy(npz_data['root_trans']).float() # T X 3 
        
        root_orient = torch.from_numpy(npz_data['root_orient']).float() # T X 3 
        pose_body = torch.from_numpy(npz_data['pose_body']).float() # T X 63 
        aa_rot_rep = torch.cat((root_orient[:, None, :], pose_body.reshape(-1, 21, 3)), dim=1) # T X 22 X 3 

        betas = torch.from_numpy(npz_data['betas']).float() # 1 X 16 
        gender = npz_data['gender']

        obj_rot = npz_data['obj_rot'] # T X 3 X 3 
        obj_scale = npz_data['obj_scale'] # T 
        obj_trans = npz_data['obj_trans'] # T X 3 X 1 

        # Get object mesh 
        obj_points = load_object_geometry(object_name, obj_scale, obj_rot, obj_trans) 

        # Get human mesh 
        human_jnts, human_verts, human_faces = run_smplx_model(root_trans[None], aa_rot_rep[None], betas, [gender], bm_dict) 

        # Extract human part vertices 
        left_hand_verts = human_verts[0][:, left_hand_vids, :].detach().cpu().numpy()
        right_hand_verts = human_verts[0][:, right_hand_vids, :].detach().cpu().numpy()
        left_foot_verts = human_verts[0][:, left_foot_vids, :].detach().cpu().numpy()
        right_foot_verts = human_verts[0][:, right_foot_vids, :].detach().cpu().numpy()

        curr_seq_contact_dict = {}
        curr_seq_contact_dict['lhand_contact_labels'] = {}
        curr_seq_contact_dict['rhand_contact_labels'] = {}
        curr_seq_contact_dict['lfoot_contact_labels'] = {}
        curr_seq_contact_dict['rfoot_contact_labels'] = {}

        # Compute contact for between each human part and object 
        num_steps = obj_points.shape[0]
        for t_idx in range(num_steps): 
            lhand_contact_part_points, lhand_contact_part_labels, lhand_contact_obj_verts = \
                            get_contact_labels(obj_points[t_idx], left_hand_verts[t_idx], left_hand_faces) 

            rhand_contact_part_points, rhand_contact_part_labels, rhand_contact_obj_verts = \
                            get_contact_labels(obj_points[t_idx], right_hand_verts[t_idx], right_hand_faces) 

            lfoot_contact_part_points, lfoot_contact_part_labels, lfoot_contact_obj_verts = \
                            get_contact_labels(obj_points[t_idx], left_foot_verts[t_idx], left_foot_faces) 

            rfoot_contact_part_points, rfoot_contact_part_labels, rfoot_contact_obj_verts = \
                            get_contact_labels(obj_points[t_idx], right_foot_verts[t_idx], right_foot_faces) 


            debug_part = False 
            if debug_part:
                dest_debug_vis_folder = os.path.join(dest_contact_vis_obj_folder, npz_name.replace(".npz", ""))
                if not os.path.exists(dest_debug_vis_folder):
                    os.makedirs(dest_debug_vis_folder) 

                dest_debug_lhand_mesh_path = os.path.join(dest_debug_vis_folder, "%05d"%(t_idx)+"_lhand.obj")
                debug_lhand_mesh = trimesh.Trimesh(
                    vertices=left_hand_verts[t_idx],
                    faces=left_hand_faces,
                    process=False)
                
                debug_lhand_mesh.export(open(dest_debug_lhand_mesh_path, 'w'), file_type='obj')

                dest_debug_rhand_mesh_path = os.path.join(dest_debug_vis_folder, "%05d"%(t_idx)+"_rhand.obj")
                debug_rhand_mesh = trimesh.Trimesh(
                    vertices=right_hand_verts[t_idx],
                    faces=right_hand_faces,
                    process=False)
                
                debug_rhand_mesh.export(open(dest_debug_rhand_mesh_path, 'w'), file_type='obj')

                dest_debug_lfoot_mesh_path = os.path.join(dest_debug_vis_folder, "%05d"%(t_idx)+"_lfoot.obj")
                debug_lfoot_mesh = trimesh.Trimesh(
                    vertices=left_foot_verts[t_idx],
                    faces=left_foot_faces,
                    process=False)
                
                debug_lfoot_mesh.export(open(dest_debug_lfoot_mesh_path, 'w'), file_type='obj')

                dest_debug_rfoot_mesh_path = os.path.join(dest_debug_vis_folder, "%05d"%(t_idx)+"_rfoot.obj")
                debug_rfoot_mesh = trimesh.Trimesh(
                    vertices=right_foot_verts[t_idx],
                    faces=right_foot_faces,
                    process=False)
                
                debug_rfoot_mesh.export(open(dest_debug_rfoot_mesh_path, 'w'), file_type='obj')

            # Save to npz files. 
            lhand_color = np.asarray([255, 87, 51])  # red 
            rhand_color = np.asarray([134, 17, 226]) # purple 
            lfoot_color = np.asarray([17, 99, 226]) # blue
            rfoot_color = np.asarray([22, 173, 100]) # green 

            # Visulization of contact for debug 
            obj_vertex_colors = get_vertex_colors_in_contact(obj_points[t_idx], \
                        [lhand_contact_part_labels, rhand_contact_part_labels, \
                        lfoot_contact_part_labels, rfoot_contact_part_labels], \
                        [lhand_color, rhand_color, lfoot_color, rfoot_color])
            
            curr_seq_contact_dict['lhand_contact_labels'][t_idx] = lhand_contact_part_labels
            curr_seq_contact_dict['rhand_contact_labels'][t_idx] = rhand_contact_part_labels
            curr_seq_contact_dict['lfoot_contact_labels'][t_idx] = lfoot_contact_part_labels
            curr_seq_contact_dict['rfoot_contact_labels'][t_idx] = rfoot_contact_part_labels

            # print("lhand contact labels sum:{0}".format(lhand_contact_part_labels.sum()))
            # print("rhand contact labels sum:{0}".format(rhand_contact_part_labels.sum()))
            # print("lfoot contact labels sum:{0}".format(lfoot_contact_part_labels.sum()))
            # print("rfoot contact labels sum:{0}".format(rfoot_contact_part_labels.sum()))
            # human_vertex_colors = np.zeros_like(human_verts[0][t_idx].detach().cpu().numpy()) # Nv X 3 
            # for tmp_idx in range(human_vertex_colors.shape[0]):
            #     human_vertex_colors[tmp_idx] = np.asarray([238, 228, 228])
            
            # dest_root_vis_folder = "./debug_contact_vis_mesh"
            # dest_debug_vis_folder = os.path.join(dest_root_vis_folder, npz_name.replace(".npz", ""))
            # if not os.path.exists(dest_debug_vis_folder):
            #     os.makedirs(dest_debug_vis_folder)

            tmp_k_name = subject_name + "_" + object_name 
            if tmp_k_name not in subject_object_dict:
                dest_debug_vis_folder = os.path.join(dest_contact_vis_obj_folder, npz_name.replace(".npz", ""))
                if not os.path.exists(dest_debug_vis_folder):
                    os.makedirs(dest_debug_vis_folder) 
                dest_debug_obj_mesh_path = os.path.join(dest_debug_vis_folder, "%05d"%(t_idx)+"_object.ply")
                # debug_obj_mesh = trimesh.Trimesh(
                #     vertices=obj_verts[t_idx],
                #     faces=obj_faces,
                #     vertex_colors=obj_vertex_colors,
                #     process=False)

                # result = trimesh.exchange.ply.export_ply(debug_obj_mesh, encoding='ascii')
                # output_file = open(dest_debug_obj_mesh_path, "wb+")
                # output_file.write(result)
                # output_file.close()

                # debug_obj_mesh.export(open(dest_debug_obj_mesh_path, 'w'), file_type='ply')

                sampled_points = trimesh.PointCloud(obj_points[t_idx], colors=obj_vertex_colors)
                sampled_points.export(dest_debug_obj_mesh_path)

                dest_debug_human_mesh_path = os.path.join(dest_debug_vis_folder, "%05d"%(t_idx)+"_human.obj")
                debug_human_mesh = trimesh.Trimesh(
                    vertices=human_verts[0][t_idx].detach().cpu().numpy(),
                    faces=human_faces,
                    # vertex_colors=obj_vertex_colors,
                    process=False)
                
                debug_human_mesh.export(open(dest_debug_human_mesh_path, 'w'), file_type='obj')

        pkl.dump(curr_seq_contact_dict, open(dest_contact_pkl_path, 'wb'))

        subject_object_dict[tmp_k_name] = 1 

def check_contact_mode():
    contact_pkl_folder = "/move/u/jiamanli/datasets/FullBodyManipCapture/processed_manip_data/final_hand_foot_contact_pkl_files" 
    pkl_files = os.listdir(contact_pkl_folder)
    for pkl_name in pkl_files:
        pkl_path = os.path.join(contact_pkl_folder, pkl_name)

        pkl_data = pkl.load(open(pkl_path, 'rb'))
        

if __name__ == "__main__":
    full_obj_path = "./human_template.ply"

    part_obj_path = "./left_hand.ply" 
    # extract_part_vert_idx(part_obj_path, full_obj_path) 

    part_obj_path = "./right_hand.ply" 
    # extract_part_vert_idx(part_obj_path, full_obj_path) 

    part_obj_path = "./left_foot.ply" 
    # extract_part_vert_idx(part_obj_path, full_obj_path) 

    part_obj_path = "./right_foot.ply" 
    # extract_part_vert_idx(part_obj_path, full_obj_path) 

    # sample_pcs_from_mesh() 

    # compute_contact_label_w_semantics() 
