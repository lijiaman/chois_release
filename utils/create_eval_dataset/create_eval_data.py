'''
Creating an evaluation dataset for single-object interaction. 

scene: frl_apartment_0 
objects in interaction: 13 obejcts 
objects in the scene: floor (8), shelf (154, 175), sofa (4), table (202, 185, 91, 119, 5, 32, 141), 
countertop (221), refrigerator (208), tv-screen (45, 231).   

floorlamp (floorlamp/clothesstand/tripod):
1. Pick up floorlamp, move floorlamp to be close to the sofa. 
2. Pick up floorlamp, move floorlamp to be close to the shelf. 
3. Pull the floorlamp to be close to the sofa. 
4. Pull the floorlamp to be close to the shelf. 

table (smalltable/largetable):
1. Lift the table, move the table to be close to the table. 
2. Push the table to be close to the shelf. 
3. Pull the table to be close to the counter. 

chair (whitechair/woodchair): 
1. Lift the chair, move the chair to be close to the table. 
2. Pull the chair to be close to the table.  

box (smallbox/largebox/plasticbox/suitcase):
1. Lift a box, move the box and put down on the table. 
2. Lift a box, move the box and put down on the countertop. 
3. Lift a box, move the box and place it under the table.
4. Push the box to be next to the countertop. 
 
monitor: 
1. Lift a monitor, move the monitor and put down on the table. 
2. Lift a monitor, move the monitor and put down on the floor in front of the TV. 

trashcan: 
1. Lift a trashcan, move the trashcan to be close to the shelf. 
2. Lift a trashcan, move the trashcan to be close to the fridge. 

primitive_functions:
- sample_pts_from_top_surface_of_object(scene_pts, class_labels, object_label) # Ns X 3, Ns X 1, string -> 1 X 3  
- sample_pts_under_object(scene_pts, class_labels, object_label) # Ns X 3, Ns X 1, string -> 1 X 3
- sample_pts_between_objects(scene_pts, class_labels, object_label_a, object_label_b) # Ns X 3, Ns X 1, string, string -> 1 X 3
- sample_pts_near_to_object(scene_pts, class_labels, object_label) # Ns X 3, Ns X 1, string -> 1 X 3

LLM aims to 
1. extract the target obejct name
2. select a function from given primitive functions 
Then based on LLM result, we can call the function, input our scene points, class labels, object_label to sample a target 3D point. 
'''
import os 
import numpy as np
import json 
import random 

import shutil 

from matplotlib import pyplot as plt

import trimesh

import torch 

import openai 

from primitive_functions import sample_pts_between_objects, sample_pts_from_top_surface_of_object, sample_pts_near_to_object, sample_pts_under_object
from primitive_functions import extract_object_id_for_faces, assign_semantics_for_obj_ids, convert_pt_to_habitat_coord 

from plan_path_on_habitat import gen_path_on_habitat, get_sim_and_agent, gen_path_for_multiple_objs_on_habitat

# Try to match the descriptions in training data. 
mapping_dict = {
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

def load_language_descriptions():
    text_dict = {}
    text_dict['floorlamp'] = []
    text_dict['floorlamp'].append("Pick up floorlamp, move floorlamp to be close to the sofa.")
    text_dict['floorlamp'].append("Pick up floorlamp, move floorlamp to be close to the shelf.")
    text_dict['floorlamp'].append("Pull the floorlamp to be close to the sofa.")
    text_dict['floorlamp'].append("Pull the floorlamp to be close to the shelf.")

    text_dict['table'] = []
    text_dict['table'].append("Lift the table, move the table to be close to the table.")
    text_dict['table'].append("Push the table to be close to the shelf.")
    text_dict['table'].append("Pull the table to be close to the countertop.")

    text_dict['chair'] = []
    text_dict['chair'].append("Lift the chair, move the chair to be close to the table.")
    text_dict['chair'].append("Pull the chair to be close to the table.")

    text_dict['box'] = []
    text_dict['box'].append("Lift a box, move the box and put down on the table.")
    text_dict['box'].append("Lift a box, move the box and put down on the countertop.")
    text_dict['box'].append("Lift a box, move the box and place it under the table.")
    text_dict['box'].append("Push the box to be next to the countertop.")

    text_dict['monitor'] = []
    text_dict['monitor'].append("Lift a monitor, move the monitor and put down on the table.")
    text_dict['monitor'].append("Lift a monitor, move the monitor and put down on the floor in front of the tv-screen.")

    text_dict['trashcan'] = []
    text_dict['trashcan'].append("Lift a trashcan, move the trashcan to be close to the shelf.")
    text_dict['trashcan'].append("Lift a trashcan, move the trashcan to be close to the refrigerator.")
    
    return text_dict 

def lm_engine(source, planning_lm_id, device):
    if source == 'huggingface':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(planning_lm_id)
        model = AutoModelForCausalLM.from_pretrained(planning_lm_id, pad_token_id=tokenizer.eos_token_id).to(device)

    def _generate(prompt, sampling_params):
        if source == 'openai':
            # response = openai.ChatCompletion.create(engine=planning_lm_id, prompt=prompt, **sampling_params)
            # response = openai.Completion.create(engine=planning_lm_id, prompt=prompt, **sampling_params)
            # generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
            # calculate mean log prob across tokens
            # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(sampling_params['n'])]

            response = openai.ChatCompletion.create(model=planning_lm_id, \
                    messages=[{"role": "user", "content": prompt}], **sampling_params)
            generated_samples = response['choices'][0]['message']['content']
        
        return generated_samples 

    return _generate

def load_LLM():
    source = 'openai'  # select from ['openai', 'huggingface']
    planning_lm_id = 'gpt-3.5-turbo'  # see comments above for all options
    # planning_lm_id = 'davinci-002'
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_STEPS = 20  # maximum number of steps to be generated
    CUTOFF_THRESHOLD = 0.8  # early stopping threshold based on matching score and likelihood score
    P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
    BETA = 0.3  # weighting coefficient used to rank generated samples
    if source == 'openai':
        openai.api_key = ""
        sampling_params = \
                {
                    "max_tokens": 20,
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "n": 10,
                    # "logprobs": 1,
                    "presence_penalty": 0.5,
                    "frequency_penalty": 0.3,
                    "stop": '\n'
                }

    generator = lm_engine(source, planning_lm_id, device)

    return generator, sampling_params 

def create_prompt(interaction_text):
    prompt = "Given a language description, please output a list with two strings. \
    First, the object name the human is actively interacting with. \
    Second, the object names related to the target position in the end.\
    Besides, please also return the function name selecting from this function list \
    [sample_pts_from_top_surface_of_object, sample_pts_under_object, sample_pts_between_objects, sample_pts_near_to_object].\
    Here're some examples of language descriptions input and the expected output.\n"

    prompt += "Input: Lift a box, move the box and put down on the table."
    prompt += "Output: [box, table], sample_pts_from_top_surface_of_object\n"

    prompt += "Input: Lift a box, move the box and place it under the table."
    prompt += "Output: [box, table], sample_pts_under_object\n"

    prompt += "Input: Push the box to be next to the countertop. "
    prompt += "Output: [box, table], sample_pts_near_to_object\n"

    prompt += "Input: Lift a box, move the box and place it between the sofa and the table."
    prompt += "Output: [box, (sofa, table)], sample_pts_between_objects\n"
    
    prompt += "\nThe language description is as follows.\n"
    
    # prompt += "Lift a box, move a box and put it down on a table."
    prompt += interaction_text 

    return prompt 

def extract_object_and_function_name(llm_out_str):
    # 'Output: [floorlamp, sofa], sample_pts_near_to_object'
    print("llm output:{0}".format(llm_out_str))
    if ":" in llm_out_str:
        object_name_list = llm_out_str.split(":")[1].split("]")[0].replace("[", "").strip().replace(" ", "").split(",") 
    else:
        object_name_list = llm_out_str.split("]")[0].replace("[", "").strip().replace(" ", "").split(",") 
    func_name = llm_out_str.split(",")[-1].strip() 

    return object_name_list, func_name 

def assign_labels_to_points(scene_points, faces, face_labels):
    """
    Assign a semantic label to each scene point based on face labels.

    Parameters:
    - scene_points: Nv x 3 numpy array of 3D coordinates.
    - faces: Nf x 3 numpy array of indices representing vertices of triangular faces.
    - face_labels: Nf x 1 array of semantic labels for each face.

    Returns:
    - A Nv x 1 array of semantic labels for each scene point.
    """
    
    # Compute centroids for each face
    face_centroids = np.mean(scene_points[faces], axis=1)
    
    # For each scene point, find the closest face centroid and assign its label
    point_labels = np.empty((len(scene_points), 1), dtype=face_labels.dtype)

    for i, point in enumerate(scene_points):
        distances = np.linalg.norm(face_centroids - point, axis=1)
        closest_face_index = np.argmin(distances)
        point_labels[i] = face_labels[closest_face_index]

    return point_labels

def assign_labels_to_vertices(faces, face_labels):
    """
    Assign a semantic label to each vertex based on face labels.
    A vertex gets the label of the last face it is found in.
    """
    # Initialize an array for vertex labels with -1 (or other non-valid value) 
    # to track unassigned vertices.
    vertex_labels = ["unassigned"] * (np.max(faces) + 1)
    
    for face, label in zip(faces, face_labels):
        for vertex in face:
            vertex_labels[vertex] = label

    return vertex_labels

def get_scene_verts_and_labels(data_root_folder, scene_name):
    mesh_path = os.path.join(data_root_folder, scene_name, "habitat", "mesh_semantic.ply")
    semantic_json_path = os.path.join(data_root_folder, scene_name, "habitat", "info_semantic.json")
    scene_mesh = trimesh.load_mesh(mesh_path)
    # scene_verts = scene_mesh.vertices # Nv X 3
    # scene_faces = scene_mesh.faces # Nf X 3 
    scene_verts_data = scene_mesh.metadata['_ply_raw']['vertex']['data']
    scene_verts_list = []
    num_verts = scene_verts_data.shape[0]
    for v_idx in range(num_verts):
        scene_verts_list.append(np.asarray([scene_verts_data[v_idx][0], scene_verts_data[v_idx][1], \
                    scene_verts_data[v_idx][2]]))
    scene_verts = np.asarray(scene_verts_list)

    scene_faces, object_ids_list = extract_object_id_for_faces(scene_mesh)  
    object_semantic_names_list = assign_semantics_for_obj_ids(semantic_json_path, object_ids_list) # a list of string, the string represents the semantic class of each face. 
    print("Semantic classes created!")

    vertex_semantic_labels_list = assign_labels_to_vertices(scene_faces, object_semantic_names_list)

    return scene_verts, scene_faces, vertex_semantic_labels_list

def generate_path_for_single_object(response_dict, out_json_path):
    response_idx_dict = {} 
    cnt = 0 
    for k in response_dict: 
        response_text = response_dict[k]

        response_idx_dict[cnt] = {}
        response_idx_dict[cnt]['text'] = k 
        response_idx_dict[cnt]['response_out'] = response_text 

        object_name_list, func_name = extract_object_and_function_name(response_text)

        # 4. Use the sampled 3D position as target 3D position, call Habitat API to plan a collision-free path. 
        if func_name == "sample_pts_from_top_surface_of_object":
            sampled_pt_list = sample_pts_from_top_surface_of_object(scene_verts, object_semantic_names_list, object_name_list[1])
        elif func_name == "sample_pts_near_to_object":
            sampled_pt_list = sample_pts_near_to_object(scene_verts, object_semantic_names_list, object_name_list[1])
        elif func_name == "sample_pts_under_object":
            sampled_pt_list = sample_pts_under_object(scene_verts, object_semantic_names_list, object_name_list[1])
        elif func_name == "sample_pts_between_objects":
            sampled_pt_list = sample_pts_between_objects(scene_verts, object_semantic_names_list, \
                        object_name_list[1], object_name_list[2])
        
        if sampled_pt_list is not None:
            sampled_pt_list = np.asarray(sampled_pt_list)
            sampled_pt_data = torch.from_numpy(sampled_pt_list).float() 
            sampled_pts_in_habitat = convert_pt_to_habitat_coord(sampled_pt_data)

            gen_path_on_habitat(sim, agent, scene_name, sampled_pts_in_habitat, output_folder, cnt)

            cnt += 1 

    json.dump(response_idx_dict, open(out_json_path, 'w'))
    print("Total number of sequences:{0}".format(len(response_idx_dict)))

def generate_path_for_single_object_in_3d_scene(response_dict, out_json_path, scene_name_list, \
        data_root_folder, output_folder):

    object2category_dict = {"floorlamp": "floorlamp", "clothesstand": "floorlamp", "tripod": "floorlamp", \
            "largetable": "table", "smalltable": "table", \
            "woodchair": "chair", "whitechair": "chair", \
            "smallbox": "box", "largebox": "box", "plasticbox": "box", "suitcase": "box", \
            "monitor": "monitor", \
            "trashcan": "trashcan"}

    category2object_list_dict = {
        "floorlamp": ["floorlamp", "clothesstand", "tripod"], \
        "table": ["largetable", "smalltable"], \
        "chair": ["woodchair", "whitechair"], \
        "box": ["smallbox", "largebox", "plasticbox", "suitcase"], \
        "monitor": ["monitor"], \
        "trashcan": ["trashcan"], \
    } 

    # 8 scenes X 14 language (2/3*2)
    for scene_name in scene_name_list:
        scene_verts, scene_faces, object_semantic_names_list = \
                get_scene_verts_and_labels(data_root_folder, scene_name)

        sim, agent = get_sim_and_agent(scene_name)

        response_idx_dict = {}
        cnt = 0 
        for k in response_dict: 
            response_text = response_dict[k]

            response_idx_dict[cnt] = {}
            response_idx_dict[cnt]['text'] = k 
            response_idx_dict[cnt]['response_out'] = response_text 

            object_name_list, func_name = extract_object_and_function_name(response_text)

            interact_type_name = object_name_list[0]
            interact_object_name_list = random.sample(category2object_list_dict[interact_type_name], 1) 

            # For each object, generate a path for it. 
            for curr_object_name in interact_object_name_list:

                sampled_pt_list = None 
                max_try = 20 
                try_cnt = 0
                while sampled_pt_list is None:
                    # 4. Use the sampled 3D position as target 3D position, call Habitat API to plan a collision-free path. 
                    if func_name == "sample_pts_from_top_surface_of_object":
                        sampled_pt_list = sample_pts_from_top_surface_of_object(scene_verts, \
                                object_semantic_names_list, object_name_list[1])
                    elif func_name == "sample_pts_near_to_object":
                        sampled_pt_list = sample_pts_near_to_object(scene_verts, \
                                object_semantic_names_list, object_name_list[1])
                    elif func_name == "sample_pts_under_object":
                        sampled_pt_list = sample_pts_under_object(scene_verts, \
                                object_semantic_names_list, object_name_list[1])
                    elif func_name == "sample_pts_between_objects":
                        sampled_pt_list = sample_pts_between_objects(scene_verts, \
                                object_semantic_names_list, object_name_list[1], object_name_list[2])
                    
                    try_cnt += 1
                    if try_cnt > max_try:
                        break 
                
                if sampled_pt_list is not None:
                    sampled_pt_list = np.asarray(sampled_pt_list)
                    sampled_pt_data = torch.from_numpy(sampled_pt_list).float() 
                    sampled_pts_in_habitat = convert_pt_to_habitat_coord(sampled_pt_data)

                    gen_path_on_habitat(sim, agent, scene_name, sampled_pts_in_habitat, output_folder, \
                                curr_object_name, cnt)

            cnt += 1 

        json.dump(response_idx_dict, open(out_json_path, 'w'))
        print("Total number of sequences:{0}".format(len(response_idx_dict)))

def generate_path_for_multiple_objects_and_transition(scene_name, scene_verts, object_semantic_names_list, \
    response_dict, output_folder, out_json_path, num_samples=20):
    response_idx_dict = {} 
    response_dict_key_list = list(response_dict.keys())

    cnt = 0 
    for idx in range(num_samples):
        # random sample three objects from response dict.
        success_flag = True 

        start_pt_list = []
        end_pt_list = []

        response_k_list = random.sample(response_dict_key_list, 3) 
        for k in response_k_list:
            response_text = response_dict[k]

            object_name_list, func_name = extract_object_and_function_name(response_text)

            # 4. Use the sampled 3D position as target 3D position, call Habitat API to plan a collision-free path. 
            if func_name == "sample_pts_from_top_surface_of_object":
                sampled_pt_list = sample_pts_from_top_surface_of_object(scene_verts, object_semantic_names_list, \
                    object_name_list[1])
            elif func_name == "sample_pts_near_to_object":
                sampled_pt_list = sample_pts_near_to_object(scene_verts, object_semantic_names_list, \
                    object_name_list[1])
            elif func_name == "sample_pts_under_object":
                sampled_pt_list = sample_pts_under_object(scene_verts, object_semantic_names_list, \
                    object_name_list[1])
            elif func_name == "sample_pts_between_objects":
                sampled_pt_list = sample_pts_between_objects(scene_verts, object_semantic_names_list, \
                    object_name_list[1], object_name_list[2])
        
            # samppled_pt_list: K X 3 
            if sampled_pt_list is None:
                success_flag = False 
                break 

            # Sample start point randomly
            tmp_sampled_pt_list = []
            for tmp_idx in range(sampled_pt_list.shape[0]):
                seed = random.sample(list(range(9999)), 1)[0]
                sim.pathfinder.seed(seed)

                start_pts_in_habitat = sim.pathfinder.get_random_navigable_point()

                tmp_sampled_pt_list.append(start_pts_in_habitat)

            start_pts_in_habitat = np.asarray(tmp_sampled_pt_list) # K X 3 

            sampled_pt_list = np.asarray(sampled_pt_list)
            sampled_pt_data = torch.from_numpy(sampled_pt_list).float() 
            target_pts_in_habitat = convert_pt_to_habitat_coord(sampled_pt_data)

            start_pt_list.append(start_pts_in_habitat) # each element is K X 3 
            end_pt_list.append(target_pts_in_habitat.detach().cpu().numpy()) # each element is K X 3 

            if cnt not in response_idx_dict:
                response_idx_dict[cnt] = {}
                response_idx_dict[cnt]['text'] = [k] 
                response_idx_dict[cnt]['response_out'] = [response_text]
            else:
                response_idx_dict[cnt]['text'].append(k)
                response_idx_dict[cnt]['response_out'].append(response_text) 

        if success_flag:
            # Generate interaction path, navigation path, interaction path, navigation path, interaction path. 
            gen_path_for_multiple_objs_on_habitat(sim, agent, scene_name, \
                start_pt_list, end_pt_list, output_folder, cnt)  

            cnt += 1 

    json.dump(response_idx_dict, open(out_json_path, 'w'))
    print("Total number of sequences:{0}".format(len(response_idx_dict)))

def gen_scene_floor_height():
    dest_json_path = "/move/u/jiamanli/datasets/replica_processed/scene_floor_height.json"
    scene_mesh_folder = "/move/u/jiamanli/datasets/replica_processed/replica_fixed_poisson_recon_objs"
    scene_names = os.listdir(scene_mesh_folder)
    scene_floor_dict = {}
    for s_name in scene_names:
        if ".obj" in s_name:
            scene_obj_path = os.path.join(scene_mesh_folder, s_name)
            scene_mesh = trimesh.load_mesh(scene_obj_path)
            scene_verts = scene_mesh.vertices 

            scene_floor_height = scene_verts[:, 2].min() 
            print("scene name:{0}".format(s_name))
            print("floor height:{0}".format(scene_floor_height))

            scene_floor_dict[s_name.replace(".obj", "")] = scene_floor_height

    # scene_floor_dict = {
    #     "apartment_1": , "apartment_2": , 
    #     "frl_apartment_0":, "frl_apartment_1":, 
    #     "frl_apartment_2":, "frl_apartment_3":, 
    #     "frl_apartment_4":, "frl_apartment_5":, 
    # }
    print("Scene floor dict: {0}".format(scene_floor_dict))
    json.dump(scene_floor_dict, open(dest_json_path, 'w'))

def filter_sampled_paths():
    # 1. Discard very short or very long paths. 
    # 2. Discard paths that are not starting from a point on the floor. 
    height_json_path = "/move/u/jiamanli/datasets/replica_processed/scene_floor_height.json"
    scene_floor_dict = json.load(open(height_json_path, 'r'))

    dest_json_path = "/viscam/u/jiamanli/github/scene_aware_manip/cvpr2024_utils/create_eval_dataset/selected_long_seq_names.json"
    dest_json_dict = {} 
    cnt = 0 
    
    npy_root_folder = "/viscam/u/jiamanli/github/scene_aware_manip/cvpr2024_utils/create_eval_dataset/replica_single_object_long_seq_data"
    dest_npy_root_folder = npy_root_folder + "_selected"
    scene_names = os.listdir(npy_root_folder)
    for s_name in scene_names:
        scene_folder_path = os.path.join(npy_root_folder, s_name)
        object_names = os.listdir(scene_folder_path)

        curr_scene_height = scene_floor_dict[s_name]

        for o_name in object_names:
            object_folder_path = os.path.join(scene_folder_path, o_name)

            text_idxs = os.listdir(object_folder_path)
            for text_index in text_idxs:
                text_folder_path = os.path.join(object_folder_path, text_index)

                npy_files = os.listdir(text_folder_path)
                for npy_name in npy_files:
                    if ".npy" in npy_name:
                        npy_path = os.path.join(text_folder_path, npy_name)

                        waypoints = np.load(npy_path)

                        num_waypoints = waypoints.shape[0]
                        # Remove distance > 15 meter or < 6 meter. 
                        total_dist = np.linalg.norm(np.diff(waypoints, axis=0), axis=-1).sum() 
                        print("Total dist:{0}".format(total_dist)) 

                        if total_dist < 3:
                            print("Sequence is too short Discarded!")
                            continue  

                        # if total_dist < 6 or total_dist > 15:
                        #     continue  

                        # Remove the sequence with starting frame not on the floor.
                        start_pt_height = waypoints[0, 1]
                        print("start_pt_height:{0}".format(start_pt_height)) 
                        if start_pt_height > curr_scene_height + 0.2:
                            print("Start pt is not on the floor! Discarded!")
                            continue 

                        # Process waypoints to be directly used for model testing 
                        converted_waypoints = convert_habitat_coord_to_model(waypoints) # K X 3 
                        # dense_waypoints = sample_and_adjust_waypoints(converted_waypoints, \
                        #     distance_range=(0.7, 0.9), remainder=3)

                        converted_waypoints = torch.from_numpy(converted_waypoints).float() # K X 3 
                        new_x_data, new_y_data = apply_heuristics_to_planned_path(converted_waypoints[:, 0:1], \
                                            converted_waypoints[:, 1:2]) 
                        # N X 3 
                        new_z_data = torch.zeros_like(new_x_data)
                        new_z_data[:-1, 0] = converted_waypoints[0, 2]
                        new_z_data[-1, 0] = converted_waypoints[-1, 2] 

                        new_xyz_data = torch.cat((new_x_data, new_y_data, new_z_data), dim=-1) # N X 3 
                        print("new xyz data shape:{0}".format(new_xyz_data.shape))
                        if new_xyz_data.shape[0] < 3:
                            continue 
                        dense_waypoints = adjust_waypoints(new_xyz_data.detach().cpu().numpy(), remainder=3) 

                        new_dense_waypoints = np.zeros((dense_waypoints.shape[0]+2, 3))
                        new_dense_waypoints[1:-1, :] = dense_waypoints.copy() 
                        new_dense_waypoints[0, :] = dense_waypoints[0, :].copy()
                        new_dense_waypoints[-1, :] = dense_waypoints[-1, :].copy()

                        assert new_dense_waypoints.shape[0] % 4 == 1 

                        dest_data_folder = os.path.join(dest_npy_root_folder, s_name, o_name, text_index)
                        if not os.path.exists(dest_data_folder):
                            os.makedirs(dest_data_folder)
                        dest_npy_path = os.path.join(dest_data_folder, npy_name)
                        dest_img_path = dest_npy_path.replace(".npy", ".png")
                        dest_fig_path = dest_npy_path.replace(".npy", "_xy.png")

                        np.save(dest_npy_path, new_dense_waypoints)
                        shutil.copy(npy_path.replace(".npy", ".png"), dest_img_path) 

                        # Also visualize dense waypoints 
                        visualize_root_translation(new_dense_waypoints[:, :2], dest_fig_path)

                        dest_json_dict[cnt] = os.path.join(s_name, o_name, text_index, npy_name) 
                        cnt += 1 

    json.dump(dest_json_dict, open(dest_json_path, 'w'))
    print("Number of sequences for evaluation:{0}".format(len(dest_json_dict)))

def visualize_root_translation(xy_data, dest_fig_path, title="Human Root Translation"):
    """
    Visualize human root translation in xy plane.

    :param xy_data: Tensor or ndarray of shape (T, 2) representing the xy root translations.
    :param title: Title for the plot.
    """
    
    # Extract x and y coordinates
    x_coords = xy_data[:, 0]
    y_coords = xy_data[:, 1]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, '-o', markersize=4, linewidth=1)  # '-o' means line with circle markers
    plt.title(title)
    plt.xlabel('X Translation')
    plt.ylabel('Y Translation')
    plt.grid(True)
    # plt.show()
    plt.savefig(dest_fig_path) 

def convert_habitat_coord_to_model(habitat_coord):
    # habitat_coord: K X 3 
    xyz_data = np.zeros_like(habitat_coord) # K X 3 

    xyz_data[:, 0] = habitat_coord[:, 0].copy()
    xyz_data[:, 1] = -habitat_coord[:, 2].copy()
    xyz_data[:, 2] = habitat_coord[:, 1].copy() 

    return xyz_data  

def add_points_in_planned_path(x_data, y_data):
    # x_data: T X 1 
    # y_data: T X 1 
    num_waypoints = x_data.shape[0]

    num_samples_per_seg = 20 
    interval = 1.0/num_samples_per_seg 
    new_x_data_list = []
    new_y_data_list = []
    for idx in range(num_waypoints-1):
        start_pt_x = x_data[idx]
        end_pt_x = x_data[idx+1]

        for t_idx in range(num_samples_per_seg+1):
            alpha = t_idx * interval 
            curr_x_data = (1-alpha) * start_pt_x + alpha * end_pt_x 

            new_x_data_list.append(curr_x_data)

        start_pt_y = y_data[idx]
        end_pt_y = y_data[idx+1] 

        for t_idx in range(num_samples_per_seg+1):
            alpha = t_idx * interval 
            curr_y_data = (1-alpha) * start_pt_y + alpha * end_pt_y

            new_y_data_list.append(curr_y_data)

    new_x_data = torch.stack(new_x_data_list, dim=0) # T X 1 
    new_y_data = torch.stack(new_y_data_list, dim=0) # T X 1 

    return new_x_data, new_y_data 

def compute_points_distance(x_data, y_data):
    # x_data: T X 1 
    # y_data: T X 1 
    abs_dist = torch.sqrt(torch.pow(x_data[1:]-x_data[:-1], 2)+torch.pow(y_data[1:]-y_data[:-1], 2)) # (T-1) X 1 

    return abs_dist 

def apply_heuristics_to_planned_path(x_data, y_data):
    # x_data: T X 1 
    # y_data: T X 1 

    # Currently, we only support the number of waypoints to be (multiple of 4 + 1).  
    # For navigation, use 0.7~0.9 for every 30 frames. Also, need to consider overlapped 10 frames for every two windows. 
    # 0   29   59   89   119   ||139   169   199  219   ||239   269   299  319   ||339   369   399   419    

    new_x_data, new_y_data = add_points_in_planned_path(x_data, y_data)
    
    num_waypoints = new_x_data.shape[0]

    abs_dist = compute_points_distance(new_x_data, new_y_data) # (T-1) X 1 

    test_x_data_list = [new_x_data[0]]
    test_y_data_list = [new_y_data[0]] 

    sum_dist = 0
    dist_list = []
   
    dist_range_min = 0.7
    dist_range_max = 0.8 

    for idx in range(num_waypoints-1):
        sum_dist += abs_dist[idx] 

        if sum_dist <= dist_range_max and sum_dist >= dist_range_min:
            test_x_data_list.append(new_x_data[idx+1])
            test_y_data_list.append(new_y_data[idx+1])

            dist_list.append(sum_dist)
            sum_dist = 0  

    test_x_data = torch.stack(test_x_data_list)
    test_y_data = torch.stack(test_y_data_list)

    # dist_list = torch.stack(dist_list)

    return test_x_data, test_y_data

def generate_dense_waypoints(waypoints, distance_range=(0.7, 0.9)):
    dense_waypoints = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        segment_length = np.linalg.norm(waypoints[i+1] - waypoints[i])
        accumulated_distance = 0
        
        while accumulated_distance + distance_range[0] <= segment_length:
            t = (accumulated_distance + distance_range[0]) / segment_length
            interpolated_point = (1 - t) * waypoints[i] + t * waypoints[i+1]
            dense_waypoints.append(interpolated_point)
            accumulated_distance += distance_range[0]

        # Check the gap between the last added point and the segment's end
        if accumulated_distance < segment_length:
            dense_waypoints.append(waypoints[i+1])
                
        accumulated_distance -= segment_length

    dense_waypoints.append(waypoints[-1])
    return np.array(dense_waypoints)

def adjust_waypoints(dense_waypoints, remainder):
    while (len(dense_waypoints) - remainder) % 4 != 0:
        dense_waypoints = dense_waypoints[1:]
    return dense_waypoints

def sample_and_adjust_waypoints(waypoints, distance_range=(0.7, 0.9), remainder=1):
    dense_waypoints = generate_dense_waypoints(waypoints, distance_range)
    adjusted_waypoints = adjust_waypoints(dense_waypoints, remainder)
    return adjusted_waypoints

if __name__ == "__main__":
    # 1. Load language descriptions for each object. (require manual work)
    # text_dict = load_language_descriptions() 

    # 2. Use LLM to extract the target object name, select a function from given primitive functions. 
    # generator, sampling_params = load_LLM()
    response_json_path = "./eval_dataset_response.json"
    # if not os.path.exists(response_json_path):
    #     response_dict = {}
    #     for object_type in text_dict:
    #         text_list = text_dict[object_type]

    #         for t_idx in range(len(text_list)):
    #             curr_text = text_list[t_idx]

    #             prompt = create_prompt(curr_text) 
    #             samples = generator(prompt, sampling_params) # 'Output: [floorlamp, sofa], sample_pts_near_to_object'
    #             # samples = 'Output: [floorlamp, sofa], sample_pts_near_to_object'

    #             response_dict[curr_text] = samples 

    #         #     break 

    #         # break 

    #     json.dump(response_dict, open(response_json_path, 'w'))
    # else:
    #     response_dict = json.load(open(response_json_path, 'r'))

    response_dict = json.load(open(response_json_path, 'r'))

    # 3. Call the returned function from LLM using 3D scene points, class_labels. 
    # scene_name = "frl_apartment_0"
    # data_root_folder = "/move/u/jiamanli/datasets/replica"
    # output_folder = "/viscam/u/jiamanli/github/scene_aware_manip/cvpr2024_utils/create_eval_dataset/single_object_long_seq_data"
    # scene_verts, scene_faces, object_semantic_names_list = get_scene_verts_and_labels(data_root_folder, scene_name)
    
    # sim, agent = get_sim_and_agent(scene_name)

    # 4.1 Generate path using Habitat for single object. 
    # out_json_path = "./eval_dataset_response_text_idx.json"
    # generate_path_for_single_object(response_dict, out_json_path)     

    # 4.3 Generate path for different scenes
    scene_name_list = ["apartment_1", "apartment_2", "frl_apartment_0", "frl_apartment_1", "frl_apartment_2", \
                    "frl_apartment_3", "frl_apartment_4", "frl_apartment_5"]
    data_root_folder = "/move/u/jiamanli/datasets/replica"
    output_folder = "/viscam/u/jiamanli/github/scene_aware_manip/cvpr2024_utils/create_eval_dataset/replica_single_object_long_seq_data"
    out_json_path = "./eval_dataset_response_text_idx.json"
    # generate_path_for_single_object_in_3d_scene(response_dict, out_json_path, scene_name_list, \
    #                 data_root_folder, output_folder)

    # gen_scene_floor_height() 
    filter_sampled_paths() 
