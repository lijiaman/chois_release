import os 
import json 
import shutil 

def get_new_json(ori_json_path, dest_json_path, selected_json_name_list):
    with open(ori_json_path, 'r') as f:
        ori_json = json.load(f)
    new_json = {}
    for json_name in selected_json_name_list:
        new_json[json_name] = ori_json[json_name]
    with open(dest_json_path, 'w') as f:
        json.dump(new_json, f) 

def copy_selected_files(ori_data_folder, dest_data_folder, selected_json_name_list, file_tag):
    for json_name in selected_json_name_list:
        ori_json_path = os.path.join(ori_data_folder, json_name + file_tag)
        dest_json_path = os.path.join(dest_data_folder, json_name + file_tag)
        shutil.copyfile(ori_json_path, dest_json_path)

if __name__ == "__main__":
    selected_json_name_list = ["0a2378ae-4042-4411-91cc-5e9d868ec63b", "6e31af89-62e3-4ec5-9718-4448bcba6557", \
                        "343dad12-3fba-4f75-8fa3-19f7aa3d5871", "0c51740b-7fd0-4b11-91fe-fb35553d6b4e", \
                        "0f95f670-3100-4cbd-85bd-e2d2167dd450", "0acc134e-65bc-4b68-b836-c591935bdec6", \
                        "0bd81869-830e-4673-bfeb-e4d1e5c60302", "0cd722cb-4cf5-4026-9906-187e51d78c67", \
                        "3a431666-c294-41c8-85ca-2247a19e3671", "0c15e144-0244-4c05-957b-ed5d0a96e38e", \
                        "0d7421d4-2656-435b-9eb1-8884b4b3dcb3", "6f3f7829-e4a6-470f-b59b-98e97921de1d", \
                        "0efd1942-d8aa-4749-a3f7-fb10ed93e1c4", "1d9698ef-2edf-4398-946c-e24de4d33c1f", \
                        "3a392911-1020-4c27-8d1e-2b39e3e69c22", "8dcdf4ef-6969-41af-b203-0bc7c22acb35", \
                        "111030f2-a815-4538-8ef4-be506bdbcf01"] 

    data_folder = "/move/u/jiamanli/datasets/semantic_manip/unseen_objects_data/selected_unseen_objects"
    dest_root_data_folder = "/move/u/jiamanli/for_chois_release/processed_data/unseen_objects_data"

    ori_json_path = os.path.join(data_folder, "selected_object_height.json")
    dest_json_path = os.path.join(dest_root_data_folder, "selected_object_height.json")
    # get_new_json(ori_json_path, dest_json_path, selected_json_name_list)    


    ori_data_folder = os.path.join(data_folder, "selected_rotated_zeroed_obj_files")
    dest_data_folder = os.path.join(dest_root_data_folder, "selected_rotated_zeroed_obj_files") 
    if not os.path.exists(dest_data_folder):
        os.makedirs(dest_data_folder)
    # copy_selected_files(ori_data_folder, dest_data_folder, selected_json_name_list, ".ply") 

    ori_data_folder = os.path.join(data_folder, "selected_rotated_zeroed_obj_sdf_256_npy_files")
    dest_data_folder = os.path.join(dest_root_data_folder, "selected_rotated_zeroed_obj_sdf_256_npy_files")
    if not os.path.exists(dest_data_folder):
        os.makedirs(dest_data_folder)
    # copy_selected_files(ori_data_folder, dest_data_folder, selected_json_name_list, ".npy")
    # copy_selected_files(ori_data_folder, dest_data_folder, selected_json_name_list, ".json")
    # copy_selected_files(ori_data_folder, dest_data_folder, selected_json_name_list, ".obj")

    