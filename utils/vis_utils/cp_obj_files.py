import os 
import numpy as np 
import json 
import shutil 

def cp_folder():
    data_root_folder = "/move/u/jiamanli/meta_cvpr2024/for_supp"

    # For single-window seen objects cmp
    # video_data_folder = os.path.join(data_root_folder, "selected_single_window_cmp_few")
    # obj_data_folder = "/move/u/jiamanli/meta_cvpr2024/objs_single_window_cmp_settings"

    # dest_data_folder = os.path.join(data_root_folder, "selected_single_window_cmp_few_objs")

    # For single-window unseen objects cmp 
    # video_data_folder = os.path.join(data_root_folder, "selected_unseen_objects_cmp_few")
    # obj_data_folder = "/move/u/jiamanli/meta_cvpr2024/objs_single_window_cmp_settings_unseen_obj"

    # dest_data_folder = os.path.join(data_root_folder, "selected_unseen_objects_cmp_few_objs")

    # For more results of single-window seen objects, CHOIS 
    # video_data_folder = os.path.join(data_root_folder, "more_results_single_window_seen")
    # obj_data_folder = "/move/u/jiamanli/meta_cvpr2024/objs_single_window_cmp_settings"

    # dest_data_folder = os.path.join(data_root_folder, "more_results_single_window_seen_objs")

    # For more results of single-window unseen objects, CHOIS 
    video_data_folder = os.path.join(data_root_folder, "more_results_single_window_unseen")
    obj_data_folder = "/move/u/jiamanli/meta_cvpr2024/objs_single_window_cmp_settings_unseen_obj"

    dest_data_folder = os.path.join(data_root_folder, "more_results_single_window_unseen_objs")

    model_folders = os.listdir(video_data_folder)
    for model_name in model_folders:
        if "model" not in model_name:
            continue 

        model_folder_path = os.path.join(video_data_folder, model_name)
        video_files = os.listdir(model_folder_path)

        for v_name in video_files:
            if ".mp4" not in v_name:
                continue 

            v_name = v_name.replace(".mp4", "")

            dest_obj_root_folder = os.path.join(dest_data_folder, model_name, v_name)
            if not os.path.exists(dest_obj_root_folder):
                os.makedirs(dest_obj_root_folder)

            if "omomo" in model_name:
                ori_obj_folder = os.path.join(obj_data_folder, model_name, v_name, "objs_step_6_bs_idx_0")

                dest_obj_folder = os.path.join(dest_obj_root_folder, "objs")

                shutil.copytree(ori_obj_folder, dest_obj_folder)
            else:
                ori_obj_folder = os.path.join(obj_data_folder, model_name, v_name, "objs_step_10_bs_idx_0")
                ori_waypoints_obj_folder = os.path.join(obj_data_folder, model_name, v_name, "ball_objs_step_10_bs_idx_0")

                dest_obj_folder = os.path.join(dest_obj_root_folder, "objs")
                dest_waypoints_obj_folder = os.path.join(dest_obj_root_folder, "ball_objs")

                shutil.copytree(ori_obj_folder, dest_obj_folder)
                shutil.copytree(ori_waypoints_obj_folder, dest_waypoints_obj_folder)

if __name__ == "__main__":
    # cp_folder() 
