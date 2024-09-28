import os 
import subprocess 
import trimesh 
import imageio 
import numpy as np 
import shutil 

BLENDER_PATH = "/viscam/u/jiamanli/blender-3.6.3-linux-x64/blender"
BLENDER_UTILS_ROOT_FOLDER = "/move/u/jiamanli/github/chois_release/manip/vis" 
BLENDER_SCENE_FOLDER = "/move/u/jiamanli/for_chois_release/processed_data/blender_files"

def images_to_video_w_imageio(img_folder, output_vid_file, fps=30):
    img_files = os.listdir(img_folder)
    img_files.sort()
    im_arr = []
    for img_name in img_files:
        img_path = os.path.join(img_folder, img_name)
        im = imageio.imread(img_path)
        im_arr.append(im)

    im_arr = np.asarray(im_arr)
    imageio.mimwrite(output_vid_file, im_arr, fps=fps, quality=8) 

def run_blender_rendering_and_save2video(obj_folder_path, out_folder_path, out_vid_path, \
    condition_folder=None, scene_blend_path=None, \
    vis_object=False, vis_human=True, \
    vis_condition=False, fps=30):
    
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    vid_folder = "/".join(out_vid_path.split("/")[:-1])
    if not os.path.exists(vid_folder):
        os.makedirs(vid_folder)

    if vis_object:
        if vis_human: # vis both human and object
            if vis_condition:
                blender_utils_path = os.path.join(BLENDER_UTILS_ROOT_FOLDER, "blender_vis_w_condition_utils.py") 
                subprocess.call(BLENDER_PATH+" -P "+blender_utils_path+\
            " -b -- --folder "+obj_folder_path+" --condition-folder "+condition_folder+" --scene "+scene_blend_path+" --out-folder "+out_folder_path, shell=True) 
            else:
                blender_utils_path = os.path.join(BLENDER_UTILS_ROOT_FOLDER, "blender_vis_utils.py") 
                subprocess.call(BLENDER_PATH+" -P "+blender_utils_path+\
            " -b -- --folder "+obj_folder_path+" --scene "+scene_blend_path+" --out-folder "+out_folder_path, shell=True) 
   
    images_to_video_w_imageio(out_folder_path, out_vid_path, fps=fps)

def vis_cmp_res():
    vis_conditons = True 

    # For single-window unseen objects more results 
    dest_video_folder = "/move/u/jiamanli/meta_cvpr2024/for_supp/more_results_single_window_unseen_videos_cam2"
    obj_root_folder = "/move/u/jiamanli/meta_cvpr2024/for_supp/more_results_single_window_unseen_objs"

    if vis_conditons:
        dest_video_folder += "_w_conditions"

    # scene_blend_path = "/move/u/jiamanli/meta_cvpr2024/for_supp/for_single_window_comparison.blend"
    scene_blend_path = "/move/u/jiamanli/meta_cvpr2024/for_supp/for_single_window_comparison_cam_2.blend"
    # scene_blend_path = "/move/u/jiamanli/meta_cvpr2024/for_supp/for_single_window_comparison_cam_plasticbox.blend"

    model_names = os.listdir(obj_root_folder)
    for model_name in model_names:
        if "model" not in model_name:
            continue 
        model_folder = os.path.join(obj_root_folder, model_name) 

        seq_names = os.listdir(model_folder)
        for seq_name in seq_names:
            # if ("largetable" not in seq_name) and ("plasticbox" not in seq_name):
            #     continue 

            # if "plasticbox" not in seq_name:
            #     continue 

            if "largetable" not in seq_name:
                continue 

            seq_folder_path = os.path.join(model_folder, seq_name)

            obj_folder_path = os.path.join(seq_folder_path, "objs")
            ball_obj_folder_path = os.path.join(seq_folder_path, "ball_objs")

            dest_out_vid_folder = os.path.join(dest_video_folder, model_name)
            if not os.path.exists(dest_out_vid_folder):
                os.makedirs(dest_out_vid_folder)

            dest_out_img_folder_path = os.path.join(seq_folder_path, "imgs")
            dest_out_vid_path = os.path.join(dest_out_vid_folder, seq_name+".mp4")

            if vis_conditons:
                dest_out_img_folder_path += "_w_conditions"

                run_blender_rendering_and_save2video(obj_folder_path, dest_out_img_folder_path, dest_out_vid_path, \
                condition_folder=ball_obj_folder_path, scene_blend_path=scene_blend_path, \
                vis_object=True, vis_human=True, \
                vis_condition=True, fps=30)
            else:
                run_blender_rendering_and_save2video(obj_folder_path, dest_out_img_folder_path, dest_out_vid_path, \
                condition_folder=None, scene_blend_path=scene_blend_path, \
                vis_object=True, vis_human=True, \
                vis_condition=False, fps=30)

def vis_res_in_3d_scene():
    # dest_video_folder = "/move/u/jiamanli/meta_cvpr2024/for_supp/selected_seq_in_scenes_videos"
    # obj_root_folder = "/move/u/jiamanli/meta_cvpr2024/objs_single_object_long_seq_cmp_settings_whole_seq"

    dest_video_folder = "/move/u/jiamanli/for_chois_release/chois_long_seq_in_scene_results/seq_in_scene_video_res"
    obj_root_folder = "/move/u/jiamanli/for_chois_release/chois_long_seq_in_scene_results/objs_single_window_cmp_settings"

    scene_blend_path = os.path.join(BLENDER_SCENE_FOLDER, "for_seq_in_frl_apartment_4.blend")

    model_names = os.listdir(obj_root_folder)
    for model_name in model_names:
        model_folder = os.path.join(obj_root_folder, model_name) 

        seq_names = os.listdir(model_folder)
        for seq_name in seq_names:
            if "topview" not in seq_name:
                continue 
            seq_folder_path = os.path.join(model_folder, seq_name)

            obj_folder_path = os.path.join(seq_folder_path, "objs_step_10_bs_idx_0")
            ball_obj_folder_path = os.path.join(seq_folder_path, "ball_objs_step_10_bs_idx_0")

            dest_out_vid_folder = os.path.join(dest_video_folder, model_name)
            if not os.path.exists(dest_out_vid_folder):
                os.makedirs(dest_out_vid_folder)

            dest_out_img_folder_path = os.path.join(seq_folder_path, "imgs")
            dest_out_vid_path = os.path.join(dest_out_vid_folder, seq_name+".mp4")

            run_blender_rendering_and_save2video(obj_folder_path, dest_out_img_folder_path, dest_out_vid_path, \
            condition_folder=None, scene_blend_path=scene_blend_path, \
            vis_object=True, vis_human=True, \
            vis_condition=False, fps=30)


if __name__ == "__main__":
    # Visualize comparison. 
    # vis_cmp_res() 

    # Visualize long-term seq in 3D scene 
    vis_res_in_3d_scene() 
