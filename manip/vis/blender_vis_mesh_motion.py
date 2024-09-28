import os 
import subprocess 
import trimesh 
import imageio 
import numpy as np 
import shutil 

BLENDER_PATH = "/viscam/u/jiamanli/blender-3.6.3-linux-x64/blender" # Put your blender path here 
BLENDER_UTILS_ROOT_FOLDER = "/move/u/jiamanli/github/chois_release/manip/vis" # Put the manip/vis folder absolute path here 
BLENDER_SCENE_FOLDER = "/move/u/jiamanli/for_chois_release/processed_data/blender_files" # Put the blender_files folder (where your store .blend files) absolute path here

def concat_multiple_videos(input_files, output_file):
    # List of input files
    # input_files = ['video1.mp4', 'video2.mp4']

    # Output file
    # output_file = 'output.mp4'

    # Step 1: Convert each video to a consistent FPS (e.g., 30 fps) and save to a temp file.
    temp_files = []
    target_fps = 30

    temp_folder = output_file.replace(".mp4", "_tmp")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    for i, file in enumerate(input_files):
        temp_filename = os.path.join(temp_folder, str(i)+".mp4")

        temp_files.append(temp_filename)
        reader = imageio.get_reader(file)
        writer = imageio.get_writer(temp_filename, fps=target_fps)
        
        for frame in reader:
            writer.append_data(frame)
        writer.close()

    # Step 2: Concatenate the temporary files.
    with imageio.get_writer(output_file, fps=target_fps) as final_writer:
        for temp_file in temp_files:
            reader = imageio.get_reader(temp_file)
            for frame in reader:
                final_writer.append_data(frame)

    # Step 3: Cleanup temp files.
    # for temp_file in temp_files:
    #     os.remove(temp_file)
    # #     shutil.rmtree(temp_folder)
    # shutil.rmtree(temp_folder)

def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-r', '30', '-y', '-threads', '16', '-i', f'{img_folder}/%05d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    # command = [
    #     'ffmpeg', '-r', '30', '-y', '-threads', '16', '-i', f'{img_folder}/%05d.png', output_vid_file,
    # ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

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

def run_blender_rendering_and_save2video(obj_folder_path, out_folder_path, out_vid_path, condition_folder=None, \
    scene_blend_path=os.path.join(BLENDER_SCENE_FOLDER, "floor_colorful_mat.blend"), \
    vis_object=False, vis_human=True, vis_hand_and_object=False, vis_gt=False, \
    vis_handpose_and_object=False, hand_pose_path=None, mat_color="blue", vis_condition=False, fps=30):
    
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
        else: # vis object only 
            blender_utils_path = os.path.join(BLENDER_UTILS_ROOT_FOLDER, "blender_vis_object_utils.py") 
            subprocess.call(BLENDER_PATH+" -P "+blender_utils_path+\
                " -b -- --folder "+obj_folder_path+" --scene "+scene_blend_path+" --out-folder "+out_folder_path, shell=True) 
    else: # Vis human only 
        if vis_condition:
            blender_utils_path = os.path.join(BLENDER_UTILS_ROOT_FOLDER, "blender_vis_human_only_w_condition_utils.py") 
            subprocess.call(BLENDER_PATH+" -P "+blender_utils_path+\
            " -b -- --folder "+obj_folder_path+" --condition-folder "+condition_folder+" --scene "+scene_blend_path+" --out-folder "+out_folder_path, shell=True) 
        else:
            blender_utils_path = os.path.join(BLENDER_UTILS_ROOT_FOLDER, "blender_vis_human_utils.py") 
            subprocess.call(BLENDER_PATH+" -P "+blender_utils_path+ \
            " -b -- --folder "+obj_folder_path+" --scene "+\
            scene_blend_path+" --out-folder "+out_folder_path+" --material-color "+mat_color, shell=True)    

    use_ffmpeg = False
    if use_ffmpeg:
        images_to_video(out_folder_path, out_vid_path)
    else:
        # For debug faster visualization 
        # fps = 1
        images_to_video_w_imageio(out_folder_path, out_vid_path, fps=fps)

def save_verts_faces_to_mesh_file(mesh_verts, mesh_faces, save_mesh_folder, save_gt=False):
    # mesh_verts: T X Nv X 3 
    # mesh_faces: Nf X 3 
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    num_meshes = mesh_verts.shape[0]
    for idx in range(num_meshes):
        mesh = trimesh.Trimesh(vertices=mesh_verts[idx],
                        faces=mesh_faces)
        if save_gt:
            curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+"_gt.ply")
        else:
            curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+".ply")
        mesh.export(curr_mesh_path)

def save_verts_faces_to_mesh_file_w_object(mesh_verts, mesh_faces, obj_verts, obj_faces, save_mesh_folder):
    # mesh_verts: T X Nv X 3 
    # mesh_faces: Nf X 3 
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    num_meshes = mesh_verts.shape[0]
    for idx in range(num_meshes):
        mesh = trimesh.Trimesh(vertices=mesh_verts[idx],
                        faces=mesh_faces)
        curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+".ply")
        mesh.export(curr_mesh_path)

        obj_mesh = trimesh.Trimesh(vertices=obj_verts[idx],
                        faces=obj_faces)
        curr_obj_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+"_object.ply")
        obj_mesh.export(curr_obj_mesh_path)
