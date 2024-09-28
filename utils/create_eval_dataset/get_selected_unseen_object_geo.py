import os 
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np 
import joblib 
import trimesh 
import json 
import time 

import torch 

import shutil 

import subprocess 

def cp_obj_file(obj_img_folder, ori_obj_geo_folder, dest_obj_geo_folder):
    if not os.path.exists(dest_obj_geo_folder):
        os.makedirs(dest_obj_geo_folder)

    img_names = os.listdir(obj_img_folder)
    for img_n in img_names:
        object_tag = img_n.replace(".jpg", "")

        ori_obj_geo_path = os.path.join(ori_obj_geo_folder, object_tag, "raw_model.obj")
        dest_obj_geo_path = os.path.join(dest_obj_geo_folder, object_tag+".obj")

        shutil.copy(ori_obj_geo_path, dest_obj_geo_path)

def generate_sdf(mesh, dest_json_path, dest_sdf_path, dest_voxel_mesh_path="", grid_dim=256, print_time=False):
    from mesh_to_sdf import mesh_to_voxels, sample_sdf_near_surface
    # Save centroid and extents data used for transforming vertices to [-1,1] while query
    # vertices = mesh.vertices - mesh.bounding_box.centroid
    # vertices *= 2 / np.max(mesh.bounding_box.extents)
    centroid = mesh.bounding_box.centroid
    extents = mesh.bounding_box.extents
    # Save centroid and extents as SDF
    json_dict = {}
    json_dict['centroid'] = centroid.tolist()
    json_dict['extents'] = extents.tolist()
    json_dict['grid_dim'] = grid_dim
    json.dump(json_dict, open(dest_json_path, 'w'))
    
    if print_time:
        start_time = time.time() 
    
    sdf = mesh_to_voxels(mesh, voxel_resolution=grid_dim)
    
    if dest_voxel_mesh_path != "":
        import skimage.measure
        vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0)
        voxel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        voxel_mesh.export(open(dest_voxel_mesh_path, 'w'), file_type='obj')
    
    if print_time:
        print("Generating SDF took {0} seconds".format(time.time()-start_time))
    
    np.save(dest_sdf_path, sdf)
    
    centroid = np.copy(centroid)
    extents = np.copy(extents)
    
    return centroid, extents, sdf

def get_objects_sdf(data_root_folder):
    # Load rest object geometry 
    # object_geo_folder = os.path.join(data_root_folder, "selected_unseen_objects", "obj_files")
    # dest_obj_sdf_folder = os.path.join(data_root_folder, "selected_unseen_objects", "sdf_256_npy_files")
    object_geo_folder = os.path.join(data_root_folder, "selected_unseen_objects", "selected_rotated_zeroed_obj_files")
    dest_obj_sdf_folder = os.path.join(data_root_folder, "selected_unseen_objects", "selected_rotated_zeroed_obj_sdf_256_npy_files")
    if not os.path.exists(dest_obj_sdf_folder):
        os.makedirs(dest_obj_sdf_folder)
  
    object_files = os.listdir(object_geo_folder)
    for object_name in object_files:
        if ".ply" in object_name:
            object_ply_path = os.path.join(object_geo_folder, object_name) 
            obj_mesh = trimesh.load_mesh(object_ply_path)
                
            dest_json_path = os.path.join(dest_obj_sdf_folder, object_name.replace(".ply", "")+".json")
            dest_sdf_path = os.path.join(dest_obj_sdf_folder, object_name.replace(".ply", "")+".npy")
            dest_voxel_mesh_path = os.path.join(dest_obj_sdf_folder, object_name.replace(".ply", "")+".obj")

            generate_sdf(obj_mesh, dest_json_path, dest_sdf_path, \
            dest_voxel_mesh_path, grid_dim=256, print_time=False)

def subdivide_and_export(input_filename, output_filename, subdivisions=1):
    """
    Load a mesh from an .obj file, subdivide its vertices, and export the subdivided mesh to another .obj file.

    :param input_filename: Path to the input .obj file.
    :param output_filename: Path where the subdivided mesh should be saved.
    :param subdivisions: Number of subdivisions to apply. 
    """

    # Load the mesh from the input file
    mesh = trimesh.load_mesh(input_filename)

    # Apply the subdivisions
    for _ in range(subdivisions):
        mesh = mesh.subdivide()

    print("Output filename:{0}".format(output_filename.split("/")[-1]))
    print("The number of vertices:{0}".format(mesh.vertices.shape)) 
    # Save the subdivided mesh to the output file
    mesh.export(output_filename)

def get_selected_obj_subdivided(ori_obj_folder, selected_obj_names, dest_obj_folder):
    if not os.path.exists(dest_obj_folder):
        os.makedirs(dest_obj_folder) 
    for object_name in selected_obj_names:
        ori_obj_path = os.path.join(ori_obj_folder, object_name)
        dest_obj_path = os.path.join(dest_obj_folder, object_name)

        subdivide_and_export(ori_obj_path, dest_obj_path, subdivisions=2)
        
def call_blender_to_rotate_obj_on_floor():
    BLENDER_PATH = "/viscam/u/jiamanli/blender-3.6.3-linux-x64/blender"
    blender_utils_path = "/viscam/u/jiamanli/github/scene_aware_manip/cvpr2024_utils/create_eval_dataset/blender_rotate_selected_objects.py"
    subprocess.call(BLENDER_PATH+" -P "+blender_utils_path+\
            " -b", shell=True) 

def zero_obj_com_pos(input_directory, output_directory, dest_json_path):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    json_dict = {} 

    # List all .ply files in the input directory
    ply_files = [f for f in os.listdir(input_directory) if f.endswith('.ply')]

    for file in ply_files:
        # Load the .ply file using trimesh
        mesh = trimesh.load_mesh(os.path.join(input_directory, file))

        # Calculate the center of mass
        center_of_mass = mesh.vertices.mean(axis=0)

        obj_tag = file.replace(".ply", "")
        json_dict[obj_tag] = {}
        json_dict[obj_tag]['com'] = center_of_mass.tolist() 

        # Translate the mesh so that its center of mass is at the origin
        mesh.vertices -= center_of_mass

        # Save the modified mesh to the output directory
        mesh.export(os.path.join(output_directory, file), file_type='ply')

    json.dump(json_dict, open(dest_json_path, 'w'))

if __name__ == "__main__": 
    ori_obj_geo_folder = "/viscam/projects/summon/3D-FUTURE"

    data_root_folder = "/move/u/jiamanli/datasets/semantic_manip/unseen_objects_data"
    obj_img_folder = os.path.join(data_root_folder, "selected_unseen_objects", "imgs")
    dest_obj_geo_folder = os.path.join(data_root_folder, "selected_unseen_objects", "obj_files")
    # cp_obj_file(obj_img_folder, ori_obj_geo_folder, dest_obj_geo_folder) 

    # get_objects_sdf(data_root_folder)
    
    # For the objects with valid SDF, process the mesh to subddivided mesh. 
    selected_sdf_obj_folder = os.path.join(data_root_folder, "selected_unseen_objects", \
            "selected_object_sdf_voxel_objs")
    selected_obj_names = os.listdir(selected_sdf_obj_folder)
    ori_obj_folder = os.path.join(data_root_folder, "selected_unseen_objects", "obj_files")
    dest_obj_folder = os.path.join(data_root_folder, "selected_unseen_objects", "selected_obj_obj_files") 
    # get_selected_obj_subdivided(ori_obj_folder, selected_obj_names, dest_obj_folder)  

    # Rotate the pbject to put it on the floor. 
    # call_blender_to_rotate_obj_on_floor() 

    # Make the com_pos be zero. 
    root_data_folder = "/move/u/jiamanli/datasets/semantic_manip/unseen_objects_data/selected_unseen_objects"
    ori_obj_folder = os.path.join(root_data_folder, "selected_rotated_obj_files")
    dest_obj_folder = os.path.join(root_data_folder, "selected_rotated_zeroed_obj_files")
    dest_json_path = os.path.join(root_data_folder, "selected_object_height.json")
    # zero_obj_com_pos(ori_obj_folder, dest_obj_folder, dest_json_path) 

    get_objects_sdf(data_root_folder) 
