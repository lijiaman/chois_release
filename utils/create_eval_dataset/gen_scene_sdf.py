import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from mesh_to_sdf import mesh_to_voxels
from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import skimage

import numpy as np 
import time 
import json 

def gen_sdf(): 
    dest_root_folder = "/move/u/jiamanli/datasets/semantic_manip/scene_data/hm3d_processed"
    dest_voxel_mesh_folder = os.path.join(dest_root_folder, "hm3d_voxel_objs_res256")
    dest_sdf_folder = os.path.join(dest_root_folder, "hm3d_sdfs_res256") 
    if not os.path.exists(dest_voxel_mesh_folder):
        os.makedirs(dest_voxel_mesh_folder)
    if not os.path.exists(dest_sdf_folder):
        os.makedirs(dest_sdf_folder)

    data_root_folder = "/move/u/jiamanli/datasets/semantic_manip/scene_data/hm3d-val-habitat-v0.2"
    scene_names = os.listdir(data_root_folder)
    for scene_n in scene_names:
        scene_folder = os.path.join(data_root_folder, scene_n)

        obj_path = os.path.join(scene_folder, scene_n.split("-")[1]+".basis.ply")
        import pdb 
        pdb.set_trace() 
        mesh = trimesh.load(obj_path)

        # Save centroid and extents data used for transforming vertices to [-1,1] while query
        # vertices = mesh.vertices - mesh.bounding_box.centroid
        # vertices *= 2 / np.max(mesh.bounding_box.extents)
        centroid = mesh.bounding_box.centroid
        extents = mesh.bounding_box.extents

        dest_json_path = os.path.join(dest_sdf_folder, scene_n + "_sdf_info.json")
        json_dict = {}
        json_dict['centroid'] = centroid.tolist()
        json_dict['extents'] = extents.tolist() 
        json.dump(json_dict, open(dest_json_path, 'w'))

        # Calculating voxels and save to obj for checking
        voxel_mesh = os.path.join(dest_voxel_mesh_folder, scene_n + "_voxel_sample.obj") 

        # Calculating sdf 
        start_time = time.time() 
        # points, sdf = sample_sdf_near_surface(mesh, number_of_points=256*256*256)
        # points, sdf = sample_sdf_near_surface(mesh, number_of_points=512*512*512)
        
        sdf = mesh_to_voxels(mesh, voxel_resolution=256, \
            surface_point_method='sample', sign_method='normal', \
            scan_count=100, scan_resolution=400, sample_point_count=10000000, \
            normal_sample_count=11, pad=False, check_result=False, return_gradients=False)

        vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        w_f = open(voxel_mesh, 'w')
        mesh.export(w_f, file_type='obj')
        
        print("Generating sdf takes:{0} seconds".format(time.time()-start_time))
        
        dest_sdf_path = os.path.join(dest_sdf_folder, scene_n + "_sdf.npy") 
        np.save(dest_sdf_path, sdf)

if __name__ == "__main__":
    gen_sdf() 
