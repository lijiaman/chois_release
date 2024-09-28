import bpy
import os

def rotate_and_export_ply(input_filepath, output_filepath):
    # Clear existing data
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Import the OBJ
    bpy.ops.import_scene.obj(filepath=input_filepath)

    # Select the imported object
    obj_name = os.path.basename(input_filepath).split('.')[0]
    obj = bpy.data.objects.get(obj_name) or bpy.context.active_object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Rotate 90 degrees around the x-axis (in radians)
    obj.rotation_euler[0] = 1.5708  # 90 degrees in radians

    # Export the rotated object as PLY
    bpy.ops.export_mesh.ply(filepath=output_filepath)

root_data_folder = "/move/u/jiamanli/datasets/semantic_manip/unseen_objects_data/selected_unseen_objects"
ori_obj_folder = os.path.join(root_data_folder, "selected_obj_obj_files")
dest_obj_folder = os.path.join(root_data_folder, "selected_rotated_obj_files")
if not os.path.exists(dest_obj_folder):
    os.makedirs(dest_obj_folder)

obj_names = os.listdir(ori_obj_folder)
for obj_name in obj_names: 
    ori_obj_path = os.path.join(ori_obj_folder, obj_name) 
    dest_obj_path = os.path.join(dest_obj_folder, obj_name.replace(".obj", ".ply"))
  
    rotate_and_export_ply(ori_obj_path, dest_obj_path) 
