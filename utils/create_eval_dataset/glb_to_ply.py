import subprocess
import sys
import os 

def convert_glb_to(blender_path, input_file, output_file, format):
    python_script = f'''
import bpy
# Clear all existing objects in the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import the GLB file
bpy.ops.import_scene.gltf(filepath=r"{input_file}")

# Apply a rotation if needed (optional)
# for obj in bpy.context.selected_objects:
#     obj.rotation_euler[0] = 1.5708  # 90 degrees in radians

# Export to the desired format
bpy.ops.export_mesh.{format}(filepath=r"{output_file}")
'''

    # Blender command to run in background, load a default scene, and execute the script
    cmd = [
        blender_path,
        '--background',
        '--factory-startup',
        '--python-expr',
        python_script
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("Error occurred during conversion.")
        sys.exit(1)

if __name__ == "__main__":
    # Path to Blender executable (Modify if required)
    BLENDER_PATH = "/viscam/u/jiamanli/blender-3.6.3-linux-x64/blender"

    scene_root_folder = "/move/u/jiamanli/datasets/semantic_manip/scene_data/hm3d-val-habitat-v0.2"
    scene_names = os.listdir(scene_root_folder)
    for scene_n in scene_names:
        print("scene name:{0}".format(scene_n))
        scene_glb_path = os.path.join(scene_root_folder, scene_n, scene_n.split("-")[1]+".basis.glb")
        dest_scene_obj_path = os.path.join(scene_root_folder, scene_n, scene_n.split("-")[1]+".basis.ply")

        # Conversion format ('obj' or 'ply')
        conversion_format = 'ply'

        # if not os.path.exists(dest_scene_obj_path):
        convert_glb_to(BLENDER_PATH, scene_glb_path, dest_scene_obj_path, conversion_format)
        print(f"Conversion to .{conversion_format} completed!")
