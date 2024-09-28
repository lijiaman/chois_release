import math
import os
import random
import sys

import imageio
import magnum as mn
import numpy as np

# %matplotlib inline
from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations import maps

# %%
# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, dest_vis_path, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    # plt.show(block=False)
    plt.savefig(dest_vis_path) 

# Change to do something like this maybe: https://stackoverflow.com/a/41432704
def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), dest_vis_path=None):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    # plt.show(block=False)
    plt.savefig(dest_vis_path)

def check_map(sim, output_path=None, display=False):
    # @markdown ###Configure Example Parameters:
    # @markdown Configure the map resolution:
    meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
    # @markdown ---
    # @markdown Customize the map slice height (global y coordinate):
    custom_height = False  # @param {type:"boolean"}
    height = 1  # @param {type:"slider", min:-10, max:10, step:0.1}
    # @markdown If not using custom height, default to scene lower limit.
    # @markdown (Cell output provides scene height range from bounding box for reference.)

    print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
    if not custom_height:
        # get bounding box minumum elevation for automatic height
        height = sim.pathfinder.get_bounds()[0][1]

    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        # @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
        # This map is a 2D boolean array
        sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)

        if display:
            # @markdown Alternatively, you can process the map using the Habitat-Lab [maps module](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/maps.py)
            hablab_topdown_map = maps.get_topdown_map(
                sim.pathfinder, height, meters_per_pixel=meters_per_pixel
            )
            recolor_map = np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )
            hablab_topdown_map = recolor_map[hablab_topdown_map]
            print("Displaying the raw map from get_topdown_view:")
            raw_map_path = os.path.join(output_path, "raw_map_topdown_view.png")
            display_map(sim_topdown_map, raw_map_path)
            print("Displaying the map from the Habitat-Lab maps module:")
            hablab_map_path = os.path.join(output_path, "hablab_map_topdown_view.png")
            display_map(hablab_topdown_map, hablab_map_path)

            # easily save a map to file:
            map_filename = os.path.join(output_path, "top_down_map.png")
            imageio.imsave(map_filename, hablab_topdown_map)

# %%
# @markdown ## Querying the NavMesh
def query_navigation(sim, output_path=None, display=False):
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        # @markdown NavMesh area and bounding box can be queried via *navigable_area* and *get_bounds* respectively.
        print("NavMesh area = " + str(sim.pathfinder.navigable_area))
        print("Bounds = " + str(sim.pathfinder.get_bounds()))

        # @markdown A random point on the NavMesh can be queried with *get_random_navigable_point*.
        pathfinder_seed = 1  # @param {type:"integer"}
        sim.pathfinder.seed(pathfinder_seed)
        nav_point = sim.pathfinder.get_random_navigable_point()
        print("Random navigable point : " + str(nav_point))
        print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))

        # @markdown The radius of the minimum containing circle (with vertex centroid origin) for the isolated navigable island of a point can be queried with *island_radius*.
        # @markdown This is analogous to the size of the point's connected component and can be used to check that a queried navigable point is on an interesting surface (e.g. the floor), rather than a small surface (e.g. a table-top).
        print("Nav island radius : " + str(sim.pathfinder.island_radius(nav_point)))

        # @markdown The closest boundary point can also be queried (within some radius).
        max_search_radius = 2.0  # @param {type:"number"}
        print(
            "Distance to obstacle: "
            + str(sim.pathfinder.distance_to_closest_obstacle(nav_point, max_search_radius))
        )
        hit_record = sim.pathfinder.closest_obstacle_surface_point(
            nav_point, max_search_radius
        )
        print("Closest obstacle HitRecord:")
        print(" point: " + str(hit_record.hit_pos))
        print(" normal: " + str(hit_record.hit_normal))
        print(" distance: " + str(hit_record.hit_dist))

        vis_points = [nav_point]

        # HitRecord will have infinite distance if no valid point was found:
        if math.isinf(hit_record.hit_dist):
            print("No obstacle found within search radius.")
        else:
            # @markdown Points near the boundary or above the NavMesh can be snapped onto it.
            perturbed_point = hit_record.hit_pos - hit_record.hit_normal * 0.2
            print("Perturbed point : " + str(perturbed_point))
            print(
                "Is point navigable? " + str(sim.pathfinder.is_navigable(perturbed_point))
            )
            snapped_point = sim.pathfinder.snap_point(perturbed_point)
            print("Snapped point : " + str(snapped_point))
            print("Is point navigable? " + str(sim.pathfinder.is_navigable(snapped_point)))
            vis_points.append(snapped_point)

        # @markdown ---
        # @markdown ### Visualization
        # @markdown Running this cell generates a topdown visualization of the NavMesh with sampled points overlayed.
        meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}

        if display:
            xy_vis_points = convert_points_to_topdown(
                sim.pathfinder, vis_points, meters_per_pixel
            )
            # use the y coordinate of the sampled nav_point for the map height slice
            top_down_map = maps.get_topdown_map(
                sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel
            )
            recolor_map = np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )
            top_down_map = recolor_map[top_down_map]
            print("\nDisplay the map with key_point overlay:")
            dest_vis_path = os.path.join(output_path, "quey_navigation.png")
            display_map(top_down_map, dest_vis_path, key_points=xy_vis_points)

def plan_a_path(sim, agent, sample2_pt=None, start_pt=None, output_path=None, display=False):
    # %%
    # @markdown ## Pathfinding Queries on NavMesh

    # @markdown The shortest path between valid points on the NavMesh can be queried as shown in this example.

    # @markdown With a valid PathFinder instance:
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        while True:
            # seed = 4  # @param {type:"integer"}
            seed = random.sample(list(range(9999)), 1)[0]
            sim.pathfinder.seed(seed)

            # fmt off
            # @markdown 1. Sample valid points on the NavMesh for agent spawn location and pathfinding goal.
            # fmt on
            sample1 = sim.pathfinder.get_random_navigable_point()

            if sample2_pt is None:
                sample2 = sim.pathfinder.get_random_navigable_point()
            else:
                sample2 = sample2_pt.data.cpu().numpy()  

            # import pdb 
            # pdb.set_trace() 

            # @markdown 2. Use ShortestPath module to compute path between samples.
            path = habitat_sim.ShortestPath()
            path.requested_start = sample1
            path.requested_end = sample2
            found_path = sim.pathfinder.find_path(path)
            geodesic_distance = path.geodesic_distance
            path_points = path.points

            # Add start and end to path_points for visualization debug. 
            path_points.insert(0, sample1)
            path_points.append(sample2)

            # @markdown - Success, geodesic path length, and 3D points can be queried.
            print("found_path : " + str(found_path))
            print("geodesic_distance : " + str(geodesic_distance))
            print("path_points : " + str(path_points))

            # @markdown 3. Display trajectory (if found) on a topdown map of ground floor
            if found_path:
                # meters_per_pixel = 0.1
                meters_per_pixel = 0.025
                # scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
                # height = scene_bb.y().min

                height = sim.pathfinder.get_bounds()[0][1]
                if display:
                    top_down_map = maps.get_topdown_map(
                        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
                    )
                    recolor_map = np.array(
                        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                    )
                    top_down_map = recolor_map[top_down_map]
                    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
                    # convert world trajectory points to maps module grid points
                    trajectory = [
                        maps.to_grid(
                            path_point[2],
                            path_point[0],
                            grid_dimensions,
                            pathfinder=sim.pathfinder,
                        )
                        for path_point in path_points
                    ]
                    grid_tangent = mn.Vector2(
                        trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
                    )
                    path_initial_tangent = grid_tangent / grid_tangent.length()
                    initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
                    # draw the agent and trajectory on the map
                    maps.draw_path(top_down_map, trajectory)
                
                    # maps.draw_agent(
                    #     top_down_map, trajectory[0], initial_angle, agent_radius_px=8
                    # )
                    print("\nDisplay the map with agent and path overlay:")
                    # display_map(top_down_map, dest_vis_path, key_points=trajectory)  
                    display_map(top_down_map, output_path)   

                    # Save waypoints to numpy file 
                    dest_npy_path = output_path.replace(".png", ".npy")
                    np.save(dest_npy_path, np.asarray(path_points))

                # @markdown 4. (optional) Place agent and render images at trajectory points (if found).
                display_path_agent_renders = False   # @param{type:"boolean"}
                if display_path_agent_renders:
                    print("Rendering observations at path points:")
                    tangent = path_points[1] - path_points[0]
                    agent_state = habitat_sim.AgentState()
                    for ix, point in enumerate(path_points):
                        if ix < len(path_points) - 1:
                            tangent = path_points[ix + 1] - point
                            agent_state.position = point
                            tangent_orientation_matrix = mn.Matrix4.look_at(
                                point, point + tangent, np.array([0, 1.0, 0])
                            )
                            tangent_orientation_q = mn.Quaternion.from_matrix(
                                tangent_orientation_matrix.rotation()
                            )
                            agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                            agent.set_state(agent_state)

                            observations = sim.get_sensor_observations()
                            rgb = observations["color_sensor"]
                            semantic = observations["semantic_sensor"]
                            depth = observations["depth_sensor"]

                            if display:
                                dest_agent_obs_path = os.path.join(output_path, "dest_agent_obs.png")
                                display_sample(rgb, semantic, depth, dest_agent_obs_path)

                break # If found a path, then stop. 
   
def plan_a_path_for_multiple_objs(sim, agent, start_pt_list, target_pt_list, s_idx, output_path=None, display=False):
    # Prepare 
    sampled_pt_list = []
    for tmp_idx in range(len(start_pt_list)):
        if tmp_idx > 0:
            sampled_pt_list.append(target_pt_list[tmp_idx-1][s_idx]) 
            sampled_pt_list.append(start_pt_list[tmp_idx][s_idx]) 

        sampled_pt_list.append(start_pt_list[tmp_idx][s_idx]) # 1 X 3 
        sampled_pt_list.append(target_pt_list[tmp_idx][s_idx])
    
    num_sub_paths = len(sampled_pt_list)//2 

    whole_seq_found_flag = True 
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        while True:
            for p_idx in range(num_sub_paths):  
                path_points = [] 

                sample1 = sampled_pt_list[p_idx*2] 
                sample2 = sampled_pt_list[p_idx*2+1]
                path = habitat_sim.ShortestPath()
                path.requested_start = sample1 
                path.requested_end = sample2
                found_path = sim.pathfinder.find_path(path)
                geodesic_distance = path.geodesic_distance
                curr_path_points = path.points

                # Add start and end to path_points for visualization debug. 
                curr_path_points.insert(0, sample1)
                curr_path_points.append(sample2)

                path_points.extend(curr_path_points)

                # @markdown - Success, geodesic path length, and 3D points can be queried.
                print("found_path : " + str(found_path))
                # print("geodesic_distance : " + str(geodesic_distance))
                # print("path_points : " + str(path_points))

                if not found_path:
                    whole_seq_found_flag = False 

            # import pdb 
            # pdb.set_trace() 
                # @markdown 3. Display trajectory (if found) on a topdown map of ground floor
            # if whole_seq_found_flag:
                # meters_per_pixel = 0.1
                meters_per_pixel = 0.025
                # scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
                # height = scene_bb.y().min

                height = sim.pathfinder.get_bounds()[0][1]
                if display:
                    top_down_map = maps.get_topdown_map(
                        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
                    )
                    recolor_map = np.array(
                        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                    )
                    top_down_map = recolor_map[top_down_map]
                    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
                    # convert world trajectory points to maps module grid points
                    trajectory = [
                        maps.to_grid(
                            path_point[2],
                            path_point[0],
                            grid_dimensions,
                            pathfinder=sim.pathfinder,
                        )
                        for path_point in path_points
                    ]
                    grid_tangent = mn.Vector2(
                        trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
                    )
                    path_initial_tangent = grid_tangent / grid_tangent.length()
                    initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
                    # draw the agent and trajectory on the map
                    maps.draw_path(top_down_map, trajectory)
                
                    # maps.draw_agent(
                    #     top_down_map, trajectory[0], initial_angle, agent_radius_px=8
                    # )
                    print("\nDisplay the map with agent and path overlay:")
                    # display_map(top_down_map, dest_vis_path, key_points=trajectory)  
                    display_map(top_down_map, output_path.replace(".png", "_"+str(p_idx)+".png"))   

                    # Save waypoints to numpy file 
                    dest_npy_path = output_path.replace(".png", "_"+str(p_idx)+".npy")
                   
                    np.save(dest_npy_path, np.asarray(path_points))

            break 

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def get_sim_and_agent(scene_name):
    data_folder = "/move/u/jiamanli/datasets/replica"

    test_scene = os.path.join(data_folder, scene_name, "habitat/mesh_semantic.ply")

    sim_settings = {
        "scene": test_scene,  # Scene path
        "default_agent": 0,  # Index of the default agent
        "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
        "width": 256,  # Spatial resolution of the observations
        "height": 256,
    }

    cfg = make_simple_cfg(sim_settings)

    # %% [markdown]
    # ### Create a simulator instance

    # %%
    try:  # Needed to handle out of order cell run in Colab
        sim.close()
    except NameError:
        pass
    sim = habitat_sim.Simulator(cfg)

    # the navmesh can also be explicitly loaded
    navmesh_path = os.path.join(data_folder, scene_name, "habitat/mesh_semantic.navmesh")
    sim.pathfinder.load_nav_mesh(
        navmesh_path
    )

    # Initialize an agent. 
    agent = sim.initialize_agent(sim_settings["default_agent"])

    # Set agent state
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
    agent.set_state(agent_state)

    # Get agent state
    agent_state = agent.get_state()
    print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

    return sim, agent 


def gen_path_on_habitat(sim, agent, scene_name, target_pts, output_folder, curr_object_name, text_idx):

    curr_scene_output_folder = os.path.join(output_folder, scene_name, curr_object_name, str(text_idx))
    if not os.path.exists(curr_scene_output_folder):
        os.makedirs(curr_scene_output_folder)
    # check_map(sim, output_path=output_path, display=True)  

    # query_navigation(sim, output_path=output_path, display=True) 
    
    num_samples_per_scene = target_pts.shape[0]
    for s_idx in range(num_samples_per_scene):
        output_path = os.path.join(curr_scene_output_folder, "%04d"%(s_idx)+".png")
        plan_a_path(sim, agent, sample2_pt=target_pts[s_idx], \
        output_path=output_path, display=True)

def gen_path_for_multiple_objs_on_habitat(sim, agent, scene_name, start_pt_list, target_pt_list, \
    output_folder, text_idx):
    # start_pt_list: a list, each element is K X 3.
    # target_pt_list: a list, each element is K X 3.

    curr_scene_output_folder = os.path.join(output_folder, scene_name, str(text_idx))
    if not os.path.exists(curr_scene_output_folder):
        os.makedirs(curr_scene_output_folder)
    # check_map(sim, output_path=output_path, display=True)  

    # query_navigation(sim, output_path=output_path, display=True) 
    
    num_samples_per_scene = start_pt_list[0].shape[0] 
    for s_idx in range(num_samples_per_scene):
        output_path = os.path.join(curr_scene_output_folder, "%04d"%(s_idx)+".png")
        plan_a_path_for_multiple_objs(sim, agent, start_pt_list, target_pt_list, s_idx=s_idx, \
        output_path=output_path, display=True)

''''
Habitat is y-axis up, Blender is z-axis up. 
1. Extract waypoints from Habitat (x,y,z). 
2. Use (x, -z) as waypoints to generate interactions. 
3. During the scene and interaction visualization in Blender, put human mesh to touch the floor (set the translation of z-dim to y.)
'''