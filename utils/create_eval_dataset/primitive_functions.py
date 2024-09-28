import os 
import trimesh 
import numpy as np 
import json 

import torch 

'''
Designed. 
1. Sample a point on the top surface of an object (table, desk, counter, etc). 
2. Sample a point under an object and on the floor (table, desk, counter, etc). 
'''

def sample_pts_from_top_surface_of_object(scene_pts, class_labels, object_label, border_buffer=0.1, \
    num_samples=30):
    """
    Sample a point from the top surface of the given object.

    Parameters:
    - scene_pts: Ns x 3 numpy array of 3D coordinates.
    - class_labels: Ns x 1 array of semantic labels.
    - object_label: label corresponding to the object.
    - border_buffer: buffer distance from the borders of the object.

    Returns:
    - A 3D point from the top surface of the object or None if no point found.
    """
    class_labels = np.asarray(class_labels)
    
    # Extract object points
    object_points = scene_pts[class_labels == object_label]

    # If no object points found, return None
    if len(object_points) == 0:
        return None

    # Compute the bounding box of the object
    min_coords = np.min(object_points, axis=0)
    max_coords = np.max(object_points, axis=0)

    # Define the region for the top surface
    z_top_surface = max_coords[2]
    x_min = min_coords[0] + border_buffer
    x_max = max_coords[0] - border_buffer
    y_min = min_coords[1] + border_buffer
    y_max = max_coords[1] - border_buffer

    # Filter object points that are on the top surface considering the border buffer
    top_surface_points = object_points[np.logical_and.reduce((
        object_points[:, 0] >= x_min,
        object_points[:, 0] <= x_max,
        object_points[:, 1] >= y_min,
        object_points[:, 1] <= y_max,
        np.isclose(object_points[:, 2], z_top_surface)  # points close to the maximum Z-coordinate
    ))]

    # If no top surface points found, return None
    if len(top_surface_points) == 0:
        return None

    # Sample a point randomly from the top surface points
    sampled_point = top_surface_points[np.random.choice(len(top_surface_points), size=num_samples, replace=True)]

    return sampled_point.reshape(-1, 3)

def sample_pts_under_object(scene_pts, class_labels, object_label, num_samples=30):
    """
    Sample a point under the given object.

    Parameters:
    - scene_pts: Ns x 3 numpy array of 3D coordinates.
    - class_labels: Ns x 1 array of semantic labels.
    - object_label: label corresponding to the object.

    Returns:
    - A 3D point under the object or None if no point found.
    """
    
    # Define the label for the floor, adjust based on your label encoding
    floor_label = "floor"
    
    class_labels = np.asarray(class_labels)

    # Extract object points and floor points
    object_points = scene_pts[class_labels == object_label]
    floor_points = scene_pts[class_labels == floor_label]

    # If no object points or floor points found, return None
    if len(object_points) == 0 or len(floor_points) == 0:
        return None

    # Compute the bounding box of the object
    min_coords = np.min(object_points, axis=0)
    max_coords = np.max(object_points, axis=0)

    # Filter floor points that are under the object using the bounding box
    under_object_floor_points = floor_points[np.logical_and(
        floor_points[:, 0] >= min_coords[0],
        floor_points[:, 0] <= max_coords[0]
    ) & np.logical_and(
        floor_points[:, 1] >= min_coords[1],
        floor_points[:, 1] <= max_coords[1]
    ) & np.logical_and(
        floor_points[:, 2] <= min_coords[2],  # Points are "under" the object if their z-coordinate is less than the minimum z-coordinate of the object
        True
    )]

    # If no floor points found under the object, return None
    if len(under_object_floor_points) == 0:
        return None

    # Sample a point randomly from the points under the object
    sampled_point = under_object_floor_points[np.random.choice(len(under_object_floor_points), size=num_samples, replace=True)]

    return sampled_point.reshape(-1, 3)

def sample_pts_between_objects(scene_pts, class_labels, object_label_a, object_label_b, num_samples=30):
    """
    Sample a point between the two given objects.

    Parameters:
    - scene_pts: Ns x 3 numpy array of 3D coordinates.
    - class_labels: Ns x 1 array of semantic labels.
    - object_label_a, object_label_b: labels corresponding to the objects.

    Returns:
    - A 3D point between the objects or None if no point found.
    """
    
    # Define the label for the floor, adjust based on your label encoding
    floor_label = "floor"

    class_labels = np.asarray(class_labels)
    
    # Extract points for the two objects and the floor
    object_a_points = scene_pts[class_labels == object_label_a]
    object_b_points = scene_pts[class_labels == object_label_b]
    floor_points = scene_pts[class_labels == floor_label]

    # If no points found for either object or the floor, return None
    if len(object_a_points) == 0 or len(object_b_points) == 0 or len(floor_points) == 0:
        return None

    # Compute the bounding boxes of the two objects
    min_coords_a = np.min(object_a_points, axis=0)
    max_coords_a = np.max(object_a_points, axis=0)

    min_coords_b = np.min(object_b_points, axis=0)
    max_coords_b = np.max(object_b_points, axis=0)

    # Identify the axis with the largest distance between the two objects
    axis_distances = np.array([
        min_coords_b[0] - max_coords_a[0],
        min_coords_b[1] - max_coords_a[1],
        min_coords_b[2] - max_coords_a[2]
    ])
    axis_idx = np.argmax(np.abs(axis_distances))

    # Filter floor points that are between the two objects
    between_floor_points = floor_points[np.logical_and(
        floor_points[:, axis_idx] >= max_coords_a[axis_idx],
        floor_points[:, axis_idx] <= min_coords_b[axis_idx]
    )]

    # If no floor points found between the objects, return None
    if len(between_floor_points) == 0:
        return None

    # Sample a point randomly from the points between the objects
    sampled_point = between_floor_points[np.random.choice(len(between_floor_points), size=num_samples, replace=True)]

    return sampled_point.reshape(-1, 3)

def sample_pts_near_to_object(scene_pts, class_labels, object_label, proximity_distance=0.5, num_samples=30):
    """
    Sample a point near the given object.

    Parameters:
    - scene_pts: Ns x 3 numpy array of 3D coordinates.
    - class_labels: Ns x 1 array of semantic labels.
    - object_label: label corresponding to the object.
    - proximity_distance: distance threshold to consider a point near to the object.

    Returns:
    - A 3D point near the object or None if no point found.
    """
    
    # Define the label for the floor, adjust based on your label encoding
    floor_label = "floor"

    class_labels = np.asarray(class_labels)
    
    # Extract object points and floor points
    object_points = scene_pts[class_labels == object_label]
    floor_points = scene_pts[class_labels == floor_label]

    # If no object points or floor points found, return None
    if len(object_points) == 0 or len(floor_points) == 0:
        return None

    # Compute the bounding box of the object
    min_coords = np.min(object_points, axis=0)
    max_coords = np.max(object_points, axis=0)

    # Expand the bounding box by the proximity distance
    expanded_min_coords = min_coords - [proximity_distance, proximity_distance, 0]  # don't expand vertically
    expanded_max_coords = max_coords + [proximity_distance, proximity_distance, 0]

    # Filter floor points that are within the expanded bounding box but not within the original bounding box of the object
    near_floor_points = floor_points[np.logical_and(
        floor_points[:, 0] >= expanded_min_coords[0],
        floor_points[:, 0] <= expanded_max_coords[0]
    ) & np.logical_and(
        floor_points[:, 1] >= expanded_min_coords[1],
        floor_points[:, 1] <= expanded_max_coords[1]
    ) & np.logical_and(
        floor_points[:, 2] >= expanded_min_coords[2],
        floor_points[:, 2] <= expanded_max_coords[2]
    )]

    # Exclude floor points that are within the original bounding box of the object (i.e., colliding with the object)
    non_colliding_floor_points = near_floor_points[np.logical_or(
        near_floor_points[:, 0] < min_coords[0],
        near_floor_points[:, 0] > max_coords[0]
    ) | np.logical_or(
        near_floor_points[:, 1] < min_coords[1],
        near_floor_points[:, 1] > max_coords[1]
    )]

    # Ensure the non-colliding points aren't part of any other object class
    valid_indices = np.isin(class_labels, [floor_label])
    valid_floor_points = non_colliding_floor_points[np.all(np.isin(non_colliding_floor_points, scene_pts[valid_indices]), axis=1)]

    # If no valid floor points found near the object, return None
    if len(valid_floor_points) == 0:
        return None

    # Sample a point randomly from the valid floor points near the object
    sampled_point = valid_floor_points[np.random.choice(len(valid_floor_points), size=num_samples, replace=True)]

    return sampled_point.reshape(-1, 3)

def sample_point_left_of_sofa(points, labels, label_sofa):
    """
    Sample a point that is on the left side of the sofa.

    Parameters:
    - points: Nx3 numpy array of 3D coordinates.
    - labels: Nx1 array of semantic labels.
    - label_sofa: label corresponding to the sofa.

    Returns:
    - A 3D point on the left side of the sofa or None if no point found.
    """

    # Example usage:
    # points = np.array([...])  # Nx3 array of 3D points
    # labels = np.array([...])  # Nx1 array of labels
    # label_sofa = "sofa"  # Adjust based on your label encoding
    # sampled_point = sample_point_left_of_sofa(points, labels, label_sofa)

    # Extract sofa points
    sofa_points = points[labels == label_sofa]

    # If no sofa points found, return None
    if len(sofa_points) == 0:
        return None

    # Calculate the mean (center) and PCA (to find orientation) of the sofa
    mean = np.mean(sofa_points, axis=0)
    centered_points = sofa_points - mean
    _, _, Vt = np.linalg.svd(centered_points)

    # The first principal component provides the main direction of the sofa
    main_direction = Vt[0]

    # Calculate a vector pointing "left" from the main direction of the sofa.
    # This is a bit arbitrary, but assuming Y is up, we can take the cross product
    # with the up direction to get a vector pointing left or right.
    up = np.array([0, 1, 0])
    left_direction = np.cross(main_direction, up)

    # Project all points onto this left direction
    projections = np.dot(points, left_direction)

    # Find points that are on the "left side" of the sofa.
    # We do this by looking for points that have a higher projection value
    # than the mean projection value of the sofa.
    mean_projection = np.dot(mean, left_direction)
    left_points = points[projections > mean_projection]

    # If no left points found, return None
    if len(left_points) == 0:
        return None

    # Sample a point randomly
    sampled_point = left_points[np.random.choice(len(left_points))]

    return sampled_point

def convert_pt_to_habitat_coord(pts_data):
    # pts_data: N X 3 
    x_data = pts_data[:, 0][:, None] # N X 1
    y_data = pts_data[:, 2][:, None] # N X 1
    z_data = -pts_data[:, 1][:, None] # N X 1 

    pts_in_habitat = torch.cat((x_data, y_data, z_data), dim=-1) # N X 3 

    return pts_in_habitat 

def compute_normal(v1, v2, v3):
    return np.cross(v2-v1, v3-v1)

def triangle_area(v1, v2, v3):
    """Compute the area of a triangle defined by vertices v1, v2, and v3."""
    return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1))

def sample_point_in_triangle(v1, v2, v3):
    """Randomly sample a point within a triangle defined by vertices v1, v2, v3."""
    # Barycentric coordinates
    s, t = sorted([np.random.rand(), np.random.rand()])
    return s * v1 + (t-s)*v2 + (1-t)*v3

def sample_point_on_desk_top(vertices, faces, labels, num_samples=20):
    """Randomly sample a point from the top surface of the desk."""
    
    # Filter faces that belong to the label "table"
    target_faces = [face for face, label in zip(faces, labels) if label == "table"]

    # Determine the maximum height (Z-coordinate) among the desk vertices
    desk_vertices = [vertices[f[i]] for f in target_faces for i in range(3)]
    max_height = max(vertex[2] for vertex in desk_vertices)
    
    # Use a tolerance to account for floating point discrepancies
    tolerance = 0.1

    # Filter the faces that are on the top surface based on their height
    top_faces = [face for face in target_faces if all(vertices[face[i]][2] >= max_height - tolerance for i in range(3))]
    
    # Compute the area of each triangle (face)
    areas = np.array([triangle_area(vertices[f[0]], vertices[f[1]], vertices[f[2]]) for f in top_faces])

    # Normalize areas to get the probability distribution
    areas /= areas.sum()

    # Randomly choose a triangle (face) based on the areas
    sampled_pts_list = []
    for s_idx in range(num_samples):
        chosen_face = top_faces[np.random.choice(len(top_faces), p=areas)]

        # Randomly sample a point within the chosen triangle
        v1 = vertices[chosen_face[0]]
        v2 = vertices[chosen_face[1]]
        v3 = vertices[chosen_face[2]]

        curr_pt = sample_point_in_triangle(v1, v2, v3)

        sampled_pts_list.append(curr_pt)

    return sampled_pts_list

def sample_point_in_rectangle(rect, num_samples):
    """Randomly sample a point within a given rectangle."""
    min_corner, max_corner = rect
    pts_list = []
    for s_idx in range(num_samples):
        pts_list.append([np.random.uniform(min_corner[0], max_corner[0]), 
            np.random.uniform(min_corner[1], max_corner[1]), 
            min_corner[2]])

    return pts_list 

def sample_point_on_floor_under_table(vertices, faces, labels, num_samples=20):
    # Get the AABB of the desk's top surface
    desk_faces = [face for face, label in zip(faces, labels) if label == "table" and compute_normal(vertices[face[0]], vertices[face[1]], vertices[face[2]])[2] > 0]
    desk_top_vertices = [vertices[face[i]] for face in desk_faces for i in range(3)]
    min_corner = np.min(desk_top_vertices, axis=0)
    max_corner = np.max(desk_top_vertices, axis=0)
    
    # Find the floor's z-value using the labeled faces. Assuming uniform floor height
    floor_faces = [face for face, label in zip(faces, labels) if label == "floor"]
    floor_z = np.min([vertices[face[0]][2] for face in floor_faces])  # Taking the minimum z-value from the floor vertices
    
    # Project AABB onto floor
    projected_rect = [[min_corner[0], min_corner[1], floor_z], [max_corner[0], max_corner[1], floor_z]]
    
    return sample_point_in_rectangle(projected_rect, num_samples) 

def extract_object_id_for_faces(scene_mesh):
    scene_data = scene_mesh.metadata['_ply_raw']['face']['data']
    num_faces = scene_data.shape[0]
    
    obj_id_list = []
    scene_faces_list = []
    for idx in range(num_faces):
        curr_obj_id = scene_data[idx][1]
        curr_face = scene_data[idx][0][1] # array([0, 1, 2, 3], dtype=uint32)

        obj_id_list.append(curr_obj_id)
        scene_faces_list.append(curr_face) 

    obj_id_list = np.asarray(obj_id_list)
    scene_faces_list = np.asarray(scene_faces_list) 

    return scene_faces_list, obj_id_list 

def assign_semantics_for_obj_ids(json_path, object_ids_list):
    json_data = json.load(open(json_path, 'r'))

    # Generate class labels dict.
    class_dict = {}
    class_data = json_data['classes']
    num_class = len(class_data)
    for c_idx in range(num_class):
        curr_c_id = class_data[c_idx]['id']
        curr_c_name = class_data[c_idx]['name']
        if curr_c_id not in class_dict:
            class_dict[curr_c_id] = curr_c_name 

    class_names_list = []

    object_dict = {}
    objects_data = json_data['objects']
    num_objects = len(objects_data)
    for idx in range(num_objects):
        curr_obj_id = objects_data[idx]['id']
        curr_class_id = objects_data[idx]['class_id']

        if curr_class_id in class_dict:
            curr_class_name = class_dict[curr_class_id] 
        else:
            curr_class_name = "none" 

        class_names_list.append(curr_class_name)

        if curr_obj_id not in object_dict:
            object_dict[curr_obj_id] = curr_class_name 

    # Assign object semantic names to each face's obejct ids
    object_names_list = []
    num_faces = len(object_ids_list)
    for f_idx in range(num_faces):
        curr_id = object_ids_list[f_idx]
        if curr_id in object_dict:
            object_names_list.append(object_dict[curr_id]) 
        else:
            object_names_list.append("none") 

    return object_names_list 
    