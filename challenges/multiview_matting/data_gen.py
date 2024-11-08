import math
from pathlib import Path
import logging
import numpy as np
import tensorflow as tf

import kubric as kb
from kubric.renderer import Blender as KubricRenderer
from kubric.renderer import Blender
from kubric import file_io
import bpy

# TODO: go to https://shapenet.org/ create an account and agree to the terms
#       then find the URL for the kubric preprocessed ShapeNet and put it here:
SHAPENET_PATH = "gs://kubric-unlisted/assets/ShapeNetCore.v2.json"

if SHAPENET_PATH == "gs://KUBRIC_SHAPENET_PATH/ShapeNetCore.v2.json":
    raise ValueError("Wrong ShapeNet path. Please visit https://shapenet.org/ "
                                     "agree to terms and conditions, and find the correct path.")


def move_point(A, B, fraction, add_noise=True, noise_std=0.01):
    """
    Move point A towards or away from point B by a specified fraction.

    Parameters:
        A (np.ndarray or tuple): Coordinates of point A.
        B (np.ndarray or tuple): Coordinates of point B.
        fraction (float): The fraction of the distance from A to B to move A.
                          - If 0 < fraction < 1, A moves closer to B.
                          - If fraction > 1, A moves beyond B.
                          - If fraction < 0, A moves away from B.

    Returns:
        np.ndarray: New coordinates of point A after moving.
    """
    # Convert A and B to numpy arrays if they aren't already
    A, B = np.array(A), np.array(B)
    
    # Calculate the direction vector from A to B and normalize it
    direction = B - A
    unit_direction = direction / np.linalg.norm(direction)
    
    # Move A by the specified fraction along the unit direction vector
    new_A = A + fraction * unit_direction

    if add_noise:
        noise = np.random.normal(0, noise_std, 3)
        noise[2] = np.abs(noise[2])
        unit_noise = noise / np.linalg.norm(noise)
        new_A += np.abs(fraction) * noise_std * unit_noise
    
    return new_A

def sample_point_in_half_sphere_shell(
        inner_radius: float,
        outer_radius: float,
        rng: np.random.RandomState
        ):
    """Uniformly sample points that are in a given distance
         range from the origin and with z >= 0."""

    while True:
        v = rng.uniform((-outer_radius, -outer_radius, obj_height/1.2),
                                        (outer_radius, outer_radius, obj_height))
        len_v = np.linalg.norm(v)
        correct_angle = True
        if add_distractors:
            cam_dir = v[:2] / np.linalg.norm(v[:2])
            correct_angle = np.all(np.dot(distractor_dir, cam_dir) < np.cos(np.pi / 9.))
        if inner_radius <= len_v <= outer_radius and correct_angle:
            return tuple(v)

def place_objects_without_overlap(objects, min_distance_multiplier=1.5):
    """ Places objects randomly on the XY plane without overlap. """
    positions = []
    count = 100000
    left= 0.2
    right = 0.8
    for i, obj in enumerate(objects):
        obj_radius = np.linalg.norm(obj.aabbox[1][:1] - obj.aabbox[0][:1])
        min_distance = obj_radius * min_distance_multiplier
        while True:
            count -= 1
            if count == 0:
                print("len objects", len(objects), positions)
                # min_distance_multiplier /= 1.1
                count = 100000
                left += 0.1
                right += 0.1
            if i % 2 == 0:
                position = np.random.uniform(left, right, size=3)
            else:
                position = np.random.uniform(-right, -left, size=3)
                position[1] = np.abs(position[1])
            position[2] = 0.5  # Keep objects on the table
            
            # Check distance from all other placed objects
            if all(np.linalg.norm(position[:2] - pos[:2]) >= min_distance for pos in positions):
                positions.append(position)
                obj.position = position
                break
                
    return positions

def get_camera_position(scene, camera, obj1, obj2, distance=3):
    """ 
    Positions the camera at height z=1 and at a specified distance 
    from the midpoint of the line connecting obj1 and obj2.
    """
    # Calculate the midpoint between obj1 and obj2
    midpoint = (obj1.position[:2] + obj2.position[:2]) / 2

    # Calculate the vector connecting obj1 and obj2
    obj_vector = obj2.position[:2] - obj1.position[:2]

    # Compute the perpendicular (normal) vector to the line between obj1 and obj2
    # This is achieved by swapping coordinates and negating one component
    normal_vector = np.array([-obj_vector[1], obj_vector[0]])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize it

    if normal_vector[0] < 0:
        normal_vector = -normal_vector

    # Position the camera along this normal vector, at a distance of `distance`
    camera_xy_position = midpoint + normal_vector * distance
    return (camera_xy_position[0], camera_xy_position[1], 1.5)


manifest_path = file_io.as_path(SHAPENET_PATH)
manifest = file_io.read_json(manifest_path)
# import pdb; pdb.set_trace()
assets = manifest["assets"]
name = manifest.get("name", manifest_path.stem)  # default to filename
data_dir = manifest.get("data_dir", manifest_path.parent)  # default to manifest dir

# --- CLI arguments (and modified defaults)
parser = kb.ArgumentParser()
parser.set_defaults(
    seed=0,
    frame_start=1,
    frame_end=3,
    resolution=(256, 256),
)

parser.add_argument("--backgrounds_split",
                                        choices=["train", "test"], default="train")
parser.add_argument("--dataset_mode",
                                        choices=["easy", "hard"], default="easy")
parser.add_argument("--hdri_dir",
                                        type=str, default="gs://mv_bckgr_removal/hdri_haven/4k/")
parser.add_argument("--hdri_assets", type=str,
                                        default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")

parser.add_argument("--floor_friction", type=float, default=1.0)
parser.add_argument("--floor_restitution", type=float, default=0.0)
parser.add_argument("--only_scale", type=bool, default=True)

FLAGS = parser.parse_args()


if FLAGS.dataset_mode == "hard":
    add_distractors = True
else:
    add_distractors = False



# --- Fetch a random asset
asset_source = kb.AssetSource.from_manifest(SHAPENET_PATH)
# all_ids = list(asset_source.db['id'])
all_ids = [name for name, unused_spec in asset_source._assets.items()]
num_total_objs = len(all_ids)
fraction = 0.1

rng_train_test_split = np.random.RandomState(1)
rng_train_test_split.shuffle(all_ids)
held_out_obj_ids = all_ids[:math.ceil(fraction * num_total_objs)]

# held_out_obj_ids = list(asset_source.db.sample(
#     frac=fraction, replace=False, random_state=42)["id"])
train_obj_ids = [id for id in all_ids if
                                 id not in held_out_obj_ids]
rng = np.random.RandomState(FLAGS.seed)
for o in range(10000):
    # --- Common setups
    kb.utils.setup_logging(FLAGS.logging_level)
    kb.utils.log_my_flags(FLAGS)
    job_dir = kb.as_path('../rayyeh/data/Ashiq/local_scaling/') #kb.as_path(FLAGS.job_dir)
    
    scene = kb.Scene.from_flags(FLAGS)

    # --- Add a renderer
    renderer = KubricRenderer(scene,
        use_denoising=True,
        adaptive_sampling=False,
        background_transparency=True)

    object_list = []
    
    for k in range(2):
        if FLAGS.backgrounds_split == "train":
            asset_id = rng.choice(train_obj_ids)
        else:
            asset_id = rng.choice(held_out_obj_ids)

        obj = asset_source.create(asset_id=asset_id)
        logging.info(f"selected '{asset_id}'")

        # --- make object flat on X/Y and not penetrate floor
        obj.quaternion = kb.Quaternion(axis=[1,0,0], degrees=90)
        obj.position = obj.position - (0, 0, obj.aabbox[0][2])

        obj_size = np.linalg.norm(obj.aabbox[1] - obj.aabbox[0])

        obj_radius = np.linalg.norm(obj.aabbox[1][:2] - obj.aabbox[0][:2])
        obj_height = np.abs(obj.aabbox[1][2] - obj.aabbox[0][2])
        # Difference along the x-axis gives the width
        obj_width = np.abs(obj.aabbox[1][0] - obj.aabbox[0][0])

        # Difference along the y-axis gives the length (or depth)
        obj_length = np.abs(obj.aabbox[1][1] - obj.aabbox[0][1])



        obj.metadata = {
                "asset_id": obj.asset_id,
                "category": [spec for name, spec in asset_source._assets.items()
                                        if name == obj.asset_id][0]['metadata']["category"],
        }
        obj.scale = (1/(2*obj_size), 1/(2*obj_size), 1/(2*obj_size))
        print("obj_size: ", obj_size, obj_radius, obj_height, obj_width, obj_length)
        scene.add(obj)
        object_list.append(obj)



    # place_objects(object_list[0], object_list[1], 0.3)
    place_objects_without_overlap(object_list, min_distance_multiplier=2.0)

    size_multiple = 1.5

    material = kb.PrincipledBSDFMaterial(
            color=kb.Color.from_hsv(rng.uniform(), 1, 1),
            metallic=1.0, roughness=0.2, ior=2.5)

    table = kb.Cube(name="floor",
                    scale=(1*size_multiple, 1*size_multiple, 0.02),
                    position=(0, 0, -0.02), material=material)
    scene += table

    logging.info("Loading background HDRIs from %s", FLAGS.hdri_dir)

    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)

    train_backgrounds, held_out_backgrounds = hdri_source.get_test_split(
            fraction=0.1)

    if FLAGS.backgrounds_split == "train":
        logging.info("Choosing one of the %d training backgrounds...",
                                len(train_backgrounds))
        background_hdri = hdri_source.create(asset_id=rng.choice(train_backgrounds))
    else:
        logging.info("Choosing one of the %d held-out backgrounds...",
                                len(held_out_backgrounds))
        background_hdri = hdri_source.create(
                asset_id=rng.choice(held_out_backgrounds))


    kubasic = kb.AssetSource.from_manifest("gs://kubric-public/assets/KuBasic/KuBasic.json")
    dome = kubasic.create(asset_id="dome", name="dome",
                                                friction=FLAGS.floor_friction,
                                                restitution=FLAGS.floor_restitution,
                                                static=True, background=True)
    assert isinstance(dome, kb.FileBasedObject)
    dome.scale = (0.8, 0.8, 0.8)
    scene += dome

    blender_renderer = [v for v in scene.views if isinstance(v, Blender)]
    if blender_renderer:
            dome_blender = dome.linked_objects[blender_renderer[0]]
            if bpy.app.version > (3, 0, 0):
                dome_blender.visible_shadow = False
            else:
                dome_blender.cycles_visibility.shadow = False
            if background_hdri is not None:
                dome_mat = dome_blender.data.materials[0]
                texture_node = dome_mat.node_tree.nodes["Image Texture"]
                texture_node.image = bpy.data.images.load(background_hdri.filename)

    renderer._set_ambient_light_hdri(background_hdri.filename)


    # --- Add Klevr-like lights to the scene
    scene += kb.assets.utils.get_clevr_lights(rng=rng)

    # --- Keyframe the camera

    # Call the function with the two objects
    # set_camera_position(scene, scene.camera, object_list[0], object_list[1], distance=3)
    scene.camera = kb.PerspectiveCamera()
    scene.camera.fov = 60
    scene.camera.aspect_ratio = 1.0
    cam_pos = get_camera_position(scene, scene.camera, object_list[0], object_list[1], distance=4)
    
    factor = 1.1
    inv_factor = 1/factor
    for frame in range(FLAGS.frame_start, FLAGS.frame_end + 1):
        print("obj_size", obj_size)
        print("object postion", obj.position)
        scene.camera.position = cam_pos
        if FLAGS.only_scale:
            object_list[0].scale = (object_list[0].scale[0]*factor, object_list[0].scale[1]*factor, object_list[0].scale[2]*factor)
            object_list[1].scale = (object_list[1].scale[0]*inv_factor, object_list[1].scale[1]*inv_factor, object_list[1].scale[2]*inv_factor)
        else:
            object_list[0].position = move_point(object_list[0].position, cam_pos, 0.3)
            object_list[1].position = move_point(object_list[1].position, cam_pos, -0.3)
        
        scene.camera.look_at(((object_list[0].position[0]+object_list[1].position[0])/2,\
            (object_list[0].position[1]+object_list[1].position[1])/2,\
                 np.mean([np.abs(np.abs(obj.aabbox[1][2] - obj.aabbox[0][2])) for obj in object_list])/2))
        
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)
        object_list[0].keyframe_insert("scale", frame)
        object_list[1].keyframe_insert("scale", frame)


    # --- Rendering
    logging.info("Rendering the scene ...")
    renderer.save_state(job_dir / "scene.blend")
    data_stack = renderer.render()

    # --- Postprocessing
    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
            data_stack["segmentation"],
            scene.assets,
            object_list).astype(np.uint8)

    kb.file_io.write_rgba_batch(data_stack["rgba"], job_dir, file_template=str(o)+"_rgba_{:05d}.png")
    kb.file_io.write_segmentation_batch(data_stack["segmentation"], job_dir, file_template=str(o)+"_segmentationn_{:05d}.png")

    # Save depth map
    depth_map = data_stack["depth"]
    kb.file_io.write_depth_batch(depth_map, job_dir, file_template=str(o)+"_depth_{:05d}.tiff")

    # --- Collect metadata
    logging.info("Collecting and storing metadata for each object.")
    data = {
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
    }
    object_metadata = {
        'obj_1': object_list[0].metadata,
        'obj_2': object_list[1].metadata,
    }
    cam_file = str(o)+"_camera_metadata.json"
    obj_file = str(o)+"_object_metadata.json"
    kb.file_io.write_json(filename=job_dir / cam_file, data=data)
    kb.file_io.write_json(filename=job_dir / obj_file, data=object_metadata)
    kb.done()
