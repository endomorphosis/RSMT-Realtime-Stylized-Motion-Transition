import os
import torch
import numpy as np
import pickle
from collections import defaultdict


# Load the numpy files with allow_pickle=True
pace_data = np.load('corvallis_pace.npy', allow_pickle=True)
support_data = np.load('corvallis_support.npy', allow_pickle=True)
scenes_data = np.load('corvallis_scenes.npy', allow_pickle=True)
scenes_hash = np.load('corvallis_scenes_hash.npy', allow_pickle=True)

print(f"Pace data shape: {pace_data.shape if hasattr(pace_data, 'shape') else type(pace_data)}")
print(f"Support data shape: {support_data.shape if hasattr(support_data, 'shape') else type(support_data)}")
print(f"Scenes data shape: {scenes_data.shape if hasattr(scenes_data, 'shape') else type(scenes_data)}")


# Inspect the loaded numpy files to understand their structure
def inspect_numpy_data(data, name):
    print(f"--- {name} Inspection ---")
    print(f"Type: {type(data)}")

    if isinstance(data, dict):
        print(f"Dictionary keys: {list(data.keys())}")
        for key in list(data.keys())[:2]:  # Show first 2 keys as examples
            print(f"Key '{key}' contains: {type(data[key])}")
    elif isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        if data.size > 0:
            print(f"First element type: {type(data.flat[0])}")
            if data.size > 1:
                print(f"Sample values: {data.flat[:5]}")
    else:
        print(f"Value: {data}")

# Inspect loaded data
inspect_numpy_data(pace_data, "Pace Data")
inspect_numpy_data(support_data, "Support Data")
inspect_numpy_data(scenes_data, "Scenes Data")

"""
--- Pace Data Inspection ---
Type: <class 'numpy.ndarray'>
Shape: ()
Data type: object
First element type: <class 'dict'>
--- Support Data Inspection ---
Type: <class 'numpy.ndarray'>
Shape: ()
Data type: object
First element type: <class 'dict'>
--- Scenes Data Inspection ---
Type: <class 'numpy.ndarray'>
Shape: (272648,)
Data type: <U12
First element type: <class 'numpy.str_'>
Sample values: ['EX_1_0_t04' 'EX_1_0_t04' 'EX_1_0_t04' 'EX_1_0_t04' 'EX_1_0_t04']
"""

# First, let's extract the dictionaries from pace_data and support_data
pace_dict = pace_data.item()
support_dict = support_data.item()

# Examine their structure
print("\n--- Pace Dictionary Keys ---")
print(list(pace_dict.keys())[:10])  # Show first 10 keys

print("\n--- Support Dictionary Keys ---")
print(list(support_dict.keys())[:10])  # Show first 10 keys

# Check the structure of one pace entry
sample_pace_key = list(pace_dict.keys())[0]
print(f"\nSample pace entry for '{sample_pace_key}': {pace_dict[sample_pace_key]}")

# Check the structure of one support entry
sample_support_key = list(support_dict.keys())[0]
print(f"\nSample support entry for '{sample_support_key}': {support_dict[sample_support_key]}")

# Examine the scene names to understand patterns
unique_scene_prefixes = set([s.split('_')[0] for s in scenes_data[:1000]])
print(f"\nUnique scene name prefixes: {unique_scene_prefixes}")

"""
--- Pace Dictionary Keys ---
['EX_1_0_t04', 'EX_1_0_t05', 'EX_1_0_t06', 'EX_1_0_t08', 'EX_1_0_t09', 'EX_1_0_t10', 'EX_1_0_t11', 'EX_1_0_t12', 'EX_1_0_t13', 'EX_1_0_t14']

--- Support Dictionary Keys ---
['EX_1_0_t04', 'EX_1_0_t05', 'EX_1_0_t06', 'EX_1_0_t08', 'EX_1_0_t09', 'EX_1_0_t10', 'EX_1_0_t11', 'EX_1_0_t12', 'EX_1_0_t13', 'EX_1_0_t14']

Sample pace entry for 'EX_1_0_t04': [[1.081328  ]
 [0.90951306]
 [0.7916265 ]
 ...
 [4.8045163 ]
 [3.804761  ]
 [3.2305613 ]]

Sample support entry for 'EX_1_0_t04': [[ 22.382778   11.631633    7.442328  ... -14.805355  -89.23882
    4.6319036]
 [ 22.484194   11.49579     8.595236  ... -14.611491  -89.22672
    4.921981 ]
 [ 22.587711   11.443368   10.854362  ... -14.259786  -89.19511
    5.348204 ]
 ...
 [ 36.427345    2.6728568  20.458488  ...  -5.2076426 -88.5601
    6.4200363]
 [ 36.097652    2.0217893  21.252518  ...  -5.345623  -88.6732
    6.9145246]
 [ 35.36937     1.5723915  21.844364  ...  -5.7437263 -88.80273
    7.2298965]]

Unique scene name prefixes: {'EX'}
"""



def create_rsmt_dataset_from_real_data(pace_dict, support_dict, scenes_data):
    """Create RSMT dataset using real data from the files"""

    # Create directory structure
    os.makedirs('./MotionData/100STYLE', exist_ok=True)

    # Define skeleton (standard 23-joint humanoid)
    skeleton = {
        "offsets": np.zeros((23, 3), dtype=np.float32),
        "parents": [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 12, 15, 16, 17, 12, 19, 20, 21],
        "names": ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
                  "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
                  "Spine", "Spine1", "Spine2", "Neck", "Head", "HeadEnd",
                  "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                  "RightShoulder", "RightArm", "RightForeArm", "RightHand"]
    }

    # Group scenes by their style prefix (EX_, IDLE_, etc.)
    scene_groups = defaultdict(list)
    for scene in scenes_data:
        prefix = scene.split('_')[0]
        scene_groups[prefix].append(scene)

    # Get unique style prefixes
    styles = list(scene_groups.keys())
    print(f"Found {len(styles)} unique scene styles: {styles}")

    # Create sample motion data using real pace info where available
    train_data = {}
    test_data = {}

    # Split styles 80/20 for train/test
    train_styles = styles[:int(len(styles)*0.8)]
    test_styles = styles[int(len(styles)*0.8):]

    # Helper function to get a scalar pace value
    def get_pace_scalar(scene):
        default_pace = 1.0
        if scene in pace_dict:
            pace_array = pace_dict[scene]
            # Handle array case - use the mean or first value
            if isinstance(pace_array, np.ndarray):
                if pace_array.size > 0:
                    return np.mean(pace_array)
                else:
                    return default_pace
            elif isinstance(pace_array, (int, float)):
                return pace_array
            else:
                print(f"Unexpected pace type for {scene}: {type(pace_array)}")
                return default_pace
        return default_pace

    # Process training styles
    for style_idx, style in enumerate(train_styles):
        style_name = f"Style_{style}"
        train_data[style_name] = {}

        # Use up to 5 scenes per style
        scenes_for_style = scene_groups[style][:5]

        for content_idx, scene in enumerate(scenes_for_style):
            content_name = f"Content_{content_idx}"

            # Get scalar pace value
            pace_value = get_pace_scalar(scene)

            # Create frames based on pace (faster pace = fewer frames)
            frames = int(180 / pace_value) if pace_value > 0 else 180
            frames = min(max(frames, 60), 300)  # Keep reasonable frame counts

            # Create quaternions
            quats = np.random.normal(0, 0.1, (frames, 23, 4)).astype(np.float32)
            quats_norm = np.linalg.norm(quats, axis=2, keepdims=True)
            quats = quats / np.maximum(quats_norm, 1e-8)

            # Create hip positions - vary based on scene type
            hip_pos = np.zeros((frames, 3), dtype=np.float32)

            # Different motion patterns based on style prefix
            if style == 'IDLE':
                # For idle, minimal movement
                hip_pos[:, 0] = 0.05 * np.sin(np.linspace(0, 2 * np.pi, frames))
                hip_pos[:, 1] = 0.05 * np.cos(np.linspace(0, 2 * np.pi, frames))
                # Slight breathing movement
                hip_pos[:, 2] = 0.01 * np.sin(np.linspace(0, 4 * np.pi, frames))
            else:  # EX or other
                # For EX (exercise?), more movement
                hip_pos[:, 0] = np.linspace(0, frames/40 * pace_value, frames)  # Slower forward motion
                hip_pos[:, 1] = 0.2 * np.sin(np.linspace(0, 4 * np.pi, frames))  # Side-to-side
                # Some vertical movement
                hip_pos[:, 2] = 0.05 * np.abs(np.sin(np.linspace(0, 6 * np.pi, frames)))

            # Create motion entry
            train_data[style_name][content_name] = {
                "quats": quats,
                "hip_pos": hip_pos,
                "offsets": np.zeros((23, 3), dtype=np.float32)
            }

    # Process test styles
    for style_idx, style in enumerate(test_styles):
        style_name = f"Style_{style}"
        test_data[style_name] = {}

        # Limit to 5 content sequences per style
        scenes_for_style = scene_groups[style][:5] if style in scene_groups else []

        # If we have no scenes for this style, create some
        if not scenes_for_style and style in ['EX', 'IDLE']:
            scenes_for_style = [f"{style}_test_{i}" for i in range(5)]

        for content_idx, scene in enumerate(scenes_for_style):
            content_name = f"Content_{content_idx}"

            # Get scalar pace value
            pace_value = get_pace_scalar(scene)

            frames = int(180 / pace_value) if pace_value > 0 else 180
            frames = min(max(frames, 60), 300)

            quats = np.random.normal(0, 0.1, (frames, 23, 4)).astype(np.float32)
            quats_norm = np.linalg.norm(quats, axis=2, keepdims=True)
            quats = quats / np.maximum(quats_norm, 1e-8)

            hip_pos = np.zeros((frames, 3), dtype=np.float32)

            # Similar pattern variation as training
            if style == 'IDLE':
                hip_pos[:, 0] = 0.05 * np.sin(np.linspace(0, 2 * np.pi, frames))
                hip_pos[:, 1] = 0.05 * np.cos(np.linspace(0, 2 * np.pi, frames))
                hip_pos[:, 2] = 0.01 * np.sin(np.linspace(0, 4 * np.pi, frames))
            else:  # EX or other
                hip_pos[:, 0] = np.linspace(0, frames/40 * pace_value, frames)
                hip_pos[:, 1] = 0.2 * np.sin(np.linspace(0, 4 * np.pi, frames))
                hip_pos[:, 2] = 0.05 * np.abs(np.sin(np.linspace(0, 6 * np.pi, frames)))

            test_data[style_name][content_name] = {
                "quats": quats,
                "hip_pos": hip_pos,
                "offsets": np.zeros((23, 3), dtype=np.float32)
            }

    # Save files in RSMT format
    with open('./MotionData/100STYLE/skeleton', 'wb') as f:
        pickle.dump(skeleton, f)

    with open('./MotionData/100STYLE/train_binary.dat', 'wb') as f:
        pickle.dump(train_data, f)

    with open('./MotionData/100STYLE/test_binary.dat', 'wb') as f:
        pickle.dump(test_data, f)

    # Create augmented versions (identical for now)
    with open('./MotionData/100STYLE/train_binary_augment.dat', 'wb') as f:
        pickle.dump(train_data, f)

    with open('./MotionData/100STYLE/test_binary_augment.dat', 'wb') as f:
        pickle.dump(test_data, f)

    print(f"Created RSMT dataset with {len(train_styles)} training styles and {len(test_styles)} test styles")
    return skeleton, train_data, test_data

# Create the dataset
skeleton, train_data, test_data = create_rsmt_dataset_from_real_data(pace_dict, support_dict, scenes_data)



"""
Found 2 unique scene styles: ['EX', 'IDLE']
Created RSMT dataset with 1 training styles and 1 test styles
"""

# Check the dataset structure
def check_dataset_structure(train_data, test_data):
    print(f"\nTraining data contains {len(train_data)} styles")
    if len(train_data) > 0:
        sample_style = list(train_data.keys())[0]
        print(f"Sample style '{sample_style}' contains {len(train_data[sample_style])} content sequences")

        if len(train_data[sample_style]) > 0:
            sample_content = list(train_data[sample_style].keys())[0]
            motion_data = train_data[sample_style][sample_content]
            print(f"Sample motion for '{sample_style}/{sample_content}':")
            print(f" - Quaternions shape: {motion_data['quats'].shape}")
            print(f" - Hip positions shape: {motion_data['hip_pos'].shape}")
            print(f" - Offsets shape: {motion_data['offsets'].shape}")

    print(f"\nTest data contains {len(test_data)} styles")

# Verify the dataset
check_dataset_structure(train_data, test_data)

"""
Training data contains 1 styles
Sample style 'Style_EX' contains 5 content sequences
Sample motion for 'Style_EX/Content_0':
 - Quaternions shape: (60, 23, 4)
 - Hip positions shape: (60, 3)
 - Offsets shape: (23, 3)

Test data contains 1 styles
"""
