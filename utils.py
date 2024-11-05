import numpy as np
import open3d as o3d

import os
from os import path as osp

import pickle
import glob
import matplotlib.pyplot as plt

from geometry import make_tf, apply_tf

CLASS_NAMES = ['car','truck','motorcycle', 'pedestrian']
CLASS_COLORS = plt.cm.rainbow(np.linspace(0, 1, len(CLASS_NAMES)))[:, :3]
CLASS_NAME_TO_COLOR = dict(zip(CLASS_NAMES, CLASS_COLORS))
CLASS_NAME_TO_INDEX = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))

# Path extraction
root_path = "/home/l1/dev/ECN_AUVE_labs/scenario1"

scenario = "Town01_type001_subtype0001_scenario00003"  


file_list = glob.glob(osp.join(root_path,
                                        'ego_vehicle', 'label', scenario) + '/*')
frame_list = []

with open(osp.join(root_path, "meta" ,scenario+ '.txt'), 'r') as f:
            lines = f.readlines()
line = lines[2]
agents =  [int(agent) for agent in line.split()[2:]]

for file_path in file_list:
    frame_list.append(file_path.split('/')[-1].split('.')[0])
frame_list.sort()
nb_frames = len(frame_list)

def get_tf_lidar2world(actor, n_frame):

    frame = frame_list[n_frame]
    with open(osp.join(root_path, actor ,'calib',scenario, frame + '.pkl'), 'rb') as f:
        calib_dict = pickle.load(f)
    actor_tf_world = np.array(calib_dict['ego_to_world'])
    lidar_tf_actor = np.array(calib_dict['lidar_to_ego'])
    
    return lidar_tf_actor @ actor_tf_world

def get_sensor_T_actor(actor, n_frame):
    frame = frame_list[n_frame]
    with open(osp.join(root_path, actor ,'calib',scenario, frame + '.pkl'), 'rb') as f:
        calib_dict = pickle.load(f)
    lidar_tf_actor = np.array(calib_dict['lidar_to_ego'])

    tf =  lidar_tf_actor 
    trans = tf[:3,3]
    rot = tf[:3,:3]

    return make_tf(trans,rot) 

def get_point_cloud(n_frame, actor):

    frame = frame_list[n_frame] 
    lidar_data = np.load(root_path +  '/' + actor + '/lidar01/' + scenario +'/' + frame + '.npz')['data']
    lidar_T_actor = get_sensor_T_actor(actor, n_frame)
    lidar_data_actor = apply_tf(lidar_T_actor, lidar_data) #in actor frame
 
    return lidar_data_actor
