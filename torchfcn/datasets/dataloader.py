#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:58:17 2017

Preprocess a folder with up to 1 hour of radar, camera, Seapath and AIS data. Add a JSON metadata file per radar/camera image. 
Add a PNG image file with target segmentation per radar/camera image. Added files will have same base name as corresponding 
image file. Existing JSON and PNG files will be replaced.

@author: AN
"""

import os
import json
import numpy as np
import math
from skimage.draw import polygon, line
from io import BytesIO
import requests
import copy
import traceback
from PIL import Image

opencv = False
try: 
    import cv2
    opencv = True
except:
    from PIL import Image

# format string for the folder name (date + hour)
FOLDER_NAME_FORMAT = os.path.join("{:%Y-%m-%d", "%Y-%m-%d-%H}")

# format string for the file name (date + time)
FILE_NAME_FORMAT_SEC = "{:%Y-%m-%d-%H_%M_%S}"
FILE_NAME_FORMAT_MIN = "{:%Y-%m-%d-%H_%M}"

RADAR_RANGE_FILTERS_DEFAULT = [[1, 2000], [2, 1000], [4, 1000], [6, 1000], [8, 1000], [8, 216], [0, 262143], [0, 262143]]
RADAR_M_PER_SAMPLE_DEFAULT = 3.4067

EARTH_RADIUS = 6367000
                     
class data_loader():
    """
    Class for reading images and labels from Seatex Polarlys dataset.
    """
    def __init__(self, path, sensor_config='dataloader.json'):
        self.size = 0
        self.path = path
        self.sensor_config = None
        self.TYPE_CAMERA = 0
        self.TYPE_RADAR = 1

        self.CHART_POLAR = 0
        self.CHART_CART = 1
        
        try:
            with open(sensor_config, 'r') as f:
                self.sensor_config = json.load(f)
        except:
            self.sensor_config = None
            print('data_loader: Unable to read configuration file {}'.format(sensor_config))
        
        
    def change_path(self, path):
        self.path = path
        
        
    def get_filename_sec(self, t, sensor_path, extension):
        """
        Function for getting a file name for a specific second
        """
        folder = FOLDER_NAME_FORMAT.format(t)
        filename_sec = FILE_NAME_FORMAT_SEC.format(t)
        
        path_sec = os.path.join(self.path, folder)
        path_sec = os.path.join(path_sec,sensor_path)
        path_sec = os.path.join(path_sec,filename_sec)
        
        return path_sec + '.' + extension
    
    
    def get_sensor_from_path(self, sensor_path):
        sensor_type, sensor_index, subsensor_index = None, None, None
        if sensor_path.find('Cam') >= 0:
            sensor_type = self.TYPE_CAMERA
            sensor_index = int(sensor_path[sensor_path.find('Cam') + 3])
            subsensor_index = int(sensor_path[sensor_path.find('Lens') + 4])
        elif sensor_path.find('Radar') >= 0:
            sensor_type = self.TYPE_RADAR
            sensor_index = int(sensor_path[sensor_path.find('Radar') + 5])
            
        return sensor_type, sensor_index, subsensor_index
        
        
    def get_sensor_folder(self, sensor_type, sensor_index, subsensor_index=None):
        folder = None
        if sensor_type == self.TYPE_CAMERA:
            folder = os.path.join('Cam{}'.format(sensor_index), 'Lens{}'.format(subsensor_index))
        elif sensor_type == self.TYPE_RADAR:
            folder = 'Radar{}'.format(sensor_index)
            
        return folder
    
    
    def get_sensor_config(self, sensor_type, sensor_index, subsensor_index=None):
        with open('dataloader.json', 'r') as f:
            self.sensor_config = json.load(f)
        cfg = {}
        if sensor_type == self.TYPE_CAMERA:
            sensor_str = 'Cam{}'.format(sensor_index)
            subsensor_str = 'Lens{}'.format(subsensor_index)
            cfg = self.sensor_config[sensor_str][subsensor_str]
        elif sensor_type == self.TYPE_RADAR:
            sensor_str = 'Radar{}'.format(sensor_index)
            cfg = self.sensor_config[sensor_str]
            
        return cfg
    
    
    def get_image_extension(self, sensor_type):
        ext = None
        if sensor_type == self.TYPE_CAMERA:
            ext = 'jpg'
        elif sensor_type == self.TYPE_RADAR:
            ext = 'bmp'
            
        return ext
    
 
    def get_basename(self, t, sensor_type, sensor_index, subsensor_index=None):
        folder = FOLDER_NAME_FORMAT.format(t)
        filename_sec = FILE_NAME_FORMAT_SEC.format(t)
        
        path_sensor = os.path.join(folder, self.get_sensor_folder(sensor_type, sensor_index, subsensor_index))
        path_sec = os.path.join(path_sensor, filename_sec)
        
        return os.path.join(self.path, path_sec)
    
    
    def load_image(self, file_name):
        if not file_name or not os.path.isfile(file_name):
            return []
        
        image = []
        try:
            if opencv:
                image = cv2.imread(file_name, -1)
                if image.ndim == 3:
                    image = image[:,:,::-1]
                #b,g,r = cv2.split(image)
                #image = cv2.merge((r,g,b))
            else:
                with Image.open(file_name) as pil:
                    image = np.array(pil, dtype=np.uint8)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return image
    
    
    def load_image_by_basename(self, file_basename):
        if file_basename.find('Cam') >= 0:
            extension = self.get_image_extension(self.TYPE_CAMERA)
        else:
            extension = self.get_image_extension(self.TYPE_RADAR)   
             
        return self.load_image(os.path.join(self.path, file_basename) + '.' + extension)
    
    
    def load_image_by_time(self, t, sensor_type, sensor_index, subsensor_index=None):
        file_name = self.get_filename_sec(t, self.get_sensor_folder(sensor_type, sensor_index, subsensor_index), self.get_image_extension(sensor_type))
        
        return self.load_image(file_name)
       
  
    def load_chart_layer(self, file_meta, chart_transform, sensor_type, sensor_index, subsensor_index=None):
        if not file_meta or not os.path.isfile(file_meta):
            return []
        
        try:
            with open(file_meta, 'r') as f:
                meta = json.load(f)
        except:
            print('data_loader: Unable to read meta data file {}'.format(file_meta))
            return []
        
        try:
            meta['config'] = self.get_sensor_config(sensor_type, sensor_index, subsensor_index)
        except:
            print('data_loader: Required sensor configuration is missing')
            return []
        
        layer = []
        try:
            if sensor_type == self.TYPE_RADAR and chart_transform == self.CHART_POLAR:
                if not meta['own_vessel_start'] == {} and not meta['own_vessel_end'] == {}:
                    pva = meta['own_vessel_end']
                    pva0 = meta['own_vessel_start']
                    pos = np.array(((pva0['position'][0] + pva['position'][0]) / 2, (pva0['position'][1] + pva['position'][1]) / 2, (pva0['position'][2] + pva['position'][2]) / 2))
                    att = np.array(((pva0['attitude'][0] + pva['attitude'][0]) / 2, (pva0['attitude'][1] + pva['attitude'][1]) / 2, (pva0['attitude'][2] + pva['attitude'][2]) / 2))
                    pos_sensor = self.translate_pos(pos, att, np.array(meta['config']['location']))
                    layer = self.get_polar_chart(pos_sensor, att, meta['radar_setup']['range_filters'], meta['config']['m_per_sample'], meta['image_dim'])
            else:
                print('data_loader: Chart transform {} for sensor type {} is not implemented'.format(chart_transform, sensor_type))
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return layer
   
 
    def load_chart_layer_by_basename(self, file_basename, chart_transform):
        """
        file_basename must contain full path, eg. '2017-10-13\2017-10-13-15\Cam0\Lens1\2017-10-13-15_08_50_470000'
        """
        sensor_type, sensor_index, subsensor_index = self.get_sensor_from_path(file_basename) 
        layer = self.load_chart_layer(os.path.join(self.path, file_basename) + '.json', chart_transform, sensor_type, sensor_index, subsensor_index)
        
        return layer
    
    
    def load_chart_layer_by_time(self, t, chart_transform, sensor_type, sensor_index, subsensor_index=None):
        file_meta = self.get_filename_sec(t, self.get_sensor_folder(sensor_type, sensor_index, subsensor_index), 'json')  
        if file_meta == None:
            return []                
        layer = self.load_chart_layer(file_meta, chart_transform, sensor_type, sensor_index, subsensor_index)
        
        return layer
    
    
    def load_ais_layer(self, file_meta, sensor_type, sensor_index, subsensor_index=None):
        if not file_meta or not os.path.isfile(file_meta):
            return []
        
        try:
            with open(file_meta, 'r') as f:
                meta = json.load(f)
        except:
            print('data_loader: Unable to read meta data file {}'.format(file_meta))
            return []
      
        try:
            meta['config'] = self.get_sensor_config(sensor_type, sensor_index, subsensor_index)
            meta['Seapath'] = self.sensor_config['Seapath']
            meta['AIS'] = self.sensor_config['AIS']
        except:
            print('data_loader: Required sensor configuration is missing')
            return []
        
        layer = []
        try:
            if sensor_type == self.TYPE_CAMERA:
                targets = meta['targets_ais']
                ais_targets = {} 
                for k,v in targets.items():
                    ais_targets[k] = targets[k].copy()
                    self.transform_ais_target(ais_targets[k], meta['own_vessel'], meta['config'], meta['Seapath']['navref_height'], meta['AIS']) 
                layer = self.get_polygon_layer(ais_targets, meta['config'], meta['image_dim'])
            elif sensor_type == self.TYPE_RADAR:
                targets_start = meta['targets_ais_start']
                targets_end = meta['targets_ais_end']
                ais_targets_start = {}
                for k,v in targets_start.items():
                    ais_targets_start[k] = targets_start[k].copy()
                    self.transform_ais_target(ais_targets_start[k], meta['own_vessel_start'], meta['config'], meta['Seapath']['navref_height'], meta['AIS']) 
                ais_targets_end = {}
                for k,v in targets_end.items():
                    ais_targets_end[k] = targets_end[k].copy()
                    self.transform_ais_target(ais_targets_end[k], meta['own_vessel_end'], meta['config'], meta['Seapath']['navref_height'], meta['AIS'])
                ais_targets = self.radar_interpolate_targets(ais_targets_start, ais_targets_end)
                layer = self.get_polygon_layer(ais_targets, meta['config'], meta['image_dim'], meta['radar_setup']['range_filters'])
            else:
                print('data_loader: Unknown sensor type {}'.format(sensor_type))
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return layer
    
    
    def load_ais_layer_by_basename(self, file_basename):
        """
        file_basename must contain full path, eg. '2017-10-13\2017-10-13-15\Cam0\Lens1\2017-10-13-15_08_50_470000'
        """
        sensor_type, sensor_index, subsensor_index = self.get_sensor_from_path(file_basename)          
        return self.load_ais_layer(os.path.join(self.path, file_basename) + '.json', sensor_type, sensor_index, subsensor_index)
    
    
    def load_ais_layer_by_time(self, t, sensor_type, sensor_index, subsensor_index=None):
        file_meta = self.get_filename_sec(t, self.get_sensor_folder(sensor_type, sensor_index, subsensor_index), 'json')
        if file_meta == None:
            return []               
        return self.load_ais_layer(file_meta, sensor_type, sensor_index, subsensor_index)
    
    
    def load_arpa_layer(self, file_meta, sensor_type, sensor_index, subsensor_index=None):
        if not file_meta or not os.path.isfile(file_meta):
            return []
        
        try:
            with open(file_meta, 'r') as f:
                meta = json.load(f)
        except:
            print('data_loader: Unable to read meta data file {}'.format(file_meta))
            return []
      
        try:
            meta['config'] = self.get_sensor_config(sensor_type, sensor_index, subsensor_index)
            meta['Seapath'] = self.sensor_config['Seapath']
            meta['ARPA'] = self.sensor_config['ARPA']
        except:
            print('data_loader: Required sensor configuration is missing')
            return []
        
        layer = []
        try:
            if sensor_type == self.TYPE_CAMERA:
                targets = meta['targets_arpa']
                arpa_targets = {} 
                for k,v in targets.items():
                    arpa_targets[k] = targets[k].copy()
                    self.transform_arpa_target(arpa_targets[k], meta['own_vessel'], meta['config'], meta['ARPA'], meta['Seapath']['navref_height']) 
                layer = self.get_polygon_layer(arpa_targets, meta['config'], meta['image_dim'])
            elif sensor_type == self.TYPE_RADAR:
                targets_start = meta['targets_arpa_start']
                targets_end = meta['targets_arpa_end']
                arpa_targets_start = {}
                for k,v in targets_start.items():
                    arpa_targets_start[k] = targets_start[k].copy()
                    self.transform_arpa_target(arpa_targets_start[k], meta['own_vessel_start'], meta['config'], meta['ARPA'], meta['Seapath']['navref_height']) 
                arpa_targets_end = {}
                for k,v in targets_end.items():
                    arpa_targets_end[k] = targets_end[k].copy()
                    self.transform_arpa_target(arpa_targets_end[k], meta['own_vessel_end'], meta['config'], meta['ARPA'], meta['Seapath']['navref_height'])
                arpa_targets = self.radar_interpolate_targets(arpa_targets_start, arpa_targets_end)
                layer = self.get_polygon_layer(arpa_targets, meta['config'], meta['image_dim'], meta['radar_setup']['range_filters'])
            else:
                print('data_loader: Unknown sensor type {}'.format(sensor_type))
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return layer
    
  
    def load_arpa_layer_by_basename(self, file_basename):
        """
        file_basename must contain full path, eg. '2017-10-13\2017-10-13-15\Cam0\Lens1\2017-10-13-15_08_50_470000'
        """
        sensor_type, sensor_index, subsensor_index = self.get_sensor_from_path(file_basename)          
        return self.load_arpa_layer(os.path.join(self.path, file_basename) + '.json', sensor_type, sensor_index, subsensor_index)
    
    
    def load_arpa_layer_by_time(self, t, sensor_type, sensor_index, subsensor_index=None):
        file_meta = self.get_filename_sec(t, self.get_sensor_folder(sensor_type, sensor_index, subsensor_index), 'json')
        if file_meta == None:
            return []                
        return self.load_arpa_layer(file_meta, sensor_type, sensor_index, subsensor_index)
    
    
    def load_horizon_layer(self, file_meta, sensor_type, sensor_index, subsensor_index=None):
        if not file_meta or not os.path.isfile(file_meta):
            return []
        
        try:
            with open(file_meta, 'r') as f:
                meta = json.load(f)
        except:
            print('data_loader: Unable to read meta data file {}'.format(file_meta))
            return []
      
        try:
            meta['config'] = self.get_sensor_config(sensor_type, sensor_index, subsensor_index)
            meta['Seapath'] = self.sensor_config['Seapath']
        except:
            print('data_loader: Required sensor configuration is missing')
            return []
        
        layer = []
        try:
            if sensor_type == self.TYPE_CAMERA:
                layer = self.get_horizon_layer(meta['own_vessel'], meta['config'], meta['Seapath']['navref_height'], meta['image_dim'])
            else:
                print('data_loader: Horizon layer for sensor type {} is not supported'.format(sensor_type))
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return layer
    

    def load_horizon_layer_by_basename(self, file_basename):
        """
        file_basename must contain full path, eg. '2017-10-13\2017-10-13-15\Cam0\Lens1\2017-10-13-15_08_50_470000'
        """
        sensor_type, sensor_index, subsensor_index = self.get_sensor_from_path(file_basename)          
        return self.load_horizon_layer(os.path.join(self.path, file_basename) + '.json', sensor_type, sensor_index, subsensor_index)
    
    
    def load_horizon_layer_by_time(self, t, sensor_type, sensor_index, subsensor_index=None):
        file_meta = self.get_filename_sec(t, self.get_sensor_folder(sensor_type, sensor_index, subsensor_index), 'json')
        if file_meta == None:
            return []               
        return self.load_horizon_layer(file_meta, sensor_type, sensor_index, subsensor_index)
    
    
    def transform_points(self, points, att, from_sensor_type, to_sensor_type, from_sensor_index, to_sensor_index, from_subsensor_index=None, to_subsensor_index=None):
        """
        Transform points in pixel coordinates (X,Y) from one sensor frame to another. Assumes that points are in the sea plane.
        The argument 'points' should be a numpy array of shape (n,2), where n is the number of points.
        The argument 'from_subsensor_index' is only required if 'from_sensor_type' is TYPE_CAMERA.
        The argument 'to_subsensor_index' is only required if 'to_sensor_type' is TYPE_CAMERA.
        The methods returns a numpy array of shape (n,2), where n is the number of input points.
        If a point falls outside the image boundaries, the coordinates for this points are set to -1.
        """
        if from_sensor_type == self.TYPE_CAMERA:
            points = self.transform_points_from_camera(points, att, from_sensor_index, from_subsensor_index)
        elif from_sensor_type == self.TYPE_RADAR:
            points = self.transform_points_from_radar(points, att, from_sensor_index)
        
        if to_sensor_type == self.TYPE_CAMERA:
            points = self.transform_points_to_camera(points, att, to_sensor_index, to_subsensor_index)
        elif to_sensor_type == self.TYPE_RADAR:
            points = self.transform_points_to_radar(points, att, to_sensor_index)
            
        return points
        
    
    def transform_points_from_camera(self, points, att, from_sensor_index, from_subsensor_index, ref_location=None, dim_x=1920, dim_y=2560):
        """
        Transform numpy array of camera image pixel coordinates (X,Y) to points in meters in (N,E,D) geographic frame. Assumes that points are in the sea plane.
        The argument 'points' should be a numpy array of shape (n,2), where n is the number of points.
        The argument 'ref_location' is the offset from the Seapath reference point to the reference 
        point to be used for the returned points in meters and vessel body coordinates (fwd,stb,dwn). 
        If 'ref_location' is None, returned points are relative to Seapath reference point.
        If argument 'dim_x' is None, default radar image height 1920 is used.
        If argument 'dim_y' is None, default radar image width 2560 is used.
        The methods returns a numpy array of shape (n,3), where n is the number of input points.
        If a point is pointing above the horizon, the coordinates for this points are set to (0,0,0).
        """
        cfg_sensor = self.get_sensor_config(self.TYPE_CAMERA, from_sensor_index, from_subsensor_index)
        
        # Get camera calibration
        camera_matrix = np.array(cfg_sensor['camera_matrix']).reshape((3,3))
        dist_coeffs = np.array(cfg_sensor['distortion_coefficients'])
        
        # Get rotation matrices
        R_ma = self.rot_matrix_from_euler(cfg_sensor['rotation'])
        R_att = self.rot_matrix_from_euler(att)       
        #print(cfg_sensor['rotation'])
        #R_ma = self.rot_matrix_from_euler((0,-17.95,0))
        # Get translation vector
        T = np.array(cfg_sensor['location']).reshape(3,1)
        if ref_location is not None:
            T -= np.array(ref_location).reshape(3,1)
         
        # Get reference point height above sea level given attitude
        hgt = self.sensor_config['Seapath']['navref_height']
        if ref_location is not None:
            hgt -= ref_location[2]
        hgt_att = R_att.transpose().dot(np.array((0, 0, hgt)).reshape((3,1)))[2]
        
        p_transformed = np.zeros((points.shape[0], 3)).astype(np.float64)
        for i in range(points.shape[0]):
            # Convert pixel coordinates to normalized coordinates by applying camera calibration
            yn, zn = self.camera_p2m(points[i,1], points[i,0], camera_matrix, dist_coeffs)   
            
            # Check diff in z for x==1m
            pn = np.array((1, yn, zn)).reshape(3,1)
            pn_geog = R_att.dot(R_ma.dot(pn))
            if(pn_geog[2] < 0):
                # Not at sea level
                p_transformed[i,:] = 0
            else:
                # Calculate distance x to sea sea plane
                x = (hgt_att - R_att.dot(T)[2]) / pn_geog[2]
                # Convert normalized coordinates to cartesian
                p = np.array((x, x*yn, x*zn)).reshape(3,1)
                # Rotate and translate to reference point
                p_geog = R_att.dot(R_ma.dot(p) + T)
                p_transformed[i,:] = p_geog.reshape(1,3)
   
        return p_transformed
    
    
    def transform_points_from_radar(self, points, att, from_sensor_index, range_filters=None, ref_location=None, dim_x=4096):
        """
        Transform numpy array of radar image pixel coordinates (X,Y) to points in meters in (N,E,D) geographic frame. Assumes that points are in the sea plane.
        The argument 'points' should be a numpy array of shape (n,2), where n is the number of points.
        If the argument 'range_filters' is None, default radar range filters are used.
        The argument 'ref_location' is the offset from the Seapath reference point to the reference 
        point to be used for the returned points in meters and vessel body coordinates (fwd,stb,dwn). 
        If 'ref_location' is None, returned points are relative to Seapath reference point.
        If argument 'dim_x' is None, default radar image height 4096 is used.
        The methods returns a numpy array of shape (n,3), where n is the number of input points.
        """
        cfg_sensor = self.get_sensor_config(self.TYPE_RADAR, from_sensor_index)
        
        if range_filters is None:
            range_filters = RADAR_RANGE_FILTERS_DEFAULT
        
        # Get rotation matrices
        R_ma = self.rot_matrix_from_euler(cfg_sensor['rotation'])
        R_att = self.rot_matrix_from_euler(att)       
        
        # Get translation vector
        T = np.array(cfg_sensor['location']).reshape(3,1)
        if ref_location is not None:
            T -= np.array(ref_location).reshape(3,1)
         
        # Get reference point height above sea level given attitude
        hgt = self.sensor_config['Seapath']['navref_height']
        if ref_location is not None:
            hgt -= ref_location[2]
        hgt_att = R_att.transpose().dot(np.array((0, 0, hgt)).reshape((3,1)))[2] 
        
        p_transformed = np.zeros((points.shape[0], 3)).astype(np.float64)
        for i in range(points.shape[0]):
            rng = self.radar_p2m([points[i,1]], range_filters, cfg_sensor['m_per_sample'])
            brg = 360 * (1 - points[i,0] / dim_x)
            
            # Iterate to get correct z in sensor frame
            hgt_error = 1000
            z = -self.sensor_config['Seapath']['navref_height'] - cfg_sensor['location'][2]
            while abs(hgt_error) > 0.1:
                dst = math.sqrt(rng**2 - z**2)
                x = dst * math.cos(brg * math.pi / 180)
                y = dst * math.sin(brg * math.pi / 180)
                p = np.array((x,y,z)).reshape(3,1)
                p_geog = R_att.dot(R_ma.dot(p) + T)
                hgt_error = hgt_att + p_geog[2]
                z -= hgt_error
            
            p_geog[2] = -hgt_att
            p_transformed[i,:] = p_geog.reshape(1,3)
            
        return p_transformed
    
    
    def transform_points_to_camera(self, points, att, to_sensor_index, to_subsensor_index, ref_location=None, dim_x=1920, dim_y=2560):
        """
        Transform points in meters in (N,E,D) geographic frame to pixels coordinates (X,Y) in camera image.
        The argument 'points' should be a numpy array of shape (n,3), where n is the number of points.
        The argument 'ref_location' is the offset from the Seapath reference point to the reference 
        point used for the input points in meters and vessel body coordinates (fwd,stb,dwn). If None, the offset is 
        assumed to be (0,0,0).
        If argument 'dim_x' is None, default radar image height 1920 is used.
        If argument 'dim_y' is None, default radar image width 2560 is used.
        The methods returns a numpy array of shape (n,2), where n is the number of input points.
        If a point falls outside the image boundaries, the coordinates for this points are set to -1.
        """       
        cfg_sensor = self.get_sensor_config(self.TYPE_CAMERA, to_sensor_index, to_subsensor_index)  
        
        # Get rotation matrices
        R_ma = self.rot_matrix_from_euler(cfg_sensor['rotation']).transpose()
        R_att = self.rot_matrix_from_euler(att).transpose()
        
        # Get translation vector
        T = np.array(cfg_sensor['location']).reshape(3,1)
        if ref_location is not None:
            T -= np.array(ref_location).reshape(3,1)
        
        # Get camera calibration
        camera_matrix = np.array(cfg_sensor['camera_matrix']).reshape((3,3))
        dist_coeffs = np.array(cfg_sensor['distortion_coefficients'])
                    
        p_transformed = np.zeros((points.shape[0], 2)).astype(np.int16)
        for i in range(points.shape[0]):
            p = points[i,:].reshape((3,1))
            
            # Translate and rotate target pos vector to sensor frame
            p_vessel = R_att.dot(p)
            p_sensor = R_ma.dot(p_vessel - T)

            # Convert to pixels by applying camera calibration
            y, x = self.camera_m2p(p_sensor[1], p_sensor[2], p_sensor[0], camera_matrix, dist_coeffs)
                    
            p_transformed[i,0] = np.round(x)
            p_transformed[i,1] = np.round(y)
            if p_sensor[0] < 0 or p_transformed[i,0] >= dim_x or p_transformed[i,1] >= dim_y:
                p_transformed[i,:] = -1
                           
        return p_transformed
    
    
    def transform_points_to_radar(self, points, att, to_sensor_index, ref_location=None, range_filters=None, dim_x=4096, dim_y=3400):
        """
        Transform points in meters in (N,E,D) geographic frame to pixels coordinates (X,Y) in radar image.
        Assumes that points are relative to Seapath reference point.
        The argument 'points' should be a numpy array of shape (n,3), where n is the number of points.
        The argument 'ref_location' is the offset from the Seapath reference point to the reference 
        point used for the input points in meters and vessel body coordinates (fwd,stb,dwn). If None, the offset is 
        assumed to be (0,0,0).
        If the argument 'range_filters' is None, default radar range filters are used.
        If argument 'dim_x' is None, default radar image height 4096 is used.
        If argument 'dim_y' is None, default radar image width 4300 is used.
        The methods returns a numpy array of shape (n,2), where n is the number of input points.
        If a point falls outside the image boundaries, the coordinates for this points are set to -1.
        """
        cfg_sensor = self.get_sensor_config(self.TYPE_RADAR, to_sensor_index)  
        
        if range_filters is None:
            range_filters = RADAR_RANGE_FILTERS_DEFAULT
            
        # Get rotation matrices
        R_ma = self.rot_matrix_from_euler(cfg_sensor['rotation']).transpose()
        R_att = self.rot_matrix_from_euler(att).transpose()
        
        # Get translation vector
        T = np.array(cfg_sensor['location']).reshape(3,1)
        if ref_location is not None:
            T -= np.array(ref_location).reshape(3,1)
            
        p_transformed = np.zeros((points.shape[0], 2)).astype(np.int16)
        for i in range(points.shape[0]):
            p = points[i,:].reshape((3,1))
            
            # Translate and rotate target pos vector to sensor frame
            p_vessel = R_att.dot(p)
            p_sensor = R_ma.dot(p_vessel - T)
        
            # Calc bearing and range
            rng = math.sqrt(p_sensor[0]**2 + p_sensor[1]**2 + p_sensor[2]**2)
            brg = math.atan2(p_sensor[1], p_sensor[0]) * 180 / math.pi
            if brg < 0.0:
                brg += 360
                
            p_transformed[i,0] = 0.5 + dim_x - (dim_x/360.0) * brg
            p_transformed[i,1] = 0.5 + self.radar_m2p([rng], range_filters, cfg_sensor['m_per_sample'])
            if p_transformed[i,1] >= dim_y:
                p_transformed[i,:] = -1
                           
        return p_transformed
        
    
    def get_metadata(self, file_meta):
        if not file_meta or not os.path.isfile(file_meta):
            return {}
        
        meta = {}
        try:
            with open(file_meta, 'r') as f:
                meta = json.load(f)
        except:
            print('data_loader: Unable to read meta data file {}'.format(file_meta))
            
        return meta
    

    def get_metadata_by_basename(self, file_basename):
        """
        file_basename must contain full path, eg. '2017-10-13\2017-10-13-15\Cam0\Lens1\2017-10-13-15_08_50_470000'
        """         
        return self.get_metadata(os.path.join(self.path, file_basename) + '.json')
    
    
    def get_metadata_by_time(self, t, sensor_type, sensor_index, subsensor_index=None):
        file_meta = self.get_filename_sec(t, self.get_sensor_folder(sensor_type, sensor_index, subsensor_index), 'json')
        if file_meta == None:
            return {}              
        return self.get_metadata(file_meta)
    
    
    def get_horizon_layer(self, pva, cfg, spt_height, image_dim):
        # Get sensor height above sea level        
        pos_spt = np.array(pva['position'])
        pos_spt[2] = spt_height
        if 'heave' in pva.keys():
            pos_spt[2] -= pva['heave']
        pos_sensor = self.translate_pos(pos_spt, np.array(pva['attitude']), np.array(cfg['location']))
        height_above_sea_level = pos_sensor[2]
        
        # Get distance to horizon
        dist_to_horizon = np.sqrt(2 * EARTH_RADIUS * height_above_sea_level + height_above_sea_level**2)     
        
        # Get additional height caused by earth curvature
        dhgt = math.sqrt(EARTH_RADIUS**2 + dist_to_horizon**2) - EARTH_RADIUS

        # Get rotation matrices
        R_ma = self.rot_matrix_from_euler(cfg['rotation']).transpose()
        R_att = self.rot_matrix_from_euler(pva['attitude']).transpose()
        camera_matrix = np.array(cfg['camera_matrix']).reshape((3,3))
        dist_coeffs = np.array(cfg['distortion_coefficients'])
        
        numb = 72
        points = list()
        for i in range(numb):
            step = 360/numb
            start = pva['attitude'][2] + cfg['rotation'][2] - 180
            x = dist_to_horizon * np.cos((start + i * step) * np.pi / 180)
            y = dist_to_horizon * np.sin((start + i * step) * np.pi / 180)
            z = height_above_sea_level + dhgt
            p = np.array([x, y, z]).reshape((3,1))
            p_vessel = R_att.dot(p)
            p_sensor = R_ma.dot(p_vessel)
            
            # Azimuth relative to sensor location and orientation
            azimuth = np.arctan2(p_sensor[1], p_sensor[0]) * 180 / np.pi
            if np.abs(azimuth) < 45:
                # Convert to pixels by applying camera calibration
                x, y = self.camera_m2p(p_sensor[1], p_sensor[2], p_sensor[0], camera_matrix, dist_coeffs)
                points.append((x, y))
    
        # Create layer with correct dimension
        dim_y, dim_x = image_dim[0], image_dim[1]
        layer = np.zeros((dim_y, dim_x), dtype=np.uint8)
        
        # Add lines between points
        points = np.round(points).astype(np.int).squeeze()
        nump = len(points)
        yp, xp = line(points[0][1], points[0][0], points[1][1], points[1][0])
        for i in range(2, nump):
            rr, cc = line(points[i-1][1], points[i-1][0], points[i][1], points[i][0])
            xp = np.append(xp, cc)
            yp = np.append(yp, rr)
           
        # Draw lines
        ind = (yp>=0) & (xp>=0) & (yp<dim_y) & (xp<dim_x)
        xp = xp[ind]
        yp = yp[ind]
        layer[yp, xp] = 1
        
        if len(yp) > 0:             
            return layer
        else:
            return []
    
    
    def get_polygon_layer(self, targets, cfg, image_dim, range_filters=None):
        """
        Function for getting a AIS target segmentation map for camera and radar images.
        """
        if targets == {}:
            return []
        
        # Create layer with correct dimension
        dim_y, dim_x = image_dim[0], image_dim[1]
        layer = np.zeros((dim_y, dim_x), dtype=np.uint8)
        
        # Loop polygons in target list 
        any_targets = False
        for k, v in targets.items():
            if 'relpos' in v.keys():
                # Polygon found, add it
                normalize = False
                if cfg['type'] == 'radar':
                    xm = np.array(v['relpos']['sensor_frame']['polygon']['range'])
                    ym = np.array(v['relpos']['sensor_frame']['polygon']['azimuth'])
                    # Avoid wrapping of points in polygon
                    diff = ym.max() - ym.min()
                    if diff > 180: 
                        for i in range(ym.size):
                            if ym[i] < 0.0:
                                ym[i] = ym[i] + 360.0
                    else:
                        ym = ym + 180
                        normalize = True
                    # Convert to pixels
                    y = dim_y - (dim_y/360.0) * ym
                    x = self.radar_m2p(xm, range_filters, cfg['m_per_sample'])
                else:
                    # Convert to cartesian coordinates
                    dst = np.array(v['relpos']['sensor_frame']['polygon']['distance'])
                    rng = np.array(v['relpos']['sensor_frame']['polygon']['range'])
                    azm = np.array(v['relpos']['sensor_frame']['polygon']['azimuth'])
                    inc = np.array(v['relpos']['sensor_frame']['polygon']['inclin'])
                    zm =  dst * np.cos(azm*np.pi/180);
                    xm =  dst * np.sin(azm*np.pi/180);
                    ym = -rng * np.sin(inc*np.pi/180);
                    # Convert to pixels by applying camera calibration
                    camera_matrix = np.array(cfg['camera_matrix']).reshape((3,3))
                    dist_coeffs = np.array(cfg['distortion_coefficients'])
                    x, y = self.camera_m2p(xm, ym, zm, camera_matrix, dist_coeffs)
    
                # Add lines between extreme points in x and y to avoid broken polygons for small targets
                x = np.round(x).astype(np.int)
                y = np.round(y).astype(np.int)
                yp, xp = line(y.min(), x[y.argmin()], y.max(), x[y.argmax()])
                rr, cc = line(y[x.argmin()], x.min(), y[x.argmax()], x.max())
                xp = np.append(xp, cc)
                yp = np.append(yp, rr)
                ind = (yp>=0) & (xp>=0) & (yp<dim_y) & (xp<dim_x)
                xp = xp[ind]
                yp = yp[ind]
                
                # Create array of pixels in polygon 
                yy, xx = polygon(y, x, shape=(dim_y, dim_x))
                xp = np.append(xp, xx)
                yp = np.append(yp, yy)
                if yp.shape[0] == 0:
                    continue
    
                if normalize:
                    for i in range(yp.size):
                        yp[i] = yp[i] - dim_y / 2
                        if yp[i] < 0.0:
                            yp[i] = yp[i] + dim_y
                              
                layer[yp, xp] = 1
                if len(yp) > 0:
                    any_targets = True
        
        if any_targets:             
            return layer
        else:
            return []
            

    def camera_m2p(self, xm, ym, zm, camera_matrix, dist_coeffs):
        """
        Function for converting cartesian points in camera frame to pixels.
        This includes first distortion of points according to camera distortion coefficients, 
        then applying the camera matrix to achieve pixel coordinates.
        """
        x = xm / zm
        y = ym / zm
    
        fx = camera_matrix[0,0]
        fy = camera_matrix[1,1]
        ux = camera_matrix[0,2]
        uy = camera_matrix[1,2]
        
        k1 = dist_coeffs[0]
        k2 = dist_coeffs[1]
        p1 = dist_coeffs[2]
        p2 = dist_coeffs[3]
        k3 = dist_coeffs[4]
    
        # Correct radial distortion    
        r2 = x*x + y*y
        xCorrected = x * (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        yCorrected = y * (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    
        # Correct tangential distortion 
        xCorrected = xCorrected + (2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x))
        yCorrected = yCorrected + (p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y)
    
        # Ideal coordinates => actual coordinates
        xCorrected = xCorrected * fx + ux
        yCorrected = yCorrected * fy + uy
          
        return xCorrected, yCorrected
    
    
    def camera_p2m(self, xp, yp, camera_matrix, dist_coeffs):
        """
        Function for converting pixel coordinates in camera image to cartesian points.
        """
        fx = camera_matrix[0,0]
        fy = camera_matrix[1,1]
        ux = camera_matrix[0,2]
        uy = camera_matrix[1,2]
        
        k1 = dist_coeffs[0]
        k2 = dist_coeffs[1]
        p1 = dist_coeffs[2]
        p2 = dist_coeffs[3]
        k3 = dist_coeffs[4]
    
        # Apply camera matrix to get meters
        xm = (xp - ux) / fx
        ym = (yp - uy) / fy
       
        # Correct tangential distortion 
        r2 = xm**2 + ym**2
        xm = xm - (2.0 * p1 * xm * ym + p2 * (r2 + 2.0 * xm * xm))
        ym = ym - (p1 * (r2 + 2.0 * ym * ym) + 2.0 * p2 * xm * ym)
        
        # Correct radial distortion  
        r2 = xm**2 + ym**2
        xmt = xm / (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        ymt = ym / (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        r2 = xmt**2 + ymt**2
        xm = xm / (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        ym = ym / (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        
        return xm, ym
    
    
    def radar_m2p(self, m, range_filters, m_per_sample):
        """
        Function for converting radar range in meters to pixels according to range filters in .txt radar image metadata file.
        """
        m = np.array(m)
        p = np.zeros(len(m))
        num_f = len(range_filters)
        for i in range(len(m)):
            f_acc = 0
            for j in range(num_f):
                f_acc = f_acc + range_filters[j][1] * m_per_sample * range_filters[j][0]
                p[i] = p[i] + range_filters[j][1]
                if m[i] < f_acc or j == (num_f-1):
                    p[i] = p[i] - (f_acc - m[i]) / (m_per_sample * range_filters[j][0])  
                    break
        return p
  
    
    def radar_p2m(self, px, range_filters, m_per_sample):
        """
        Function for converting radar range in pixels to meters according to range filters in .txt radar image metadata file.
        """
        px = np.array(px)
        m = np.zeros(len(px))
        num_f = len(range_filters)
        for i in range(len(px)):
            px_acc = 0
            for j in range(num_f):
                px_next = range_filters[j][1]
                if j < (num_f-1) and px[i] > (px_acc + px_next):
                    m[i] += range_filters[j][1] * range_filters[j][0] * m_per_sample
                else:
                    m[i] += (px[i] - px_acc) * range_filters[j][0] * m_per_sample
                    break
                px_acc = px_acc + px_next    
        return m
          
    
    def polar2cart(self, r, theta, center):
    
        y = -r * np.cos(theta) + center[0]
        x = -r * np.sin(theta) + center[1]
        return x, y


    def img_cart2polar(self, img, center, final_radius, initial_radius = None, phase_width = 500):
    
        if initial_radius is None:
            initial_radius = 0
    
        theta , R = np.meshgrid(np.linspace(0, 2*np.pi, phase_width), 
                                np.arange(initial_radius, final_radius))
    
        Xcart, Ycart = self.polar2cart(R, theta, center)
    
        Xcart = Xcart.astype(int)
        Ycart = Ycart.astype(int)
    
        if img.ndim == 3:
            polar_img = img[Ycart,Xcart,:]
            polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width,img.shape[2]))
        else:
            polar_img = img[Ycart,Xcart]
            polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width))
    
        return polar_img.swapaxes(0,1)


    def get_polar_chart(self, pos, att, range_filters, m_per_sample, image_dim):
        
        # Request images of different scale from WMS chart server
        dim_y, dim_x = image_dim[0], image_dim[1]
        im = np.zeros((dim_y, dim_x, 1), dtype=np.uint8)
        x_offset = 0
        for j in range(1):
            if j==0:
                pixels_acc = range_filters[j][1]
            else:
                pixels_acc = pixels_acc*(range_filters[j-1][0] / range_filters[j][0]) + range_filters[j][1]
                
            scale = range_filters[j][0]*m_per_sample
            dim = int(2*pixels_acc)
            center = int(dim/2)
            url = 'http://navdemo:9669/WmsServer?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image/png&LAYERS=CHART&styles=base&HEIGHT={}&WIDTH={}&REFLATLON={},{},{},{},{},{}'.format(dim, dim, pos[0]*np.pi/180, pos[1]*np.pi/180, center, center, scale, att[2]*np.pi/180)
            r = requests.get(url)
            with Image.open(BytesIO(r.content)) as im_pil:
                im_cart = np.array(im_pil, dtype=np.uint8).reshape(dim,dim,1)
                
            im_pol = self.img_cart2polar(im_cart, (center, center), center, phase_width=dim_y)
            if x_offset+range_filters[j][1] <= dim_x:
                im[:,x_offset:x_offset+range_filters[j][1],:] = im_pol[:dim_y,-range_filters[j][1]:,:]
            else:
                im[:,x_offset:,:] = im_pol[:dim_y,-range_filters[j][1]:-range_filters[j][1]+dim_x-x_offset,:]
                break
            
            x_offset = x_offset + range_filters[j][1]
            if x_offset == dim_x:
                break
        im = (im[:,:,0]!=22).astype(np.uint8)
        return im
    
    
    def transform_ais_target(self, target, pva, cfg, spt_height, cfg_ais):
        """
        Function for calculating and transforming relative AIS target position/polygon to sensor location and orientation.
        """
        
        # Delete 'relpos' key to indicate that no position is calculated yet
        del target['relpos']
        
        # Check if moored, anchored or aground
        moving = True
        if 'nav_status' in target.keys():
            if target['nav_status'] == 1 or target['nav_status'] == 5 or target['nav_status'] == 6:
                moving = False
                
        # Check is age is within limits
        if moving:
            if abs(target['age']) > cfg_ais['timeout_moving']:
                return
        else:
            if abs(target['age']) > cfg_ais['timeout_static']:
                return
            
        # Get sensor position and height above sea level        
        pos_spt = np.array(pva['position'])
        pos_spt[2] = spt_height
        if 'heave' in pva.keys():
            pos_spt[2] -= pva['heave']
        pos_sensor = self.translate_pos(pos_spt, np.array(pva['attitude']), np.array(cfg['location']))
        height_above_sea_level = pos_sensor[2]

        # Convert decimal degrees to radians 
        lat1, lon1, lat2, lon2 = map(math.radians, [pos_sensor[0], pos_sensor[1], target['position'][0], target['position'][1]])

        # Get relative position to target in meters
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        x = self.dlat2m(dlat)
        y = self.dlon2m(dlon, lat2)
        
        # Get sensor height
        dhgt = math.sqrt(EARTH_RADIUS**2 + math.sqrt(x**2 + y**2)**2) - EARTH_RADIUS # Additional height caused by earth curvature
        z = height_above_sea_level + dhgt
        
        # Interpolate to current time
        if 'sog' in target.keys():
            cog = target['cog'] * np.pi / 180
            x_vel = target['sog'] * math.cos(cog)
            y_vel = target['sog'] * math.sin(cog)
            x = x + target['age'] * x_vel
            y = y + target['age'] * y_vel
        
        # Get rotation matrices
        R_ma = self.rot_matrix_from_euler(cfg['rotation']).transpose()
        R_att = self.rot_matrix_from_euler(pva['attitude']).transpose()
        
        # Translate and rotate target pos vector to sensor frame
        p = np.array([x, y, z]).reshape((3,1))
        p_vessel = R_att.dot(p)
        p_sensor = R_ma.dot(p_vessel)
                
        # Transform target polygon to sensor frame
        shape = 'ship'
        if 'dimension' in target.keys() and (target['dimension'][0] > 0 or target['dimension'][1] > 0) and (target['dimension'][2] > 0 or target['dimension'][3] > 0):
            a,b,c,d = target['dimension'][0], target['dimension'][1], target['dimension'][2], target['dimension'][3]
            if 'true_heading' in target.keys():
                heading_target = target['true_heading'] * np.pi / 180.0
            else:
                a = max(target['dimension'])
                shape = 'octagon'
        else:
            a = 1 # Assume minimum size of 1 meter
            shape = 'octagon'
        
        if shape == 'ship':
            # Draw a ship shape
            corner = 0.85*(a+b)-b
            x_poly0 = np.array([-b,corner,a,corner,-b,-b])
            y_poly0 = np.array([-c,-c,(d-c)/2,d,d,-c])
            x_poly = x_poly0 * np.cos(heading_target) - y_poly0 * np.sin(heading_target)
            y_poly = x_poly0 * np.sin(heading_target) + y_poly0 * np.cos(heading_target)
            z_poly = np.zeros(x_poly.shape)
        else:
            # Draw an octagon
            x_poly = a*np.cos(np.array([np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi, 0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi]))
            y_poly = a*np.sin(np.array([np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi, 0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi]))
            z_poly = np.zeros(x_poly.shape)
            
        # Add relative position
        x_poly = (x_poly + x)
        y_poly = (y_poly + y)
        z_poly = (z_poly + z)
        poly = np.zeros((3,x_poly.size))

        # Translate and rotate to sensor frame
        poly[0,:] = x_poly
        poly[1,:] = y_poly
        poly[2,:] = z_poly
        poly_vessel = R_att.dot(poly)
        poly_sensor = R_ma.dot(poly_vessel)
            
        # Position relative to sensor location
        relpos = {}
        rng = math.sqrt(x**2 + y**2 + z**2)
        distance = np.sqrt(x**2+y**2)
        brg_north = math.atan2(y, x) * 180 / math.pi
        if brg_north < 0:
            brg_north += 360.0
        relpos['north'] = x
        relpos['east'] = y
        relpos['down'] = z
        relpos['distance'] = distance
        relpos['range'] = rng
        relpos['bearing_north'] = brg_north
         
        # Position relative to sensor location and vessel orientation
        relpos['vessel_frame'] = {}
        relpos['vessel_frame']['distance'] = math.sqrt(p_vessel[0]**2 + p_vessel[1]**2)
        relpos['vessel_frame']['azimuth'] = math.atan2(p_vessel[1], p_vessel[0]) * 180 / np.pi
        relpos['vessel_frame']['inclin'] = -math.asin(p_vessel[2]/rng) * 180 / np.pi
              
        # Position relative to sensor location and orientation
        relpos['sensor_frame'] = {}
        relpos['sensor_frame']['distance'] = math.sqrt(p_sensor[0]**2 + p_sensor[1]**2)
        relpos['sensor_frame']['azimuth'] = math.atan2(p_sensor[1], p_sensor[0]) * 180 / np.pi
        relpos['sensor_frame']['inclin'] = -math.asin(p_sensor[2]/rng) * 180 / np.pi

        # Vessel polygon relative to sensor location and orientation
        relpos['sensor_frame']['polygon'] = {}
        rng_poly = np.sqrt(poly_sensor[0,:]**2 + poly_sensor[1,:]**2 + poly_sensor[2,:]**2)
        az_poly = np.arctan2(poly_sensor[1,:], poly_sensor[0,:]) * 180 / np.pi
        relpos['sensor_frame']['polygon']['range'] = list(rng_poly)
        relpos['sensor_frame']['polygon']['distance'] = list(np.sqrt(poly_sensor[0,:]**2 + poly_sensor[1,:]**2))
        relpos['sensor_frame']['polygon']['azimuth'] = list(az_poly)
        relpos['sensor_frame']['polygon']['inclin'] = list(-np.arcsin(poly_sensor[2,:] / rng_poly) * 180 / np.pi)
        
        # Check if ouside HFOV
        if cfg['type'] == 'camera' and (abs(az_poly) > 40.0).all():
            return  
        
        # Check if target is below horizon
        dist_to_horizon = np.sqrt(2 * EARTH_RADIUS * height_above_sea_level + height_above_sea_level**2)
        if cfg['type'] == 'camera' and distance > dist_to_horizon:
            return 

        target['relpos'] = relpos


    def transform_arpa_target(self, target, pva, cfg, cfg_arpa, spt_height):
        """
        Function for transforming relative ARPA target position to sensor location and orientation.
        """
        
        # TODO: Handle all fields correctly
        
        # Check is age is within limits
        if abs(target['age']) > cfg_arpa['timeout']:
            return
            
        # Check if status is tracked
        if not target['status'] == 'T':
            return
            
        # Get cunning point and height above sea level        
        pos_spt = np.array(pva['position'])
        pos_spt[2] = spt_height
        if 'heave' in pva.keys():
            pos_spt[2] -= pva['heave']
        pos_conning = self.translate_pos(pos_spt, np.array(pva['attitude']), np.array(cfg_arpa['conning_point']['location']))
        height_above_sea_level_conning = pos_conning[2]
        
        # Get relative position to target in meters
        x = 1852 * target['distance'] * math.cos(target['bearing'] * math.pi / 180)
        y = 1852 * target['distance'] * math.sin(target['bearing'] * math.pi / 180)
        
        # Get additional height caused by earth curvature
        dhgt = math.sqrt(EARTH_RADIUS**2 + (1852 * target['distance'])**2) - EARTH_RADIUS 
        z = height_above_sea_level_conning + dhgt
            
        # Interpolate to current time
        cog = target['cog'] * np.pi / 180
        x_vel = target['sog'] * math.cos(cog) * 0.514444
        y_vel = target['sog'] * math.sin(cog) * 0.514444
        x = x + (target['age']) * x_vel
        y = y + (target['age']) * y_vel
        
        # Get rotation matrices
        R_ma = self.rot_matrix_from_euler(cfg['rotation']).transpose()
        R_att = self.rot_matrix_from_euler(pva['attitude']).transpose()
        
        # Translate and rotate target pos vector to sensor frame
        # Assume that target position is measured on radar 0
        T = (np.array(cfg['location']) - np.array(cfg_arpa['conning_point']['location'])).reshape(3,1)
        loc_rotated = R_att.transpose().dot(T)
        p = np.array([x, y, z]).reshape((3,1)) - loc_rotated
        p_vessel = R_att.dot(p)
        p_sensor = R_ma.dot(p_vessel)
        p = p.squeeze()
        
        # Position relative to sensor location
        relpos = {}
        rng = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
        distance = np.sqrt(p[0]**2 + p[1]**2)
        brg_north = math.atan2(p[1], p[0]) * 180 / math.pi
        if brg_north < 0:
            brg_north += 360.0
              
        relpos['north'] = p[0]
        relpos['east'] = p[1]
        relpos['down'] = p[2]
        relpos['distance'] = distance
        relpos['range'] = rng
        relpos['bearing_north'] = brg_north
         
        # Position relative to sensor location and vessel orientation
        relpos['vessel_frame'] = {}
        relpos['vessel_frame']['distance'] = math.sqrt(p_vessel[0]**2 + p_vessel[1]**2)
        relpos['vessel_frame']['azimuth'] = math.atan2(p_vessel[1], p_vessel[0]) * 180 / np.pi
        relpos['vessel_frame']['inclin'] = -math.asin(p_vessel[2]/rng) * 180 / np.pi
              
        # Position relative to sensor location and orientation
        relpos['sensor_frame'] = {}
        relpos['sensor_frame']['distance'] = math.sqrt(p_sensor[0]**2 + p_sensor[1]**2)
        relpos['sensor_frame']['azimuth'] = math.atan2(p_sensor[1], p_sensor[0]) * 180 / np.pi
        relpos['sensor_frame']['inclin'] = -math.asin(p_sensor[2]/rng) * 180 / np.pi

        # Vessel polygon relative to sensor location and orientation
        relpos['sensor_frame']['polygon'] = {}
        relpos['sensor_frame']['polygon']['range'] = [rng]
        relpos['sensor_frame']['polygon']['distance'] = [relpos['sensor_frame']['distance']]
        relpos['sensor_frame']['polygon']['azimuth'] = [relpos['sensor_frame']['azimuth']]
        relpos['sensor_frame']['polygon']['inclin'] = [relpos['sensor_frame']['inclin']]
        
        # Check if ouside HFOV
        if cfg['type'] == 'camera' and abs(relpos['sensor_frame']['azimuth']) > 40.0:
            return 
        
        # Check if target is below horizon
        dist_to_horizon = np.sqrt(2 * EARTH_RADIUS * p[2] + p[2]**2)
        if cfg['type'] == 'camera' and distance > dist_to_horizon:
            return 

        target['relpos'] = relpos
              
                            
    def radar_interpolate_targets(self, targets0, targets1):
        """
        Function for compensating target position/polygons for a sweep time of about 2.5 seconds in a radar image.
        """
        targets = copy.deepcopy(targets0)
        for mmsi, dat0 in targets0.items():
            if mmsi in targets1.keys():
                dat1 = targets1[mmsi]
            else:
                del targets[mmsi]
                continue
            
            if not 'relpos' in dat0.keys() or not 'relpos' in dat1.keys():
                del targets[mmsi]
                continue
            
            # Iterate to find target azimuth and interpolation weight
            delta = 0
            az0 = dat0['relpos']['sensor_frame']['azimuth']
            az1 = dat1['relpos']['sensor_frame']['azimuth']
            if az0 < 0:
                az0 = az0 + 360
            if az1 < 0:
                az1 = az1 + 360
            if az0 < 90 and az1 > 270:
                az1 = az1 - 360
            elif az1 < 90 and az0 > 270:
                az0 = az0 - 360
            for i in range(4):
                delta = (1-delta) * az0 + delta * az1
                if delta < 0:
                    delta = delta + 360
                delta = delta / 360.0
            
            # Interpolate relative position
            relpos = {}
            relpos['north'] = (1-delta) * dat0['relpos']['north'] + delta * dat1['relpos']['north']
            relpos['east'] = (1-delta) * dat0['relpos']['east'] + delta * dat1['relpos']['east']
            relpos['down'] = (1-delta) * dat0['relpos']['down'] + delta * dat1['relpos']['down']
            relpos['range'] = (1-delta) * dat0['relpos']['range'] + delta * dat1['relpos']['range']
            relpos['distance'] = (1-delta) * dat0['relpos']['distance'] + delta * dat1['relpos']['distance']
            if dat0['relpos']['bearing_north'] > 270 and dat1['relpos']['bearing_north'] < 90:
                relpos['bearing_north'] = (1-delta) * (dat0['relpos']['bearing_north'] - 360) + delta * dat1['relpos']['bearing_north']
                if relpos['bearing_north'] < 0:
                    relpos['bearing_north'] = relpos['bearing_north'] + 360
            elif dat1['relpos']['bearing_north'] > 270 and dat0['relpos']['bearing_north'] < 90:
                relpos['bearing_north'] = (1-delta) * dat0['relpos']['bearing_north'] + delta * (dat1['relpos']['bearing_north'] - 360)
                if relpos['bearing_north'] < 0:
                    relpos['bearing_north'] = relpos['bearing_north'] + 360
            else:
                relpos['bearing_north'] = (1-delta) * dat0['relpos']['bearing_north'] + delta * dat1['relpos']['bearing_north']
            
            relpos['vessel_frame'] = {}
            relpos['vessel_frame']['distance'] = (1-delta) * dat0['relpos']['vessel_frame']['distance'] + delta * dat1['relpos']['vessel_frame']['distance']
            relpos['vessel_frame']['azimuth'] = delta * 360
            if relpos['vessel_frame']['azimuth'] > 180:
                relpos['vessel_frame']['azimuth'] = relpos['vessel_frame']['azimuth'] - 360
            relpos['vessel_frame']['inclin'] = (1-delta) * dat0['relpos']['vessel_frame']['inclin'] + delta * dat1['relpos']['vessel_frame']['inclin']
            
            relpos['sensor_frame'] = {}
            relpos['sensor_frame']['distance'] = (1-delta) * dat0['relpos']['sensor_frame']['distance'] + delta * dat1['relpos']['sensor_frame']['distance']
            relpos['sensor_frame']['azimuth'] = delta * 360
            if relpos['sensor_frame']['azimuth'] > 180:
                relpos['sensor_frame']['azimuth'] = relpos['sensor_frame']['azimuth'] - 360
            relpos['sensor_frame']['inclin'] = (1-delta) * dat0['relpos']['sensor_frame']['inclin'] + delta * dat1['relpos']['sensor_frame']['inclin']
            
            # Interpolate points in vessel polygon
            range0 = np.array(dat0['relpos']['sensor_frame']['polygon']['range'])
            range1 = np.array(dat1['relpos']['sensor_frame']['polygon']['range'])
            dist0 = np.array(dat0['relpos']['sensor_frame']['polygon']['distance'])
            dist1 = np.array(dat1['relpos']['sensor_frame']['polygon']['distance'])
            azimuth0 = np.array(dat0['relpos']['sensor_frame']['polygon']['azimuth'])
            azimuth1 = np.array(dat1['relpos']['sensor_frame']['polygon']['azimuth'])
            inclin0 = np.array(dat0['relpos']['sensor_frame']['polygon']['inclin'])
            inclin1 = np.array(dat1['relpos']['sensor_frame']['polygon']['inclin'])
            rang = 0*range0
            dist = 0*dist0
            azimuth = 0*azimuth0
            inclin = 0*inclin0
            for i in range(azimuth0.size):
                delta = 0
                az0 = azimuth0[i]
                az1 = azimuth1[i]
                if az0 < 0:
                    az0 = az0 + 360
                if az1 < 0:
                    az1 = az1 + 360
                if az0 < 90 and az1 > 270:
                    az1 = az1 - 360
                elif az1 < 90 and az0 > 270:
                    az0 = az0 - 360
                for j in range(4):
                    delta = (1-delta) * az0 + delta * az1
                    if delta < 0:
                        delta = delta + 360
                    delta = delta / 360.0
                
                rang[i] = (1-delta) * range0[i] + delta * range1[i]
                dist[i] = (1-delta) * dist0[i] + delta * dist1[i]
                azimuth[i] = delta * 360
                if azimuth[i] > 180:
                    azimuth[i] = azimuth[i] - 360
                inclin[i] = (1-delta) * inclin0[i] + delta * inclin1[i]
            
            relpos['sensor_frame']['polygon'] = {}
            relpos['sensor_frame']['polygon']['range'] = list(rang)
            relpos['sensor_frame']['polygon']['distance'] = list(dist)
            relpos['sensor_frame']['polygon']['azimuth'] = list(azimuth)
            relpos['sensor_frame']['polygon']['inclin'] = list(inclin)
    
            targets[mmsi]['relpos'] = relpos
            
        return targets
    
    
    def dlat2m(self, dlat):
        """
        Function for converting difference in latitude (radians) to meters.
        """
        return EARTH_RADIUS * dlat
       
        
    def m2dlat(self, m):
        """
        Function for converting meters to difference in latitude (radians).
        """
        return m / EARTH_RADIUS
    
       
    def dlon2m(self, dlon, lat):
        """
        Function for converting difference in longitude (radians) to meters.
        """
        return EARTH_RADIUS * dlon * math.cos(lat)
    
    
    def m2dlon(self, m, lat):
        """
        Function for converting meters to difference in longitude (radians).
        """
        return m / (EARTH_RADIUS * math.cos(lat))
    
    
    def rot_matrix_from_euler(self, att):
        """
        Function for creating a rotation matrix from euler angles. Assumed rotation order is RPY.
        """
        # Roll rotation
        Rx = np.zeros((3,3))
        Rx[0,0] = 1
        Rx[1,1] = np.cos(att[0] * np.pi / 180)
        Rx[1,2] = -np.sin(att[0] * np.pi / 180)
        Rx[2,1] = np.sin(att[0] * np.pi / 180)
        Rx[2,2] = np.cos(att[0] * np.pi / 180)
        # Pitch rotation
        Ry = np.zeros((3,3))
        Ry[0,0] = np.cos(att[1] * np.pi / 180)
        Ry[0,2] = np.sin(att[1] * np.pi / 180)
        Ry[1,1] = 1
        Ry[2,0] = -np.sin(att[1] * np.pi / 180)
        Ry[2,2] = np.cos(att[1] * np.pi / 180)
        # Yaw rotation
        Rz = np.zeros((3,3))
        Rz[0,0] = np.cos(att[2] * np.pi / 180)
        Rz[0,1] = -np.sin(att[2] * np.pi / 180)
        Rz[1,0] = np.sin(att[2] * np.pi / 180)
        Rz[1,1] = np.cos(att[2] * np.pi / 180)
        Rz[2,2] = 1
            
        R = Rz.dot(Ry.dot(Rx))
        return R
    
    
    def translate_pos(self, pos, att, offset):
        """
        Function for translating a position given offset and attitude.
        """
        # reshape translation vector
        T = offset.reshape((3,1))
        
        # Get rotation matrix
        R_att = self.rot_matrix_from_euler(att)
        
        # Translate position
        loc_rotated = R_att.dot(T)
        lat = pos[0] + self.m2dlat(loc_rotated[0]) * 180 / np.pi
        lon = pos[1] + self.m2dlon(loc_rotated[1], pos[0] * np.pi / 180) * 180 / np.pi
        hgt = pos[2] - loc_rotated[2]
        return np.array((lat[0], lon[0], hgt[0]))