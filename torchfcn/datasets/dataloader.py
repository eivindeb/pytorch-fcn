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
import cmath
from skimage.draw import polygon, line, circle
from io import BytesIO
import requests
import traceback
import datetime
import struct
import bisect
from PIL import Image

pytorch = False
try:
    import torch
    pytorch = True
except:
    pass

opencv = False
try: 
    import cv2
    opencv = True
except:
    pass

# format string for the folder name (date + hour)
FOLDER_NAME_FORMAT = os.path.join('{:%Y-%m-%d', '%Y-%m-%d-%H}')

# format string for the file name (date + time)
FILE_NAME_FORMAT_SEC = '{:%Y-%m-%d-%H_%M_%S}'
FILE_NAME_FORMAT_MIN = '{:%Y-%m-%d-%H_%M}'

EARTH_RADIUS = 6367000
                
class SeapathReader():
    """
    Class for reading a seapath log file and fetching pos, vel and attitude for a given time tag.
    """
    def __init__(self, filename):
        self.filename = filename
        self.data = []
        self.ts = []
        self.num = 0

        with open(filename, "rb") as f:
            while True:
                msg = f.read(132)
                item = {}
    
                if len(msg) == 0:
                    break
    
                item['length'] = struct.unpack('H', msg[4:6])[0]
                item['version'] = struct.unpack('H', msg[6:8])[0]
                item['sec'] = struct.unpack('I', msg[8:12])[0]
                item['nsec'] = struct.unpack('I', msg[12:16])[0]
                item['status'] = hex(struct.unpack('I', msg[16:20])[0])
                item['lat'] = struct.unpack('d', msg[20:28])[0]
                item['lon'] = struct.unpack('d', msg[28:36])[0]
                item['hgt'] = struct.unpack('f', msg[36:40])[0]
                item['attitude'] = struct.unpack('3f', msg[40:52])
                item['heave'] = struct.unpack('f', msg[52:56])[0]   # positive downwards
                item['attrate'] = struct.unpack('3f', msg[56:68])
                item['velocity'] = struct.unpack('3f', msg[68:80])
                item['laterror'] = struct.unpack('f', msg[80:84])[0]
                item['lonerror'] = struct.unpack('f', msg[84:88])[0]
                item['hgterror'] = struct.unpack('f', msg[88:92])[0]
                item['atterror'] = struct.unpack('3f', msg[92:104])
                item['heaveerror'] = struct.unpack('f', msg[104:108])[0]
                item['acceleration'] = struct.unpack('3f', msg[108:120])
                item['sec_delheave'] = struct.unpack('I', msg[120:124])[0]   
                item['usec_delheave'] = struct.unpack('I', msg[124:128])[0]
                item['delheave'] = struct.unpack('f', msg[128:132])[0]
                
                t = datetime.datetime.utcfromtimestamp(int(item['sec']) + int(item['nsec'])/1000000000)
                item['utc'] = t
                self.ts.append(t)
                self.data.append(item)
                self.num = self.num + 1


    def get_posvelatt(self, t):
        """
        Function for fetching position, velocity and attitude for a given time tag and sensor location.
        """
        item = {}
        try:
            i = bisect.bisect_right(self.ts, t)

            if i == 0:
                item = self.data[0]
            elif i == self.num:
                item = self.data[-1]
            elif abs((t - self.data[i-1]['utc']).total_seconds()) < abs((t - self.data[i]['utc']).total_seconds()):
                item = self.data[i-1]
            else:
                item = self.data[i]
                
            diff = abs((t - item['utc']).total_seconds())  
            if diff > 0.020: # Allow maximum 20ms time difference
                item = {}
        except:
            pass
                
        pva = {}
        if not item == {}:
            # Extract position, velocity and attitude
            pva['utc'] = str(item['utc'])
            pva['status'] = item['status']
            status = int(item['status'], 0)
            if (status & 1) == 0:
                pva['position'] = (item['lat'], item['lon'], item['hgt'])
                pva['velocity'] = item['velocity']
                pva['poserror'] = (item['laterror'], item['lonerror'], item['hgterror'])
            if ((status >> 3) & 1) == 0:
                pva['heave'] = item['heave']
            if ((status >> 1) & 3) == 0:
                pva['attitude'] = item['attitude']
                pva['attrate'] = item['attrate']
                pva['atterror'] = item['atterror']
            
        return pva


class AisReader():
    """
    Class for reading an AIS log file and fetching target list for a given time tag.
    """
    def __init__(self, filename):
        self.filename = filename
        self.data = {}
        self.num = 0
        with open(filename, "r") as f:
            for l in f:
                self.num = self.num + 1
                msg = json.loads(l)
                if not msg['mmsi'] in self.data.keys():
                    self.data[msg['mmsi']] = {}
                self.data[msg['mmsi']][len(self.data[msg['mmsi']])] = msg
        
   
    def get_targets(self, t, cfg_ais, own_pos=None, max_range=None):
        """
        Function for fetching target list for a given time tag.
        The argument 'cfg_ais' is the configuration item 'AIS' from dataloader.json.
        If arguments 'own_pos' and 'max_range' is specified, targets ouside 'max_range' will be removed from list.
        """
        targets = {}
 
        # Loop AIS messages and build a target list
        for k,v in self.data.items():
            targets[k] = {}
            
            for i, msg in v.items():
                ID = msg['id']
                if ID == 5:
                    targets[k]['name'] = msg['name']
                    targets[k]['dimension'] = (msg['dim_a'], msg['dim_b'], msg['dim_c'], msg['dim_d'])
                    targets[k]['type_and_cargo'] = msg['type_and_cargo']                 
                    targets[k]['draught'] = msg['draught']
                    targets[k]['eta_month'] = msg['eta_month']
                    targets[k]['eta_day'] = msg['eta_day']
                    targets[k]['eta_hour'] = msg['eta_hour']
                    targets[k]['eta_minute'] = msg['eta_minute']
                    targets[k]['destination'] = msg['destination']
                elif ID == 24:
                    if msg['part_num'] == 0:
                        targets[k]['name'] = msg['name']
                    elif msg['part_num'] == 1:
                        targets[k]['dimension'] = (msg['dim_a'], msg['dim_b'], msg['dim_c'], msg['dim_d'])
                        targets[k]['type_and_cargo'] = msg['type_and_cargo']
                elif ID == 21:
                    t_msg = datetime.datetime.strptime(msg['utc'], '%Y-%m-%d %H:%M:%S.%f')
                    t_timestamp = t_msg
                    t_timestamp = t_timestamp.replace(microsecond = 0)
                    age = (t - t_timestamp).total_seconds()
                    if not 'age' in targets[k].keys() or abs(age) < abs(targets[k]['age']):
                        targets[k]['aton_type'] = msg['aton_type']
                        targets[k]['name'] = msg['name']
                        if abs(msg['y']) <= 90.0 and abs(msg['x']) <= 180.0:
                            targets[k]['position'] = (msg['y'], msg['x'])
                            targets[k]['position_accuracy'] = msg['position_accuracy']
                        targets[k]['dimension'] = (msg['dim_a'], msg['dim_b'], msg['dim_c'], msg['dim_d'])
                        targets[k]['off_pos'] = msg['off_pos']
                        targets[k]['aton_status'] = msg['aton_status']
                        targets[k]['virtual_aton'] = msg['virtual_aton']
                        targets[k]['assigned_mode'] = msg['assigned_mode']
                        targets[k]['msg_utc'] = str(t_msg)
                        targets[k]['age'] = age
                        targets[k]['pos_msg'] = ID       
                elif ID == 1 or ID == 2 or ID == 3:
                    t_msg = datetime.datetime.strptime(msg['utc'], '%Y-%m-%d %H:%M:%S.%f')
                    t_timestamp = t_msg
                    timestamp = msg['timestamp']
                    if timestamp < 60:
                        t_timestamp = t_timestamp.replace(second = timestamp, microsecond = 0)
                        if t_timestamp > t_msg:
                            t_timestamp = t_timestamp + datetime.timedelta(minutes = -1)
                        age = (t - t_timestamp).total_seconds()
                        if not 'age' in targets[k].keys() or abs(age) < abs(targets[k]['age']):
                            if abs(msg['y']) <= 90.0 and abs(msg['x']) <= 180.0:
                                targets[k]['position'] = (msg['y'], msg['x'])
                                targets[k]['position_accuracy'] = msg['position_accuracy']
                            targets[k]['nav_status'] = msg['nav_status']
                            if not msg['rot_over_range']:
                                targets[k]['rot'] = msg['rot']
                            if msg['sog'] <= 102.2 and msg['cog'] < 360.0:
                                targets[k]['sog'] = msg['sog'] * 0.514444 # Convert from knots to m/s
                                targets[k]['cog'] = msg['cog'] # Already in degrees
                            if msg['true_heading'] < 360:
                                targets[k]['true_heading'] = msg['true_heading']
                            targets[k]['special_manoeuvre'] = msg['special_manoeuvre']
                            targets[k]['msg_utc'] = str(t_msg)
                            targets[k]['timestamp'] = str(t_timestamp)
                            targets[k]['age'] = age
                            targets[k]['pos_msg'] = ID 
                elif ID == 18:
                    t_msg = datetime.datetime.strptime(msg['utc'], '%Y-%m-%d %H:%M:%S.%f')
                    t_timestamp = t_msg
                    timestamp = msg['timestamp']
                    if timestamp < 60:
                        t_timestamp = t_timestamp.replace(second = timestamp, microsecond = 0)
                        if t_timestamp > t_msg:
                            t_timestamp = t_timestamp + datetime.timedelta(minutes = -1)
                        age = (t - t_timestamp).total_seconds()
                        if not 'age' in targets[k].keys() or abs(age) < abs(targets[k]['age']):
                            if abs(msg['y']) <= 90.0 and abs(msg['x']) <= 180.0:
                                targets[k]['position'] = (msg['y'], msg['x'])
                                targets[k]['position_accuracy'] = msg['position_accuracy']
                            if msg['sog'] <= 102.2 and msg['cog'] < 360.0:
                                targets[k]['sog'] = msg['sog'] * 0.514444 # Convert from knots to m/s
                                targets[k]['cog'] = msg['cog'] # Already in degrees
                            if msg['true_heading'] < 360:
                                targets[k]['true_heading'] = msg['true_heading']
                            targets[k]['msg_utc'] = str(t_msg)
                            targets[k]['timestamp'] = str(t_timestamp)
                            targets[k]['age'] = age
                            targets[k]['pos_msg'] = ID 
                elif ID == 19:
                    targets[k]['name'] = msg['name']
                    targets[k]['dimension'] = (msg['dim_a'], msg['dim_b'], msg['dim_c'], msg['dim_d'])
                    targets[k]['type_and_cargo'] = msg['type_and_cargo']
                    t_msg = datetime.datetime.strptime(msg['utc'], '%Y-%m-%d %H:%M:%S.%f')
                    t_timestamp = t_msg
                    timestamp = msg['timestamp']
                    if timestamp < 60:
                        t_timestamp = t_timestamp.replace(second = timestamp, microsecond = 0)
                        if t_timestamp > t_msg:
                            t_timestamp = t_timestamp + datetime.timedelta(minutes = -1)
                        age = (t - t_timestamp).total_seconds()
                        if not 'age' in targets[k].keys() or abs(age) < abs(targets[k]['age']):
                            if abs(msg['y']) <= 90.0 and abs(msg['x']) <= 180.0:
                                targets[k]['position'] = (msg['y'], msg['x'])
                                targets[k]['position_accuracy'] = msg['position_accuracy']
                            if msg['sog'] <= 102.2 and msg['cog'] < 360.0:
                                targets[k]['sog'] = msg['sog'] * 0.514444 # Convert from knots to m/s
                                targets[k]['cog'] = msg['cog'] # Already in degrees
                            if msg['true_heading'] < 360:
                                targets[k]['true_heading'] = msg['true_heading']
                            targets[k]['msg_utc'] = str(t_msg)
                            targets[k]['timestamp'] = str(t_timestamp)
                            targets[k]['age'] = age
                            targets[k]['pos_msg'] = ID
                elif ID == 4 or ID == 6 or ID == 8 or ID == 9 or ID == 10 or ID == 11 or ID == 15 or ID == 16 or ID == 17 or ID == 20 or ID == 26 or ID == 27:
                    pass
                else:
                    print('AIS message not handled, id={}'.format(ID))
                    print(msg)
                
        for mmsi, dat in dict(targets).items():
              
            # Check if moored, anchored or aground
            moving = True
            if 'nav_status' in targets[mmsi].keys():
                if targets[mmsi]['nav_status'] == 1 or targets[mmsi]['nav_status'] == 5 or targets[mmsi]['nav_status'] == 6:
                    moving = False
            
            # Remove unwanted targets
            if not 'position' in targets[mmsi].keys():
                # No position, delete
                del targets[mmsi]
            elif mmsi == 257888888 or mmsi == 257999329 or mmsi == 257999349 or mmsi == 259322000:
                # Seatex test targets or MS Polarlys, delete
                del targets[mmsi]
            elif moving and (targets[mmsi]['age'] < -cfg_ais['timeout_moving'] or targets[mmsi]['age'] > cfg_ais['timeout_moving']):
                # Moving old target or from the future, delete
                del targets[mmsi]
            elif not moving and (targets[mmsi]['age'] < -cfg_ais['timeout_static'] or targets[mmsi]['age'] > cfg_ais['timeout_static']):
                # Static old target or from the future, delete
                del targets[mmsi]
            elif not self.check_range(targets[mmsi], own_pos, max_range):
                # Out of range, out of view
                del targets[mmsi]
                
        return targets 
                
                
    def check_range(self, target, own_pos, max_range):
        """
        The method return False if the target is outside max_range.
        """
        if own_pos and max_range:
            # Convert decimal degrees to radians 
            lat1, lon1, lat2, lon2 = map(math.radians, [own_pos[0], own_pos[1], target['position'][0], target['position'][1]])
    
            # Get relative position to target in meters
            dlon = lon2 - lon1 
            dlat = lat2 - lat1 
            x = DataLoader.dlat2m(dlat)
            y = DataLoader.dlon2m(dlon, lat2)
            
            # Check if out of range
            distance = np.sqrt(x**2+y**2)
            if distance > max_range:
                return False
            
            # Add position relative to reference point
            relpos = {}
            brg_north = math.atan2(y, x) * 180 / math.pi
            if brg_north < 0:
                brg_north += 360.0
            relpos['north'] = x
            relpos['east'] = y
            relpos['distance'] = distance
            relpos['bearing_north'] = brg_north
            target['relpos'] = relpos
            
        return True
    
    
class DataLoader():
    """
    Class for reading images and labels from Seatex Polarlys dataset.
    """
    def __init__(self, path, sensor_config='dataloader.json'):
        self.size = 0
        self.path = path
        self.sensor_config = None
        self.sensor_config_path = sensor_config
        self.TYPE_CAMERA = 0
        self.TYPE_RADAR = 1
        
        try:
            with open(sensor_config, 'r') as f:
                self.sensor_config = json.load(f)
        except:
            self.sensor_config = None
            print('data_loader: Unable to read configuration file {}'.format(sensor_config))
        
       
    def get_seapath_data(self, t):
        """
        Read position, velocity and attitude from Seapath log file for a specific time stamp.
        """
        folder = FOLDER_NAME_FORMAT.format(t)
        filename = '{:%Y-%m-%d-%H_%M}.bin'.format(t)
        
        path = os.path.join(self.path, folder)
        path = os.path.join(path, 'Seapath')
        path = os.path.join(path, filename)
        
        if not os.path.isfile(path):
            return {}
        
        sr = SeapathReader(path)
 
        return sr.get_posvelatt(t)
  
    
    def get_ais_targets(self, t, own_pos=None, max_range=None):
        """
        Read target list from AIS log file for a specific time stamp.
        If arguments 'own_pos' and 'max_range' is specified, targets ouside 'max_range' will be removed from list.
        """
        folder = FOLDER_NAME_FORMAT.format(t)
        filename = '{:%Y-%m-%d-%H_%M}.json'.format(t)
        
        path = os.path.join(self.path, folder)
        path = os.path.join(path, 'AIS')
        path = os.path.join(path, filename)

        if not os.path.isfile(path):
            return {}
        
        sr = AisReader(path)
        
        return sr.get_targets(t, self.sensor_config['AIS'], own_pos, max_range) 
            
    
    def change_path(self, path):
        self.path = path
        
        
    def get_filename_sec(self, t, sensor_path, extension):
        """
        Function for getting a file name for a specific second
        """
        folder = FOLDER_NAME_FORMAT.format(t)
        filename_sec = FILE_NAME_FORMAT_SEC.format(t)
        
        path_sec = os.path.join(self.path, folder)
        path_sec = os.path.join(path_sec, sensor_path)
        path_sec = os.path.join(path_sec, filename_sec)
        
        return path_sec + '.' + extension
    
    
    def get_sensor_from_basename(self, file_basename):
        """
        file_basename must contain full path, eg. '2017-10-13\2017-10-13-15\Cam0\Lens1\2017-10-13-15_08_50_470000'
        """
        sensor_type, sensor_index, subsensor_index = None, None, None
        if file_basename.find('Cam') >= 0:
            sensor_type = self.TYPE_CAMERA
            sensor_index = int(file_basename[file_basename.find('Cam') + 3])
            subsensor_index = int(file_basename[file_basename.find('Lens') + 4])
        elif file_basename.find('Radar') >= 0:
            sensor_type = self.TYPE_RADAR
            sensor_index = int(file_basename[file_basename.find('Radar') + 5])
            
        return sensor_type, sensor_index, subsensor_index
 

    def get_time_from_basename(self, file_basename):
        """
        file_basename must contain full path, eg. '2017-10-13\2017-10-13-15\Cam0\Lens1\2017-10-13-15_08_50_470000'
        """
        basename = os.path.split(file_basename)[1]
        t = datetime.datetime.strptime(basename, '%Y-%m-%d-%H_%M_%S')

        return t
       
        
    def get_sensor_folder(self, sensor_type, sensor_index, subsensor_index=None):
        folder = None
        if sensor_type == self.TYPE_CAMERA:
            folder = os.path.join('Cam{}'.format(sensor_index), 'Lens{}'.format(subsensor_index))
        elif sensor_type == self.TYPE_RADAR:
            folder = 'Radar{}'.format(sensor_index)
            
        return folder
    
    
    def get_sensor_config(self, sensor_type, sensor_index, subsensor_index=None):
        with open(self.sensor_config_path, 'r') as f:
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
    
    
    def load_image(self, t, sensor_type, sensor_index, subsensor_index=None):
        """
        Load an image for a given time tag and sensor from the Polarlys data set.
        The method returns a numpy array of shape (1920,2560,3) of type np.uint8 for camera images.
        The method returns a numpy array of shape (4096,3400) of type np.uint8 for radar images.
        """
        file_name = self.get_filename_sec(t, self.get_sensor_folder(sensor_type, sensor_index, subsensor_index), self.get_image_extension(sensor_type))
        if not file_name or not os.path.isfile(file_name):
            return []
        
        image = []
        try:
            if opencv:
                image = cv2.imread(file_name, -1)
                if image.ndim == 3:
                    image = image[:,:,::-1]
            else:
                with Image.open(file_name) as pil:
                    image = np.array(pil, dtype=np.uint8)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return image
       
  
    def load_chart_layer(self, t, dim, scale, binary=False):
        """
        Load chart layer for a given time tag in (N,E) coordinate frame. 
        The position of Polarlys at time 't' is used as origin in the chart (Seapath reference point).
        The argument 'dim' is the height and width of the returned image in pixels.
        The argument 'scale' is the number of meters pr. pixel in the returned image.
        The argument 'binary' decides if the output is a binary land mask or grayscale.
        Origin is located in the center (dim/2,dim/2) of the chart.
        The method returns a numpy array of shape (dim,dim) of type np.uint8.   
        The method also returns a dictionary with Seapath data for given time tag.
        """
        layer = []
        pva = {}
        try:
            pva = self.get_seapath_data(t)
            pos = pva['position']
            if binary:
                style = 'base'
            else:
                style = 'standard'
            url = 'http://navdemo:9669/WmsServer?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image/png&LAYERS=CHART&styles={}&HEIGHT={}&WIDTH={}&REFLATLON={},{},{},{},{},{}'.format(style, dim, dim, pos[0]*np.pi/180, pos[1]*np.pi/180, dim//2, dim//2, scale, 0)
            r = requests.get(url)
            with Image.open(BytesIO(r.content)) as im_pil:
                layer = np.array(im_pil, dtype=np.uint8).reshape(dim,dim)
            if binary:
                layer = (layer[:,:] != 22).astype(np.uint8)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return layer, pva
        
        
    def load_chart_layer_sensor(self, t, sensor_type, sensor_index, subsensor_index=None, binary=False):
        """
        Load chart layer for a given time tag and sensor. 
        The argument 'dim' is the height and width of the chart requested from the chart server in pixels.
        The argument 'scale' is the number of meters pr. pixel in the image requested from the chart server.
        The argument 'binary' decides if the output is a binary land mask or grayscale.
        The method returns a numpy array of shape (height,width) of type np.uint8.
        Height and width of returned image is the same as in original sensor image.
        """
        layer = []
        try:
            meta = self.get_metadata(t, sensor_type, sensor_index, subsensor_index)
            if sensor_type == self.TYPE_CAMERA:
                pos = meta['own_vessel']['position']
            else:
                pos = meta['own_vessel_end']['position']
            if binary:
                style = 'base'
            else:
                style = 'standard'
            dim = 2000
            scale = 20
            url = 'http://navdemo:9669/WmsServer?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image/png&LAYERS=CHART&styles={}&HEIGHT={}&WIDTH={}&REFLATLON={},{},{},{},{},{}'.format(style, dim, dim, pos[0]*np.pi/180, pos[1]*np.pi/180, dim//2, dim//2, scale, 0)
            r = requests.get(url)
            with Image.open(BytesIO(r.content)) as im_pil:
                chart_large = np.array(im_pil, dtype=np.uint8).reshape(dim,dim)
            if binary:
                chart_large = (chart_large[:,:] != 22).astype(np.uint8)
            chart_large_sensor = self.transform_image_to_sensor(t, sensor_type, sensor_index, chart_large, pos, subsensor_index, scale)

            scale = 4
            url = 'http://navdemo:9669/WmsServer?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image/png&LAYERS=CHART&styles={}&HEIGHT={}&WIDTH={}&REFLATLON={},{},{},{},{},{}'.format(style, dim, dim, pos[0]*np.pi/180, pos[1]*np.pi/180, dim//2, dim//2, scale, 0)
            r = requests.get(url)
            with Image.open(BytesIO(r.content)) as im_pil:
                chart_medium = np.array(im_pil, dtype=np.uint8).reshape(dim,dim)
            if binary:
                chart_medium = (chart_medium[:,:] != 22).astype(np.uint8)
            chart_medium_sensor = self.transform_image_to_sensor(t, sensor_type, sensor_index, np.stack([chart_medium, np.zeros((dim,dim), dtype=np.uint16) + 1], axis=2), pos, subsensor_index, scale)

            scale = 0.5
            url = 'http://navdemo:9669/WmsServer?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image/png&LAYERS=CHART&styles={}&HEIGHT={}&WIDTH={}&REFLATLON={},{},{},{},{},{}'.format(style, dim, dim, pos[0]*np.pi/180, pos[1]*np.pi/180, dim//2, dim//2, scale, 0)
            r = requests.get(url)
            with Image.open(BytesIO(r.content)) as im_pil:
                chart_small = np.array(im_pil, dtype=np.uint8).reshape(dim,dim)
            if binary:
                chart_small = (chart_small[:,:] != 22).astype(np.uint8)
            chart_small_sensor = self.transform_image_to_sensor(t, sensor_type, sensor_index, np.stack([chart_small, np.zeros((dim,dim), dtype=np.uint16) + 1], axis=2), pos, subsensor_index, scale)
            
            layer = chart_large_sensor
            layer = layer * (1 - chart_medium_sensor[:,:,1]) + chart_medium_sensor[:,:,0] * chart_medium_sensor[:,:,1]
            layer = layer * (1 - chart_small_sensor[:,:,1]) + chart_small_sensor[:,:,0] * chart_small_sensor[:,:,1]
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return layer
    
    
    def draw_polygon_layer(self, polygons, dim_x, dim_y):
        """
        Draw a list of polygons in an image of dimension (dim_x, dim_y).
        The polygons are filled.
        Returns a numpy array of type np.uint8.
        """
        layer = []
        try:
            layer = np.zeros((dim_x,dim_y)).astype(np.uint8)
            any_polygons = False
            for poly in polygons:
                # Add lines between extreme points in x and y to avoid broken polygons for small targets
                x = poly[:,0].astype(np.int16)
                y = poly[:,1].astype(np.int16)               
                        
                xp, yp = line(x.min(), y[x.argmin()], x.max(), y[x.argmax()])
                rr, cc = line(x[y.argmin()], y.min(), x[y.argmax()], y.max())
                xp = np.append(xp, rr)
                yp = np.append(yp, cc)
                ind = (xp>=0) & (yp>=0) & (xp<dim_x) & (yp<dim_y)
                xp = xp[ind]
                yp = yp[ind]
                
                # Create array of pixels in polygon 
                rr, cc = polygon(x, y, shape=(dim_x, dim_y))
                xp = np.append(xp, rr)
                yp = np.append(yp, cc)
                if yp.shape[0] == 0:
                    continue
                              
                layer[xp, yp] = 1 #TODO: Encode pixel class
                if len(yp) > 0:
                    any_polygons = True
        
            if not any_polygons:             
                layer = []
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return layer
              
     
    def load_lbl_targets(self, t, max_range=20000):
        raise NotImplementedError


    def load_lbl_targets_sensor(self, t, sensor_type, sensor_index, subsensor_index=None):
        raise NotImplementedError

     
    def load_lbl_layer(self, t, dim, scale, binary=False):
        raise NotImplementedError

    
    def load_lbl_layer_sensor(self, t, sensor_type, sensor_index, subsensor_index=None, binary=False):
        raise NotImplementedError

        
    def load_nav_targets(self, t, max_range=20000):
        raise NotImplementedError


    def load_nav_targets_sensor(self, t, sensor_type, sensor_index, subsensor_index=None):
        raise NotImplementedError
    
    
    def load_nav_layer(self, t, dim, scale, binary=False):
        raise NotImplementedError
    
    
    def load_nav_layer_sensor(self, t, sensor_type, sensor_index, subsensor_index=None, binary=False):
        raise NotImplementedError


    def load_ais_targets(self, t, max_range=20000):
        """
        Load AIS targets for a given time tag in (N,E) coordinate frame. 
        The position of Polarlys at time 't' is used as origin in the image (Seapath reference point).
        The argument 'max_range' is the maximum distance for targets to include.
        The method returns a dictionary of AIS targets with MMSI as key.
        The method also returns a dictionary of AIS polygons with MMSI as key.
        The method also returns a dictionary with Seapath data for given time tag.
        """
        targets = {}
        polygons = {}
        pva = {}
        try:
            pva = self.get_seapath_data(t)
            targets = self.get_ais_targets(t, pva['position'], max_range)
            polygons = self.get_target_polygons(targets)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return targets, polygons, pva

    
    def load_ais_targets_sensor(self, t, sensor_type, sensor_index, subsensor_index=None):
        """
        Load AIS targets for a given time tag and sensor. 
        The method returns a dictionary of AIS polygons with MMSI as key.
        """
        polygons_sensor = {}
        try:
            meta = self.get_metadata(t, sensor_type, sensor_index, subsensor_index)
            dim_x, dim_y = meta['image_dim'][0], meta['image_dim'][1]
            cfg_ais = self.sensor_config['AIS']
            
            if sensor_type == self.TYPE_CAMERA:
                targets = self.prepare_ais_targets(meta['targets_ais'], cfg_ais)
                polygons = self.get_target_polygons(targets)
                polygons_sensor = []
                for id in polygons.keys():
                    poly_sensor = self.transform_points_to_sensor(t, polygons[id], sensor_type, sensor_index, subsensor_index, invalidate_pixels=False)
                    outside = np.logical_or.reduce((poly_sensor[:,0] < 0, poly_sensor[:,0] >= dim_x, poly_sensor[:,1] < 0, poly_sensor[:,1] >= dim_y))
                    if not outside.all():
                        polygons_sensor.append(poly_sensor)
            else:
                targets_start = self.prepare_ais_targets(meta['targets_ais_start'], cfg_ais)
                targets_end = self.prepare_ais_targets(meta['targets_ais_end'], cfg_ais)
                polygons_start = self.get_target_polygons(targets_start)
                polygons_end = self.get_target_polygons(targets_end)
                polygons_sensor = []
                for id in polygons_start.keys():
                    if id in polygons_end.keys():
                        poly_sensor = self.transform_points_to_sensor(t, (polygons_start[id], polygons_end[id]), sensor_type, sensor_index, invalidate_pixels=False)
                        if (poly_sensor[:,0].max() - poly_sensor[:,0].min()) > dim_x/2:
                            # Handle polygons at bearing == 0 degrees
                            poly_sensor[poly_sensor[:,0] > dim_x/2,0] -= dim_x
                            polygons_sensor.append(poly_sensor.copy())
                            poly_sensor[:,0] += dim_x
                            polygons_sensor.append(poly_sensor)
                        else:
                            polygons_sensor.append(poly_sensor)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return polygons_sensor
    
    
    def load_ais_layer(self, t, dim, scale, binary=False):
        """
        Load AIS layer for a given time tag in (N,E) coordinate frame. 
        The position of Polarlys at time 't' is used as origin in the image (Seapath reference point).
        The argument 'dim' is the height and width of the returned image in pixels.
        The argument 'scale' is the number of meters pr. pixel in the returned image.
        The argument 'binary' decides if the output is a binary mask or grayscale according to classification.
        Origin is located in the center (dim/2,dim/2) of the chart.
        The method returns a numpy array of shape (dim,dim) of type np.uint8.
        The method also returns a dictionary with Seapath data for given time tag.
        """
        layer = []
        pva = {}
        try:
            targets, polygons, pva = self.load_ais_targets(t, dim*scale//2)
            polygons_ned = []
            for id in polygons.keys():
                pos = (pva['position'][0], pva['position'][1], 0)
                poly = self.transform_points_to_map(polygons[id], pos, scale)
                polygons_ned.append(np.stack([-poly[:,0]+dim/2, poly[:,1]+dim/2], axis=1))
                
            layer = self.draw_polygon_layer(polygons_ned, dim, dim)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return layer, pva
        
    
    def load_ais_layer_sensor(self, t, sensor_type, sensor_index, subsensor_index=None, binary=False):
        """
        Load AIS layer for a given time tag and sensor. 
        The argument 'binary' decides if the output is a binary mask or grayscale according to classification.
        The method returns a numpy array of shape (height,width) of type np.uint8.
        Height and width matches the dimension of the original sensor image.
        """
        layer = []
        try:
            meta = self.get_metadata(t, sensor_type, sensor_index, subsensor_index)
            dim_x, dim_y = meta['image_dim'][0], meta['image_dim'][1]
            polygons_sensor = self.load_ais_targets_sensor(t, sensor_type, sensor_index, subsensor_index)
            layer = self.draw_polygon_layer(polygons_sensor, dim_x, dim_y)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return layer


    def load_sky_polygon_sensor(self, t, sensor_type, sensor_index, subsensor_index=None):
        """
        Load sky polygon for a given time tag and sensor.
        The method returns a numpy array with polygon of the sky in image.
        """
        points = np.zeros((360, 3))
        try:
            meta = self.get_metadata(t, sensor_type, sensor_index, subsensor_index)
            cfg_sensor = self.get_sensor_config(sensor_type, sensor_index, subsensor_index)
            cfg_seapath = self.sensor_config['Seapath']
            if sensor_type == self.TYPE_CAMERA:

                # Get sensor height above sea level
                pos_sensor = self.get_sensor_pos(meta['own_vessel'], cfg_sensor['location'],
                                                 cfg_seapath['navref_height'])
                height_above_sea_level = pos_sensor[2]

                # Get distance to horizon (subtract 10 meters to ensure the points are not invalidated
                dist_to_horizon = np.sqrt(2 * EARTH_RADIUS * height_above_sea_level + height_above_sea_level ** 2)-10.0
                # Create circle with radius equal to range
                points[:, 0] = dist_to_horizon * np.cos(np.linspace(0, 2 * np.pi, points.shape[0]))
                points[:, 1] = dist_to_horizon * np.sin(np.linspace(0, 2 * np.pi, points.shape[0]))
                points = self.transform_points_to_camera(t, points, sensor_index, subsensor_index,
                                                         invalidate_pixels=True)
                # Add points on both sides and the top corners of picture
                points = np.array((points[points[:, 1].argsort()]))
                points = np.squeeze(points[np.nonzero(points[:, 1] > -1),:], axis=0)
                fit = np.polyfit(points[0:2,1],points[0:2,0],1)
                points = np.vstack(([int(fit[1]),0],points))
                points = np.vstack(([0, 0], points))
                fit = np.polyfit(points[-2:, 1], points[-2:, 0], 1)
                dim_y = meta['image_dim'][1]
                points = np.vstack((points, [int(fit[0] * dim_y + fit[1]), dim_y]))
                points = np.vstack((points, [0, dim_y]))
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')

        return points


    def load_sky_layer_sensor(self, t, sensor_type, sensor_index, subsensor_index=None):
        """
        Load sky layer for a given time tag and sensor. 
        The method returns a numpy array of shape (height,width) of type np.uint8.
        Height and width matches the dimension of the original sensor image.
        The returned values are 1 for pixels above horizon and 0 for all other pixels.
        """
        layer = []
        try:
            meta = self.get_metadata(t, sensor_type, sensor_index, subsensor_index)
            cfg_sensor = self.get_sensor_config(sensor_type, sensor_index, subsensor_index)
            cfg_seapath = self.sensor_config['Seapath']
            if sensor_type == self.TYPE_CAMERA:
                dim = 4000
                scale = 10
                
                # Get sensor height above sea level        
                pos_sensor = self.get_sensor_pos(meta['own_vessel'], cfg_sensor['location'], cfg_seapath['navref_height'])
                height_above_sea_level = pos_sensor[2]
                
                # Get distance to horizon
                dist_to_horizon = np.sqrt(2 * EARTH_RADIUS * height_above_sea_level + height_above_sea_level**2) 
        
                hor = np.zeros((dim,dim), dtype=np.uint8)
                x, y = circle(dim//2, dim//2, dist_to_horizon//scale, (dim,dim))
                hor[x,y] = 1
                layer = self.transform_image_to_sensor(t, sensor_type, sensor_index, hor, meta['own_vessel']['position'], subsensor_index, scale=scale)
                layer = (1-layer)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return layer
    
    
    def load_horizon_sensor(self, t, sensor_type, sensor_index, subsensor_index=None):
        raise NotImplementedError
        
        
    def transform_points(self, t, points, from_sensor_type, to_sensor_type, from_sensor_index, to_sensor_index, from_subsensor_index=None, to_subsensor_index=None):
        """
        Transform points in pixel coordinates (X,Y) from one sensor frame to another. Assumes that points are in the sea plane.
        The argument 'points' should be a numpy array of shape (n,2), where n is the number of points.
        The argument 'from_subsensor_index' is only required if 'from_sensor_type' is TYPE_CAMERA.
        The argument 'to_subsensor_index' is only required if 'to_sensor_type' is TYPE_CAMERA.
        The method returns a numpy array of shape (n,2), where n is the number of input points.
        If a point falls outside the image boundaries, the coordinates for this points are set to -1.
        """
        if from_sensor_type == self.TYPE_RADAR:
            points = (points, points.copy())
        points = self.transform_points_from_sensor(t, points, from_sensor_type, from_sensor_index, from_subsensor_index)
        valid = np.logical_or.reduce((points[:,0] != 0, points[:,1] != 0, points[:,2] != 0))
        if to_sensor_type == self.TYPE_RADAR:
            points = (points, points.copy())
        points = self.transform_points_to_sensor(t, points, to_sensor_type, to_sensor_index, to_subsensor_index)
        valid = np.logical_and.reduce((points[:,0] >= 0, points[:,1] >= 0, valid == True))
        points[valid==False,:] = -1
        return points
        
    
    def transform_points_from_map(self, points, origin, scale=1):
        """
        Transform numpy array of map pixel coordinates (X,Y,dwn) to points in degrees in (lat,lon,hgt). 
        Map is assumed to be oriented north up.
        The argument 'points' should be a numpy array of shape (n,3), where n is the number of points.
        The argument 'origin' is the position of the map origin in (lat,lon,hgt) as tuple or numpy array.
        The argument 'scale' is number of meters pr pixel used for transformation.
        The method returns a numpy array of shape (n,3), where n is the number of input points.
        """
        points[:,0] = origin[0] + (self.m2dlat(scale * points[:,0]) * 180 / np.pi)
        points[:,1] = origin[1] + (self.m2dlon(scale * points[:,1], origin[0] * np.pi / 180) * 180 / np.pi)
        points[:,2] = origin[2] - points[:,2]
        
        return points
    

    def transform_points_to_map(self, points, origin, scale=1):
        """
        Transform points in degrees in (lat,lon,hgt) to pixels coordinates (X,Y,dwn) in map image.
        Map is assumed to be oriented north up.
        The argument 'points' should be a numpy array of shape (n,3), where n is the number of points.
        The argument 'origin' is the position of the map origin in (lat,lon,hgt)
        The argument 'scale' is number of meters pr pixel used for transformation.
        The method returns a numpy array of shape (n,3), where n is the number of input points.
        """
        points[:,0] = (1/scale) * self.dlat2m((points[:,0] - origin[0]) * np.pi/180)
        points[:,1] = (1/scale) * self.dlon2m((points[:,1] - origin[1]) * np.pi/180, origin[0] * np.pi / 180)
        points[:,2] = origin[2] - points[:,2]
        
        return points
        
        
    def transform_points_from_sensor(self, t, points, from_sensor_type, from_sensor_index, from_subsensor_index=None):
        """
        Transform numpy array of sensor image pixel coordinates (X,Y) to points in degrees in (lat,lon,hgt). Assumes that points are in the sea plane.
        The argument 'points' should be a numpy array of shape (n,2), where n is the number of points.
        If 't_sensor_type' is TYPE_RADAR, the argument 'points' can also be tuple of (n,3) arrays, the
        first array valid at start of scan, and the second array valid at end of the scan. If not a tuple, 
        the points are assumed to be valid at end of scan.
        The method returns a numpy array of shape (n,3), where n is the number of input points.
        If a point is pointing above the horizon, the coordinates for this points are set to (0,0,0).
        """
        meta = self.get_metadata(t, from_sensor_type, from_sensor_index, from_subsensor_index)
        
        if from_sensor_type == self.TYPE_CAMERA:
            points = self.transform_points_from_camera(t, points, from_sensor_index, from_subsensor_index)
            pos = meta['own_vessel']['position'].copy()
            pos[2] = self.get_spt_height(meta['own_vessel']['attitude'], self.sensor_config['Seapath']['navref_height'])
            valid = np.logical_or.reduce((points[:,0] != 0, points[:,1] != 0, points[:,2] != 0))
            points = self.transform_points_from_map(points, pos)
        else:
            pos_start = meta['own_vessel_start']['position'].copy()
            pos_start[2] = self.get_spt_height(meta['own_vessel_start']['attitude'], self.sensor_config['Seapath']['navref_height'])
            pos_end = meta['own_vessel_end']['position'].copy()
            pos_end[2] = self.get_spt_height(meta['own_vessel_end']['attitude'], self.sensor_config['Seapath']['navref_height'])
            if isinstance(points, tuple):
                points_start, points_end = points
                frac_interpolation = points_end[:,0] / meta['image_dim'][0]
                points_start = self.transform_points_from_radar(t, points_start, from_sensor_index, start_of_scan=True)
                valid_start = np.logical_or.reduce((points_start[:,0] != 0, points_start[:,1] != 0, points_start[:,2] != 0))
                points_start = self.transform_points_from_map(points_start, pos_start)
                
                points_end = self.transform_points_from_radar(t, points_end, from_sensor_index, start_of_scan=False)
                valid = np.logical_or.reduce((points_end[:,0] != 0, points_end[:,1] != 0, points_end[:,2] != 0, valid_start==True))
                points_end = self.transform_points_from_map(points_end, pos_end)
                
                # Interpolate points to correct for duration of radar scan (about 2.5s)
                points = frac_interpolation.reshape(-1,1) * points_start + (1 - frac_interpolation.reshape(-1,1)) * points_end
            else:
                points = self.transform_points_from_radar(t, points, from_sensor_index, start_of_scan=False)
                valid = np.logical_or.reduce((points[:,0] != 0, points[:,1] != 0, points[:,2] != 0))
                points = self.transform_points_from_map(points, pos_end)
                print(points)
        
        points[valid==False,:] = 0
        
        return points
    

    def transform_points_to_sensor(self, t, points, to_sensor_type, to_sensor_index, to_subsensor_index=None, invalidate_pixels=True):
        """
        Transform points in degrees in (lat,lon,hgt) to pixels coordinates (X,Y) in sensor image.
        The argument 'points' should be a numpy array of shape (n,3), where n is the number of points.
        If 't_sensor_type' is TYPE_RADAR, the argument 'points' can also be tuple of (n,3) arrays, the
        first array valid at start of scan, and the second array valid at end of the scan. If not a tuple, 
        the points are assumed to be valid at end of scan.
        The method returns a numpy array of shape (n,2), where n is the number of input points.
        If a point falls outside the image boundaries, the coordinates for this points are set to -1.
        """
        meta = self.get_metadata(t, to_sensor_type, to_sensor_index, to_subsensor_index)
        
        if to_sensor_type == self.TYPE_CAMERA:
            pos = meta['own_vessel']['position'].copy()
            pos[2] = self.get_spt_height(meta['own_vessel']['attitude'], self.sensor_config['Seapath']['navref_height'])
            points = self.transform_points_to_map(points, pos)
            points = self.transform_points_to_camera(t, points, to_sensor_index, to_subsensor_index, invalidate_pixels=invalidate_pixels)
        else:
            pos_start = meta['own_vessel_start']['position'].copy()
            pos_start[2] = self.get_spt_height(meta['own_vessel_start']['attitude'], self.sensor_config['Seapath']['navref_height'])
            pos_end = meta['own_vessel_end']['position'].copy()
            pos_end[2] = self.get_spt_height(meta['own_vessel_end']['attitude'], self.sensor_config['Seapath']['navref_height'])
            if isinstance(points, tuple):
                dim_x = meta['image_dim'][0]
                points_start, points_end = points
                points_start = self.transform_points_to_map(points_start, pos_start)
                points_start = self.transform_points_to_radar(t, points_start, to_sensor_index, start_of_scan=True, invalidate_pixels=invalidate_pixels) 
                points_end = self.transform_points_to_map(points_end, pos_end)
                points_end = self.transform_points_to_radar(t, points_end, to_sensor_index, start_of_scan=False, invalidate_pixels=invalidate_pixels)
                
                valid = np.logical_and.reduce((points_start[:,0] >= 0, points_end[:,0] >= 0, (points_end[:,0] - points_start[:,0]) < dim_x/2))

                # Interpolate points to correct for duration of radar scan (about 2.5s)
                frac_interpolation = points_end[valid,0] / dim_x
                wrap_dn = (points_end[valid,0] - points_start[valid,0]) < -dim_x/2
                frac_interpolation[wrap_dn] = 0
                points = np.zeros(points_start.shape).astype(np.int16) - 1
                points[valid,:] = frac_interpolation.reshape(-1,1) * points_start[valid,:] + (1 - frac_interpolation.reshape(-1,1)) * points_end[valid,:] + 0.5
            else:  
                points = self.transform_points_to_map(points, pos_end)
                points = self.transform_points_to_radar(t, points, to_sensor_index, start_of_scan=False, invalidate_pixels=invalidate_pixels)
        
        return points
        
        
    def transform_points_from_camera(self, t, points, from_sensor_index, from_subsensor_index, ref_location=None, head_up=False):
        """
        Transform numpy array of camera image pixel coordinates (X,Y) to points in meters in (N,E,D) or heading frame. Assumes that points are in the sea plane.
        The argument 'points' should be a numpy array of shape (n,2), where n is the number of points.
        The argument 'ref_location' is the offset from the Seapath reference point to the reference.
        point to be used for the returned points in meters and vessel body coordinates (fwd,stb,dwn).
        If 'ref_location' is None, returned points are relative to Seapath reference point.
        The argument 'head_up' decides the frame of the transformed points. If True, the frame is heading frame, else it is (N,E,D).
        The method returns a numpy array of shape (n,3), where n is the number of input points.
        If a point is pointing above the horizon, the coordinates for this points are set to (0,0,0).
        """
        meta = self.get_metadata(t, self.TYPE_CAMERA, from_sensor_index, from_subsensor_index)
        cfg_sensor = self.get_sensor_config(self.TYPE_CAMERA, from_sensor_index, from_subsensor_index)
        
        # Get camera calibration
        camera_matrix = np.array(cfg_sensor['camera_matrix']).reshape((3,3))
        dist_coeffs = np.array(cfg_sensor['distortion_coefficients'])
        
        # Get rotation matrices
        R_ma = self.rot_matrix_from_euler(cfg_sensor['rotation'])
        att = meta['own_vessel']['attitude'].copy()
        if head_up:
            att[2] = 0
        R_att = self.rot_matrix_from_euler(att)       

        # Get translation vector
        T = np.array(cfg_sensor['location']).reshape(3,1)
        if ref_location is not None:
            T -= np.array(ref_location).reshape(3,1)
         
        # Get sensor point height above sea level given attitude
        pos_sensor = self.get_sensor_pos(meta['own_vessel'], T, self.sensor_config['Seapath']['navref_height'])
        height_above_sea_level = pos_sensor[2] 
        
        # Convert pixel coordinates to normalized coordinates by applying camera calibration
        yn, zn = self.camera_p2m(points[:,1], points[:,0], camera_matrix, dist_coeffs) 
        
        # Check diff in z for x==1m
        pn = np.zeros((points.shape[0], 3)) + 1
        pn[:,1] = yn
        pn[:,2] = zn
        pn_geog = R_att.dot(R_ma.dot(pn.transpose())).transpose()
        valid = pn_geog[:,2] >= 0
        
        p_transformed = np.zeros((points.shape[0],3)).astype(np.float64)
        
        # Calculate distance x to sea plane
        x = height_above_sea_level / pn_geog[valid,2]
        
        # Convert normalized coordinates to cartesian
        p = np.array((x, x*yn[valid], x*zn[valid])).transpose()
        
        # Rotate and translate to reference point
        p_geog = R_att.dot(R_ma.dot(p.transpose()) + T).transpose()
        
        # Compensate for reduced height caused by earth curvature
        dst = np.sqrt(p_geog[:,0]**2 + p_geog[:,1]**2)
        dhgt = np.sqrt(EARTH_RADIUS**2 + dst**2) - EARTH_RADIUS
        p_geog[:,2] -= dhgt
        
        # Invalidate points above horizon
        dist_to_horizon = np.sqrt(2 * EARTH_RADIUS * height_above_sea_level + height_above_sea_level**2) 
        p_geog[dst>dist_to_horizon,:] = 0
        
        p_transformed[valid,:] = p_geog
   
        return p_transformed
    
    
    def transform_points_from_radar(self, t, points, from_sensor_index, start_of_scan, ref_location=None, head_up=False):
        """
        Transform numpy array of radar image pixel coordinates (X,Y) to points in meters in (N,E,D) or heading frame. 
        Assumes that points are in the radar (R,P) plane.
        The argument 'points' should be a numpy array of shape (n,2), where n is the number of points.
        If the argument 'start_of_scan' is True, the transformation will be valid at start of scan.
        If the argument 'start_of_scan' is False, the transformation will be valid at end of scan.
        The argument 'ref_location' is the offset from the Seapath reference point to the reference 
        point to be used for the returned points in meters and vessel body coordinates (fwd,stb,dwn). 
        If 'ref_location' is None, returned points are relative to Seapath reference point.
        The argument 'head_up' decides the frame of the transformed points. If True, the frame is heading frame, else it is (N,E,D).
        The method returns a numpy array of shape (n,3), where n is the number of input points.
        If a point is invalidated, the coordinates for this points are set to (0,0,0).
        """
        meta = self.get_metadata(t, self.TYPE_RADAR, from_sensor_index)
        range_filters = meta['radar_setup']['range_filters']
        dim_x = meta['image_dim'][0]    
        
        cfg_sensor = self.get_sensor_config(self.TYPE_RADAR, from_sensor_index)
         
        # Get rotation matrices
        R_ma = self.rot_matrix_from_euler(cfg_sensor['rotation'])
        if start_of_scan:
            pva = meta['own_vessel_start']
        else:
            pva = meta['own_vessel_end']
        att = pva['attitude'].copy()
        att[0] = att[1] = 0
        if head_up:
            att[2] = 0
        R_att = self.rot_matrix_from_euler(att)        
        
        # Get translation vector
        T = np.array(cfg_sensor['location']).reshape(3,1)
        if ref_location is not None:
            T -= np.array(ref_location).reshape(3,1)
        
        rng = self.radar_p2m(points[:,1], range_filters, cfg_sensor['m_per_sample'])
        brg = 360 * (1 - points[:,0] / dim_x)
        
        x = rng * np.cos(brg * math.pi / 180)
        y = rng * np.sin(brg * math.pi / 180)
        z = np.zeros((points.shape[0], 1))
        p = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)
        p_transformed = R_att.dot(R_ma.dot(p.transpose()) + T).transpose()
        p_transformed[:,2] = self.get_spt_height(pva['attitude'], self.sensor_config['Seapath']['navref_height'])
        p_transformed[rng<0,:] = 0
        
        return p_transformed
    
    
    def transform_points_to_camera(self, t, points, to_sensor_index, to_subsensor_index, ref_location=None, head_up=False, invalidate_pixels=True):
        """
        Transform points in meters in (N,E,D) or heading frame to pixels coordinates (X,Y) in camera image.
        The argument 'points' should be a numpy array of shape (n,3), where n is the number of points.
        The argument 'ref_location' is the offset from the Seapath reference point to the reference 
        point used for the input points in meters and vessel body coordinates (fwd,stb,dwn). If None, the offset is 
        assumed to be (0,0,0).
        The argument 'head_up' decides the coordinate frame of the input points. If True, the frame is heading frame, else it is (N,E,D).
        The method returns a numpy array of shape (n,2), where n is the number of input points.
        If a point falls outside the image boundaries, the coordinates for this points are set to -1.
        """   
        meta = self.get_metadata(t, self.TYPE_CAMERA, to_sensor_index, to_subsensor_index)
        dim_x, dim_y = meta['image_dim'][0], meta['image_dim'][1]
        
        cfg_sensor = self.get_sensor_config(self.TYPE_CAMERA, to_sensor_index, to_subsensor_index)  
            
        # Get rotation matrices
        R_ma = self.rot_matrix_from_euler(cfg_sensor['rotation']).transpose()
        att = meta['own_vessel']['attitude'].copy()
        if head_up:
            att[2] = 0
        R_att = self.rot_matrix_from_euler(att).transpose()
        
        # Get translation vector
        T = np.array(cfg_sensor['location']).reshape(3,1)
        if ref_location is not None:
            T -= np.array(ref_location).reshape(3,1)
        
        # Get camera calibration
        camera_matrix = np.array(cfg_sensor['camera_matrix']).reshape((3,3))
        dist_coeffs = np.array(cfg_sensor['distortion_coefficients'])
        
        # Compensate for increased height caused by earth curvature
        dhgt = np.sqrt(EARTH_RADIUS**2 + points[:,0]**2 + points[:,1]**2) - EARTH_RADIUS
        points[:,2] += dhgt
              
        # Translate and rotate target pos vector to sensor frame
        p_vessel = R_att.dot(points.transpose()).transpose()
        p_sensor = R_ma.dot(p_vessel.transpose() - T).transpose()
        
        # Convert to pixels by applying camera calibration
        y, x = self.camera_m2p(p_sensor[:,1], p_sensor[:,2], p_sensor[:,0], camera_matrix, dist_coeffs)

        p_transformed = np.zeros((points.shape[0], 2)).astype(np.float64) - 1
        if invalidate_pixels:
            # Get sensor point height above sea level given attitude
            pos_sensor = self.get_sensor_pos(meta['own_vessel'], T, self.sensor_config['Seapath']['navref_height'])
            height_above_sea_level = pos_sensor[2]
            # Invalidate points above horizon and ouside image boundaries
            dist_to_horizon = np.sqrt(2 * EARTH_RADIUS * height_above_sea_level + height_above_sea_level**2)
            rng = np.sqrt(p_sensor[:,0]**2 + p_sensor[:,1]**2 + p_sensor[:,2]**2)
            valid = np.logical_and.reduce((p_sensor[:,0] > 0, rng < dist_to_horizon, x >= -0.5, x < (dim_x-0.5), y >= -0.5, y < (dim_y-0.5)))
            p_transformed[valid,0] = x[valid]
            p_transformed[valid,1] = y[valid]
        else:
            p_transformed[:,0] = x
            p_transformed[:,1] = y
                           
        return p_transformed
    
    
    def transform_points_to_radar(self, t, points, to_sensor_index, start_of_scan, ref_location=None, head_up=False, invalidate_pixels=True):
        """
        Transform points in meters in (N,E,D) or heading frame to pixels coordinates (X,Y) in radar image.
        Assumes that points are relative to Seapath reference point.
        The argument 'points' should be a numpy array of shape (n,3), where n is the number of points.
        If the argument 'start_of_scan' is True, the transformation will be valid at start of scan.
        If the argument 'start_of_scan' is False, the transformation will be valid at end of scan.
        The argument 'ref_location' is the offset from the Seapath reference point to the reference 
        point used for the input points in meters and vessel body coordinates (fwd,stb,dwn). If None, the offset is 
        assumed to be (0,0,0).
        The argument 'head_up' decides the coordinate frame of the input points. If True, the frame is heading frame, else it is (N,E,D).
        The method returns a numpy array of type np.int16 and shape (n,2), where n is the number of input points.
        If a point falls outside the image boundaries, the coordinates for this points are set to -1.
        """
        meta = self.get_metadata(t, self.TYPE_RADAR, to_sensor_index)
        range_filters = meta['radar_setup']['range_filters']
        dim_x, dim_y = meta['image_dim'][0], meta['image_dim'][1]
        
        cfg_sensor = self.get_sensor_config(self.TYPE_RADAR, to_sensor_index)  
                    
        # Get rotation matrices
        R_ma = self.rot_matrix_from_euler(cfg_sensor['rotation']).transpose()
        if start_of_scan:
            pva = meta['own_vessel_start']
        else:
            pva = meta['own_vessel_end']
        att = pva['attitude'].copy()
        att[0] = att[1] = 0
        if head_up:
            att[2] = 0
        R_att = self.rot_matrix_from_euler(att).transpose()       
        
        # Get translation vector
        T = np.array(cfg_sensor['location']).reshape(3,1)
        if ref_location is not None:
            T -= np.array(ref_location).reshape(3,1)
            
        # Translate and rotate target pos vector to sensor frame
        points[:,2] = 0
        p_vessel = R_att.dot(points.transpose()).transpose()
        p_sensor = R_ma.dot(p_vessel.transpose() - T).transpose()
        
        # Calc bearing and range
        rng = np.sqrt(p_sensor[:,0]**2 + p_sensor[:,1]**2)
        brg = np.arctan2(p_sensor[:,1], p_sensor[:,0]) * 180 / math.pi
           
        p_transformed = np.zeros((points.shape[0], 2)).astype(np.float64)
        p_transformed[:,0] = dim_x - dim_x * (brg / 360)
        p_transformed[p_transformed[:,0] < 0] += dim_x
        p_transformed[p_transformed[:,0] >= dim_x] -= dim_x
        p_transformed[:,1] = self.radar_m2p(rng, range_filters, cfg_sensor['m_per_sample']) - 0.5
        
        if invalidate_pixels:
            p_transformed[p_transformed[:,1] >= (dim_y - 0.5),:] = -1
            p_transformed[p_transformed[:,1] < 0,:] = -1
                     
        return p_transformed
        
    
    def transform_image(self, t, from_sensor_type, to_sensor_type, from_sensor_index, to_sensor_index, image=[], from_subsensor_index=None, to_subsensor_index=None):
        """
        Transform image from one sensor frame to another. Assumes that all points are in the sea plane.
        The argument 't' is a datetime object with second resolution for the image transformation.
        The argument 'from_sensor_type' can be either TYPE_CAMERA or TYPE_RADAR.
        The argument 'to_sensor_type' can be either TYPE_CAMERA or TYPE_RADAR.
        The argument 'from_sensor_index' is the index of the sensor to transform from.
        The argument 'to_sensor_index' is the index of the sensor to transform to.
        The argument 'image' is the image to transform and should be a numpy array of shape (height, width, n), where n is the number of color channels.
        Height and width of 'image' has to match original sensor image. If 'image' is empty, the original sensor image is used as input.
        The argument 'from_subsensor_index' is only required if 'from_sensor_type' is TYPE_CAMERA.
        The argument 'to_subsensor_index' is only required if 'to_sensor_type' is TYPE_CAMERA.
        The method returns a numpy array of shape (height,width,n), where n is the number of color channels in input image.
        Height and width matches the dimension of the original sensor image.
        If a pixel value is undetermined, the color channels are set to 0.
        """
        image_return = []
        try:
            meta_src = self.get_metadata(t, from_sensor_type, from_sensor_index, from_subsensor_index)
            meta_dst = self.get_metadata(t, to_sensor_type, to_sensor_index, to_subsensor_index)
            dim_x_src, dim_y_src = meta_src['image_dim'][0], meta_src['image_dim'][1]
            dim_x_dst, dim_y_dst = meta_dst['image_dim'][0], meta_dst['image_dim'][1]
            
            if image == []:
                image = self.load_image(t, from_sensor_type, from_sensor_index, from_subsensor_index)
                if image == []:
                    return []
            else:
                if dim_x_src != image.shape[0] or dim_y_src != image.shape[1]:
                    print('transform_image: Image of dimension ({}x{}) is required, but got dimension ({}x{})'.format(
                            dim_x_src, dim_y_src, image.shape[0], image.shape[1]))
                    return []
                      
            x_dest, y_dest = np.meshgrid(np.arange(dim_x_dst), np.arange(dim_y_dst))
            p = np.stack([x_dest.reshape(-1), y_dest.reshape(-1), np.zeros(dim_x_dst * dim_y_dst)], axis=1)
            px = self.transform_points(t, p, to_sensor_type, from_sensor_type, to_sensor_index, from_sensor_index, to_subsensor_index, from_subsensor_index)
            px = (px+0.5).astype(np.int16)
            valid = np.where(px[:,0] >= 0)
            if image.ndim == 3:
                image_transformed = np.zeros((dim_x_dst * dim_y_dst, image.shape[2])).astype(np.uint8)
                image_transformed[valid,:] = image[px[valid,0], px[valid,1], :]
                image_transformed = np.reshape(image_transformed,(dim_y_dst, dim_x_dst, image.shape[2]))
            else:
                image_transformed = np.zeros((dim_x_dst * dim_y_dst)).astype(np.uint8)
                image_transformed[valid] = image[px[valid,0], px[valid,1]]
                image_transformed = np.reshape(image_transformed,(dim_y_dst, dim_x_dst))
                
            image_return = image_transformed.swapaxes(0,1)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')

        return image_return
    
    
    def transform_image_from_sensor(self, t, sensor_type, sensor_index, origin, subsensor_index=None, dim=2000, scale=1, image=[]):
        """
        Transform a sensor image to an image in geographic frame with Seapath reference point as origin.
        Origin is located in the center (dim/2, dim/2) of the transformed image.
        Assumes that points in camera image are in the sea plane.
        The argument 't' is a datetime object with second resolution for the image to transform.
        The argument 'sensor_type' can be either TYPE_CAMERA or TYPE_RADAR.
        The argument 'sensor_index' is the index of the sensor to transform.
        The argument 'origin' is the origin of the tranformed image in degrees (lat,lon) as a tuple or numpy array.
        The argument 'subsensor_index' is the sub index of the sensor to transform. Only used if 'sensor_type' is TYPE_CAMERA.
        The argument 'dim' is the height and width of the transformed image in pixels.
        The argument 'scale' is the number of meters pr. pixel in the transformed image.
        The argument 'image' is the image to transform and should be a numpy array of shape (height, width, n), where n is the number of color channels.
        Height and width of 'image' has to match original sensor image. If 'image' is empty, the original sensor image is used as input.
        The method returns a numpy array of shape (dim,dim,n), where n is the number of color channels in input image.
        If a point is outside sensor field of view, the color channels are set to 0.
        """
        image_return = []
        try:
            meta = self.get_metadata(t, sensor_type, sensor_index, subsensor_index)
            dim_x, dim_y = meta['image_dim'][0], meta['image_dim'][1]
            
            if image == []:
                image = self.load_image(t, sensor_type, sensor_index, subsensor_index)
                if image == []:
                    return []
            else:
                if dim_x != image.shape[0] or dim_y != image.shape[1]:
                    print('transform_image_from_sensor: Image of dimension ({}x{}) is required, but got dimension ({}x{})'.format(
                            dim_x, dim_y, image.shape[0], image.shape[1]))
                    return []
            
            radius = dim*scale/2
            x_dest, y_dest = np.meshgrid(np.linspace(radius, -radius, dim), np.linspace(-radius, radius, dim))
            p = np.stack([x_dest.reshape(-1), y_dest.reshape(-1), np.zeros(dim*dim)], axis=1)
            origin = (origin[0], origin[1], 0)
            p = self.transform_points_from_map(p, origin)
            if sensor_type==self.TYPE_RADAR:
                p = (p, p.copy())
            px = self.transform_points_to_sensor(t, p, sensor_type, sensor_index, subsensor_index)  
            px = (px+0.5).astype(np.int16)             
            px_outside = np.where(px[:,0] == -1)
            
            if image.ndim == 3:
                image_transformed = image[px[:,0], px[:,1], :]
                image_transformed[px_outside,:] = 0
                image_transformed = np.reshape(image_transformed,(dim, dim, image.shape[2]))
            else:
                image_transformed = image[px[:,0], px[:,1]]
                image_transformed[px_outside] = 0
                image_transformed = np.reshape(image_transformed,(dim, dim))
                    
            image_return = image_transformed.swapaxes(0,1)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')

        return image_return
    
    
    def transform_image_to_sensor(self, t, sensor_type, sensor_index, image, origin, subsensor_index=None, scale=1):
        """
        Transform a geographic image with Seapath reference point as origin to a sensor image.
        The origin in the input image is assumed to be located in the center (dim/2,dim/2) of the transformed image.
        The argument 't' is a datetime object with second resolution for the transformed image.
        The argument 'sensor_type' can be either TYPE_CAMERA or TYPE_RADAR.
        The argument 'sensor_index' is the index of the sensor to transform to.
        The argument 'image' is a numpy array of shape (dim,dim,n), where n is the number of color channels.
        The argument 'origin' is the origin of the input image in degrees (lat,lon) as a tuple or numpy array.
        The argument 'subsensor_index' is the sub index of the sensor to transform to. Only used if 'sensor_type' is TYPE_CAMERA.
        The argument 'scale' is the number of meters pr. pixel in the input image.
        The method returns a numpy array of shape (height,width,n), where n is the number of color channels in input image.
        Height and width matches the dimension of the original sensor image.
        If a pixel value is undetermined, the color channels are set to 0.
        """
        image_return = []
        try:
            meta = self.get_metadata(t, sensor_type, sensor_index, subsensor_index) 
            dim_x, dim_y = meta['image_dim'][0], meta['image_dim'][1]
            
            x_dest, y_dest = np.meshgrid(np.arange(dim_x), np.arange(dim_y))
            p = np.stack([x_dest.reshape(-1), y_dest.reshape(-1)], axis=1)
            if sensor_type==self.TYPE_RADAR:
                p = (p, p.copy())
            p = self.transform_points_from_sensor(t, p, sensor_type, sensor_index, subsensor_index)
            origin = (origin[0], origin[1], 0)
            px = self.transform_points_to_map(p, origin)
            px_outside = np.logical_and.reduce((px[:,0] == 0, px[:,1] == 0, px[:,2] == 0))  
            px = px / scale
            px[:,0] = -px[:,0] + image.shape[0]/2
            px[:,1] += image.shape[1]/2 
            px = px.astype(np.int16)
    
            valid = np.logical_and.reduce((px[:,0] >= 0, px[:,0] < image.shape[0], px[:,1] >= 0, px[:,1] < image.shape[1], px_outside == False))
            if image.ndim == 3:
                image_transformed = np.zeros((dim_x * dim_y, image.shape[2])).astype(np.uint8)
                image_transformed[valid,:] = image[px[valid,0], px[valid,1], :]
                image_transformed = np.reshape(image_transformed,(dim_y, dim_x, image.shape[2]))
            else:
                image_transformed = np.zeros((dim_x * dim_y)).astype(np.uint8)
                image_transformed[valid] = image[px[valid,0], px[valid,1]]
                image_transformed = np.reshape(image_transformed,(dim_y, dim_x))
                    
            image_return = image_transformed.swapaxes(0,1)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')

        return image_return
        
    
    def get_metadata(self, t, sensor_type, sensor_index, subsensor_index=None):
        file_meta = self.get_filename_sec(t, self.get_sensor_folder(sensor_type, sensor_index, subsensor_index), 'json')
        if not file_meta or not os.path.isfile(file_meta):
            return {}
        
        meta = {}
        try:
            with open(file_meta, 'r') as f:
                meta = json.load(f)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return meta
            

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
        a = (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        xp = x * a
        yp = y * a
    
        # Correct tangential distortion 
        xp = xp + (2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x))
        yp = yp + (p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y)
    
        # Ideal coordinates => actual coordinates
        xp = xp * fx + ux
        yp = yp * fy + uy
          
        return xp, yp
    
    
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
        a = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xmt = xm / a
        ymt = ym / a
        r2 = xmt**2 + ymt**2
        a = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xm = xm / a
        ym = ym / a
        
        return xm, ym
    
    
    def radar_m2p(self, m, range_filters, m_per_sample):
        """
        Function for converting radar range in meters to pixels according to range filters.
        """
        m = np.array(m)
        p = np.zeros(len(m)).astype(np.float64) - 1
        num_f = len(range_filters)
        
        dst_acc = 0
        px_acc = 0
        for j in range(num_f):
            if range_filters[j][0] == 0:
                break
            dst_acc_prev = dst_acc
            dst_acc += range_filters[j][1] * m_per_sample * range_filters[j][0]
            ind = np.logical_and.reduce((m > dst_acc_prev, m <= dst_acc))
            p[ind] = px_acc + (m[ind] - dst_acc_prev) / (m_per_sample * range_filters[j][0])
            px_acc += range_filters[j][1]
        
        return p
  
    
    def radar_p2m(self, px, range_filters, m_per_sample):
        """
        Function for converting radar range in pixels to meters according to range filters.
        """
        px = np.array(px + 0.5)
        m = np.zeros(len(px)).astype(np.float64) - 1
        num_f = len(range_filters)
        
        dst_acc = 0
        px_acc = 0
        for j in range(num_f):
            if range_filters[j][0] == 0:
                break
            px_acc_prev = px_acc
            px_acc += range_filters[j][1]
            ind = np.logical_and.reduce((px > px_acc_prev, px <= px_acc))
            m[ind] = dst_acc + (px[ind] - px_acc_prev) * range_filters[j][0] * m_per_sample
            dst_acc += range_filters[j][1] * m_per_sample * range_filters[j][0]
           
        return m
        
    
    def prepare_ais_targets(self, targets, cfg_ais):
        """
        Position and heading is extrapolated according to velocity and age.
        Targets with age greater than limit are removed.
        """
        targets_prep = {}
        for mmsi,target in targets.items():
       
            # Check if moored, anchored or aground
            moving = True
            if 'nav_status' in target.keys():
                if target['nav_status'] == 1 or target['nav_status'] == 5 or target['nav_status'] == 6:
                    moving = False
                    
            # Check if age is within limits
            if moving:
                if abs(target['age']) > cfg_ais['timeout_moving']:
                    continue
            else:
                if abs(target['age']) > cfg_ais['timeout_static']:
                    continue
            
            # Convert decimal degrees to radians 
            lat, lon = map(math.radians, [target['position'][0], target['position'][1]])   
        
            # Copy target before modifying it
            target = target.copy()
            
            # Interpolate to current time
            if 'sog' in target.keys():
                cog = target['cog'] * np.pi / 180
                x_vel = target['sog'] * math.cos(cog)
                y_vel = target['sog'] * math.sin(cog)
                lat += self.m2dlat(target['age'] * x_vel)
                lon += self.m2dlon(target['age'] * y_vel, lat)
                target['position'] = (lat*180/np.pi, lon*180/np.pi)
                
            if 'rot' in target.keys() and 'true_heading' in target.keys():
                target['true_heading'] += target['age'] * target['rot'] / 60
                if target['true_heading'] < 0:
                    target['true_heading'] += 360
                elif target['true_heading'] >= 360:
                    target['true_heading'] -= 360

            targets_prep[mmsi] = target
            
        return targets_prep
                   
    
    def get_target_polygons(self, targets):
        """
        Creates a list of polygons from a target list.
        The method returns a dictionary (key=mmsi) of AIS target polygons in (lat,lon,hgt) coordinate frame.
        """
        polygons = {}
        for mmsi,target in targets.items():
            
            # Convert decimal degrees to radians 
            lat, lon = map(math.radians, [target['position'][0], target['position'][1]])   
                
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
                    
                # Rotate ship polygon
                x_poly = x_poly0 * np.cos(heading_target) - y_poly0 * np.sin(heading_target)
                y_poly = x_poly0 * np.sin(heading_target) + y_poly0 * np.cos(heading_target)
            else:
                # Draw an octagon
                x_poly = a*np.cos(np.array([np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi, 0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi]))
                y_poly = a*np.sin(np.array([np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi, 0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi]))
                
            # Add relative position
            poly = np.zeros((x_poly.shape[0], 3))
            poly[:,0] = self.m2dlat(x_poly) * 180/np.pi + target['position'][0]
            poly[:,1] = self.m2dlon(y_poly, lat) * 180/np.pi + target['position'][1]
            poly[:,2] = 0
            
            polygons[mmsi] = poly
        
        return polygons
    
    
    @staticmethod
    def dlat2m(dlat):
        """
        Function for converting difference in latitude (radians) to meters.
        """
        return EARTH_RADIUS * dlat
     
        
    @staticmethod   
    def m2dlat(m):
        """
        Function for converting meters to difference in latitude (radians).
        """
        return m / EARTH_RADIUS
    
    
    @staticmethod
    def dlon2m(dlon, lat):
        """
        Function for converting difference in longitude (radians) to meters.
        """
        return EARTH_RADIUS * dlon * math.cos(lat)
    
    
    @staticmethod
    def m2dlon(m, lat):
        """
        Function for converting meters to difference in longitude (radians).
        """
        return m / (EARTH_RADIUS * math.cos(lat))
    
    
    @staticmethod
    def rot_matrix_from_euler(att):
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
    
    
    def get_spt_height(self, att, spt_height):
        """
        Method for calculating Seapath reference point height above sea level.
        The argument 'att' is current attitude as numpy array or tuple.
        The argument 'spt_height' is the height of Seapath reference point in vessel frame.
        """
        # Get rotation matrix
        R_att = self.rot_matrix_from_euler(att)
        
        # Get current height of Seapath reference point        
        ref_hgt = R_att.dot(np.array((0,0,spt_height)).reshape((3,1)))[2]
        
        return ref_hgt
        
    
    def get_sensor_pos(self, pva, offset, spt_height):
        """
        Method for calculating sensor position in (lat,lon,hgt) geographic frame.
        The argument 'pva' is a Seapath data dictionary with keys 'position', 'heave' and 'attitude'.
        The argument 'offset' is the sensor location relative to Seapath reference point in vessel.
        The argument 'spt_height' is the height of Seapath reference point in vessel frame.
        """
        # Get rotation matrix
        R_att = self.rot_matrix_from_euler(pva['attitude'])

        # Get current height of Seapath reference point        
        ref_hgt = R_att.dot(np.array((0,0,spt_height)).reshape((3,1)))[2]
            
        # Translate position to sensor location
        loc_rotated = R_att.dot(np.array(offset).reshape((3,1)))
        lat = pva['position'][0] + self.m2dlat(loc_rotated[0]) * 180 / np.pi
        lon = pva['position'][1] + self.m2dlon(loc_rotated[1], pva['position'][0] * np.pi / 180) * 180 / np.pi
        hgt = ref_hgt - loc_rotated[2]
        if 'heave' in pva.keys():
            hgt -= pva['heave']
            
        return np.array((lat[0], lon[0], hgt[0]))
    
    
    def average_angle(self, list_of_angles,unit='RAD',bNonNegative=False):
        """ Calculates the average of angles in list
    
        Args:
            list_of_angles: 1xN array with N angle values
            unit: RAD (default) or DEG
            bNonNegative: Angle range
                False (default) Range -pi:pi (or -180:180)
                True            Range 0:2*pi (or 0:360)
        Returns:
            Average value of angles in arr
        """
    
        lim=math.pi
        if(not('RAD' in unit)):
            lim=180.0
    
        a=np.array(list_of_angles).astype(np.float64)
        
        # map to -pi:pi / -180:180
        a=((-a+lim)%(2*lim)-lim)*-1
        
        # convert to radians  
        a*=math.pi/lim
        
        # make a new list of vectors
        vectors= [cmath.rect(1, angle) for angle in a]
        vector_sum= sum(vectors)
        mean_angle = cmath.phase(vector_sum)
            
        # convert to desired format
        mean_angle*=lim/math.pi
        if(bNonNegative):
            if(mean_angle<0):
                mean_angle+=2*lim
                
        return(mean_angle)