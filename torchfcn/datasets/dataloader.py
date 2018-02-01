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
import time
import xml.etree.ElementTree as et

gpu = False
try:
    import torch
    import torch.nn.functional as F
    from torch.autograd import Variable
    if torch.cuda.is_available():
         gpu = True
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


class WaterLevelReader:
    BASE_DIR = "/nas0"
    WATERLEV_FNAME_FORMAT = os.path.join(BASE_DIR, "{:%Y-%m-%d}", "waterlevels.json")
    SEAPATH_FNAME_FORMAT = os.path.join(BASE_DIR, "{:%Y-%m-%d}", "{:%Y-%m-%d-%H}", "Seapath", "{:%Y-%m-%d-%H_%M}.bin")

    def __init__(self):
        self.cache = {}

    def waterlev_fname(self, t):
        return self.WATERLEV_FNAME_FORMAT.format(t)

    def seapath_fname(self, t):
        return self.SEAPATH_FNAME_FORMAT.format(t, t, t)

    def dump_file(self, d, t):
        with open(self.waterlev_fname(t), "w+") as f:
            json.dump(d, f)

    def load_file(self, t):
        with open(self.waterlev_fname(t), "r") as f:
            return json.load(f)

    def retrieve_data_for_time(self, t):
        """
        Retrieve water level for time t from sehavniva.no, using the actual position
        at that time from SeapathReader. Returns level value or None on error
        """
        try:
            sr = SeapathReader(self.seapath_fname(t))
            pos = sr.get_posvelatt(t)['position']
        except FileNotFoundError:
            # missing SeaPath file for this timestamp, use the one before if present
            try:
                u = t - datetime.timedelta(seconds=1)
                sr = SeapathReader(self.seapath_fname(u))
                pos = sr.get_posvelatt(u)['position']
            except FileNotFoundError:
                return None

        # minimum query time period supported by sehavniva.no is 1 hour
        end = t + datetime.timedelta(hours=1)
        url = "http://api.sehavniva.no/tideapi.php?lat={}&lon={}&fromtime={}&totime={}&datatype=obs&refcode=cd&place=&file=&lang=nn&interval=10&dst=0&tzone=0&tide_request=locationdata".format(
            pos[0], pos[1], t.isoformat(), end.isoformat())
        print(url)
        res = requests.get(url)

        # return the first water level found in the retrieved xml document
        root = et.fromstring(res.text)
        locationdata = root.findall("locationdata")
        data = locationdata[0].findall("data")
        waterlevel = data[0].findall("waterlevel")
        level = waterlevel[0].attrib["value"]
        return float(level)

    def retrieve_data_for_day(self, t, backoff_delay=2.0):
        """
        Retrieve water levels from disk. Updates file from sehavniva.no if values are missing
        (but not if they are None in the file).
        """
        try:
            levels = self.load_file(t)
            self.cache.update(levels)
        except IOError:
            levels = {}

        updated = False
        day = t.replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(49):
            d = day + i * datetime.timedelta(minutes=30)
            if not d.isoformat() in self.cache:
                updated = True
                level = self.retrieve_data_for_time(d)
                levels.update({d.isoformat(): level})
                print("Updated water level {} {}".format(d, level))
                if level is not None and backoff_delay is not None:
                    time.sleep(backoff_delay)
            else:
                print("Cached water level {} {}".format(d, self.cache[d.isoformat()]))

        if updated:
            self.dump_file(levels, t)
            self.cache.update(levels)

    @staticmethod
    def time_range(t):
        """
        Return the nearest half-hour times before and after t. If t is already on a half-hour,
        the times returned will be equal.
        """
        if t.minute in (0, 30) and t.second == 0 :
            return t.replace(microsecond=0), t.replace(microsecond=0)
        if t.minute >= 30:
            return t.replace(microsecond=0, second=0, minute=30), t.replace(microsecond=0, second=0, minute=0) + datetime.timedelta(hours=1)
        return t.replace(microsecond=0, second=0, minute=0), t.replace(microsecond=0, second=0, minute=30)

    def get_waterlevel(self, t):
        """
        Get interpolated water level for datetime t. Raises FileNotFound if no data on
        disk, or KeyError if values are missing from the data set.
        """
        start_time, end_time = self.time_range(t)
        try:
            # first try the cache
            start_level = self.cache[start_time.isoformat()]
            end_level = self.cache[end_time.isoformat()]
        except KeyError:
            # nope, not in cache; try to update cache from disk
            levels = self.load_file(t)
            self.cache.update(levels)
            start_level = self.cache[start_time.isoformat()]
            end_level = self.cache[end_time.isoformat()]

        if start_level is None or end_level is None:
            return None

        f = (t - start_time).total_seconds() / 1800.0
        return start_level + f * (end_level - start_level)


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
        
    
    def get_waterlevel(self, t):
        """
        Read waterlevel above chart zero level for the position of Polarlys at a specific time stamp.
        """
        raise NotImplementedError
        
        
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
            chart_large_sensor = self.transform_image_to_sensor(t, sensor_type, sensor_index, chart_large, pos, subsensor_index, scale, use_gpu=True)

            scale = 4
            url = 'http://navdemo:9669/WmsServer?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image/png&LAYERS=CHART&styles={}&HEIGHT={}&WIDTH={}&REFLATLON={},{},{},{},{},{}'.format(style, dim, dim, pos[0]*np.pi/180, pos[1]*np.pi/180, dim//2, dim//2, scale, 0)
            r = requests.get(url)
            with Image.open(BytesIO(r.content)) as im_pil:
                chart_medium = np.array(im_pil, dtype=np.uint8).reshape(dim,dim)
            if binary:
                chart_medium = (chart_medium[:,:] != 22).astype(np.uint8)
            chart_medium_sensor = self.transform_image_to_sensor(t, sensor_type, sensor_index, np.stack([chart_medium, np.zeros((dim,dim), dtype=np.uint8) + 1], axis=2), pos, subsensor_index, scale, use_gpu=True)

            scale = 0.5
            url = 'http://navdemo:9669/WmsServer?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image/png&LAYERS=CHART&styles={}&HEIGHT={}&WIDTH={}&REFLATLON={},{},{},{},{},{}'.format(style, dim, dim, pos[0]*np.pi/180, pos[1]*np.pi/180, dim//2, dim//2, scale, 0)
            r = requests.get(url)
            with Image.open(BytesIO(r.content)) as im_pil:
                chart_small = np.array(im_pil, dtype=np.uint8).reshape(dim,dim)
            if binary:
                chart_small = (chart_small[:,:] != 22).astype(np.uint8)
            chart_small_sensor = self.transform_image_to_sensor(t, sensor_type, sensor_index, np.stack([chart_small, np.zeros((dim,dim), dtype=np.uint8) + 1], axis=2), pos, subsensor_index, scale, use_gpu=True)
            
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
                poly = self.transform_points_to_ned(polygons[id], pos, scale)
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
                layer = self.transform_image_to_sensor(t, sensor_type, sensor_index, hor, meta['own_vessel']['position'], subsensor_index, scale=scale, use_gpu=True)
                layer = (1-layer)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            
        return layer
        
        
    def transform_points(self, t, points, from_sensor_type, to_sensor_type, from_sensor_index, to_sensor_index, from_subsensor_index=None, to_subsensor_index=None):
        """
        Transform points in pixel coordinates (X,Y) from one sensor frame to another. Assumes that points are in the sea plane.
        The argument 'points' should be an array of shape (n,2), where n is the number of points.
        The argument 'points' can be a numpy array, torch.Tensor or torch.cuda.Tensor. If the type is numpy,
        transformed points are returned as np.float64, else the points are transformed using the GPU and 
        returned as torch.cuda.DoubleTensor.
        The argument 'from_subsensor_index' is only required if 'from_sensor_type' is TYPE_CAMERA.
        The argument 'to_subsensor_index' is only required if 'to_sensor_type' is TYPE_CAMERA.
        The method returns an array of shape (n,2), where n is the number of input points.
        If a point falls outside the image boundaries, the coordinates for this points are set to -1.
        """
        use_gpu = gpu and torch.is_tensor(points)
        points = self.transfer_gpu(points, use_gpu)
        
        if from_sensor_type == self.TYPE_RADAR:
            points = (points, points.clone()) if use_gpu else (points, points.copy())
        points = self.transform_points_from_sensor(t, points, from_sensor_type, from_sensor_index, from_subsensor_index)
        if use_gpu:
            not_valid = points[:,0].eq(0) & points[:,1].eq(0) & points[:,2].eq(0)
        else:
            not_valid = np.logical_and.reduce((points[:,0] == 0, points[:,1] == 0, points[:,2] == 0))
        
        if to_sensor_type == self.TYPE_RADAR:
            points = (points, points.clone()) if use_gpu else (points, points.copy())
        points = self.transform_points_to_sensor(t, points, to_sensor_type, to_sensor_index, to_subsensor_index)
        if use_gpu:
            not_valid = (points[:,0].lt(0) | points[:,1].lt(0) | not_valid)
        else:
            not_valid = np.logical_or.reduce((points[:,0] < 0, points[:,1] < 0, not_valid))

        if use_gpu:
            points.masked_fill_(not_valid.unsqueeze(1), -1)
        else:
            points[not_valid,:] = -1
            
        return points
        
   
    def transform_points_from_ned(self, points, origin, scale=1):
        """
        Transform numpy array of local coordinates (N,E,D) to geographical points in degrees in (lat,lon,hgt). 
        The argument 'points' should be an array of shape (n,3), where n is the number of points.
        The argument 'points' can be a numpy array, torch.Tensor or torch.cuda.Tensor. If the type is numpy,
        transformed points are returned as np.float64, else the points are transformed using the GPU and 
        returned as torch.cuda.DoubleTensor.
        The argument 'origin' is the position of the origin in (lat,lon,hgt) as tuple or numpy array.
        The argument 'scale' is number of meters pr pixel used for transformation.
        The method returns an array of shape (n,3), where n is the number of input points.
        """
        use_gpu = gpu and torch.is_tensor(points)
        points = self.transfer_gpu(scale * points, use_gpu)
        R = self.transfer_gpu(self.rot_matrix_from_geog(origin).transpose(), use_gpu)
        origin = self.transfer_gpu(np.array(origin).reshape(1,3), use_gpu)
        if use_gpu:
            points_c = R.matmul(points.transpose(0,1) + 0).transpose(0,1)
        else:
            points_c = R.dot(points.transpose()).transpose()
        
        origin_c = self.geog2cart(origin)
        points = self.cart2geog(origin_c + points_c)
        
        return points
    

    def transform_points_to_ned(self, points, origin, scale=1):
        """
        Transform geographical points in degrees in (lat,lon,hgt) to local coordinates (N,E,D).
        The argument 'points' should be an array of shape (n,3), where n is the number of points.
        The argument 'points' can be a numpy array, torch.Tensor or torch.cuda.Tensor. If the type is numpy,
        transformed points are returned as np.float64, else the points are transformed using the GPU and 
        returned as torch.cuda.DoubleTensor.
        The argument 'origin' is the position of the origin in (lat,lon,hgt)
        The argument 'scale' is number of meters pr pixel used for transformation.
        The method returns an array of shape (n,3), where n is the number of input points.
        """
        use_gpu = gpu and torch.is_tensor(points)
        points = self.transfer_gpu(points, use_gpu)
        R = self.transfer_gpu(self.rot_matrix_from_geog(origin), use_gpu)
        origin = self.transfer_gpu(np.array(origin).reshape(1,3), use_gpu)
        points_c = self.geog2cart(points)
        origin_c = self.geog2cart(origin)
        if use_gpu:
            points = R.matmul((points_c - origin_c).transpose(0,1) + 0).transpose(0,1)
        else:
            points = R.dot((points_c - origin_c).transpose()).transpose()

        return (1 / scale) * points
        
        
    def transform_points_from_sensor(self, t, points, from_sensor_type, from_sensor_index, from_subsensor_index=None):
        """
        Transform numpy array of sensor image pixel coordinates (X,Y) to points in degrees in (lat,lon,hgt). 
        Assumes that points are in the sea plane.
        The argument 'points' should be an array of shape (n,2), where n is the number of points.
        The argument 'points' can be a numpy array, torch.Tensor or torch.cuda.Tensor. If the type is numpy,
        transformed points are returned as np.float64, else the points are transformed using the GPU and 
        returned as torch.cuda.DoubleTensor.
        If 't_sensor_type' is TYPE_RADAR, the argument 'points' can also be tuple of (n,3) arrays, the
        first array valid at start of scan, and the second array valid at end of the scan. If not a tuple, 
        the points are assumed to be valid at end of scan.
        The method returns an array of shape (n,3), where n is the number of input points.
        If a point is pointing above the horizon, the coordinates for this points are set to (0,0,0).
        """
        if isinstance(points, tuple):
            use_gpu = gpu and torch.is_tensor(points[0])
            points = (self.transfer_gpu(points[0], use_gpu), self.transfer_gpu(points[1], use_gpu))
        else:
            use_gpu = gpu and torch.is_tensor(points)
            points = self.transfer_gpu(points, use_gpu)

        meta = self.get_metadata(t, from_sensor_type, from_sensor_index, from_subsensor_index)
        
        if from_sensor_type == self.TYPE_CAMERA:
            points = self.transform_points_from_camera(t, points, from_sensor_index, from_subsensor_index)
            pos = meta['own_vessel']['position'].copy()
            pos[2] = self.get_spt_height(meta['own_vessel']['attitude'], self.sensor_config['Seapath']['navref_height'])
            if use_gpu:
                not_valid = (points[:,0].eq(0) & points[:,1].eq(0) & points[:,2].eq(0))
            else:
                not_valid = np.logical_and.reduce((points[:,0] == 0, points[:,1] == 0, points[:,2] == 0))
            points = self.transform_points_from_ned(points, pos)
        else:
            pos_start = meta['own_vessel_start']['position'].copy()
            pos_start[2] = self.get_spt_height(meta['own_vessel_start']['attitude'], self.sensor_config['Seapath']['navref_height'])
            pos_end = meta['own_vessel_end']['position'].copy()
            pos_end[2] = self.get_spt_height(meta['own_vessel_end']['attitude'], self.sensor_config['Seapath']['navref_height'])
            if isinstance(points, tuple):
                points_start, points_end = points
                frac_interpolation = points_end[:,0] / meta['image_dim'][0]
                points_start = self.transform_points_from_radar(t, points_start, from_sensor_index, start_of_scan=True)
                if use_gpu:
                    not_valid_start = points_start[:,0].eq(0) & points_start[:,1].eq(0) & points_start[:,2].eq(0)
                else:
                    not_valid_start = np.logical_and.reduce((points_start[:,0] == 0, points_start[:,1] == 0, points_start[:,2] == 0))
                points_start = self.transform_points_from_ned(points_start, pos_start)
                
                points_end = self.transform_points_from_radar(t, points_end, from_sensor_index, start_of_scan=False)
                if use_gpu:
                    not_valid = ((points_end[:,0].eq(0) & points_end[:,1].eq(0) & points_end[:,2].eq(0)) | not_valid_start)
                else:
                    not_valid = np.logical_and.reduce((points_end[:,0] == 0, points_end[:,1] == 0, points_end[:,2] == 0))
                    not_valid = np.logical_or.reduce((not_valid_start, not_valid))
                points_end = self.transform_points_from_ned(points_end, pos_end)
                
                # Interpolate points to correct for duration of radar scan (about 2.5s)
                if use_gpu:
                    frac_interpolation = torch.stack((frac_interpolation, frac_interpolation, frac_interpolation), dim=1).type(torch.cuda.DoubleTensor)
                    points = frac_interpolation * points_start + (1 - frac_interpolation) * points_end
                else:
                    points = frac_interpolation.reshape(-1,1) * points_start + (1 - frac_interpolation.reshape(-1,1)) * points_end
            else:
                points = self.transform_points_from_radar(t, points, from_sensor_index, start_of_scan=False)
                if use_gpu:
                    not_valid = (points[:,0].eq(0) & points[:,1].eq(0) & points[:,2].eq(0))
                else:
                    not_valid = np.logical_and.reduce((points[:,0] == 0, points[:,1] == 0, points[:,2] == 0))
                points = self.transform_points_from_ned(points, pos_end)
        
        if use_gpu:
            points.masked_fill_(not_valid.unsqueeze(1), 0)
        else:
            points[not_valid,:] = 0
        
        return points
    

    def transform_points_to_sensor(self, t, points, to_sensor_type, to_sensor_index, to_subsensor_index=None, invalidate_pixels=True):
        """
        Transform points in degrees in (lat,lon,hgt) to pixels coordinates (X,Y) in sensor image.
        The argument 'points' should be an array of shape (n,3), where n is the number of points.
        The argument 'points' can be a numpy array, torch.Tensor or torch.cuda.Tensor. If the type is numpy,
        transformed points are returned as np.float64, else the points are transformed using the GPU and 
        returned as torch.cuda.DoubleTensor.
        If 't_sensor_type' is TYPE_RADAR, the argument 'points' can also be tuple of (n,3) arrays, the
        first array valid at start of scan, and the second array valid at end of the scan. If not a tuple, 
        the points are assumed to be valid at end of scan.
        If the argument 'invalidate_pixels' is True, coordinates for points outside the image boundaries are set to -1.
        The method returns an array of shape (n,2), where n is the number of input points.
        """
        if isinstance(points, tuple):
            use_gpu = gpu and torch.is_tensor(points[0])
            points = (self.transfer_gpu(points[0], use_gpu), self.transfer_gpu(points[1], use_gpu))
        else:
            use_gpu = gpu and torch.is_tensor(points)
            points = self.transfer_gpu(points, use_gpu)
            
        meta = self.get_metadata(t, to_sensor_type, to_sensor_index, to_subsensor_index)
        
        if to_sensor_type == self.TYPE_CAMERA:
            pos = meta['own_vessel']['position'].copy()
            pos[2] = self.get_spt_height(meta['own_vessel']['attitude'], self.sensor_config['Seapath']['navref_height'])
            points = self.transform_points_to_ned(points, pos)
            points = self.transform_points_to_camera(t, points, to_sensor_index, to_subsensor_index, invalidate_pixels=invalidate_pixels)
        else:
            pos_start = meta['own_vessel_start']['position'].copy()
            pos_start[2] = self.get_spt_height(meta['own_vessel_start']['attitude'], self.sensor_config['Seapath']['navref_height'])
            pos_end = meta['own_vessel_end']['position'].copy()
            pos_end[2] = self.get_spt_height(meta['own_vessel_end']['attitude'], self.sensor_config['Seapath']['navref_height'])
            if isinstance(points, tuple):
                dim_x = meta['image_dim'][0]
                points_start, points_end = points
                points_start = self.transform_points_to_ned(points_start, pos_start)
                points_start = self.transform_points_to_radar(t, points_start, to_sensor_index, start_of_scan=True, invalidate_pixels=invalidate_pixels) 
                points_end = self.transform_points_to_ned(points_end, pos_end)
                points_end = self.transform_points_to_radar(t, points_end, to_sensor_index, start_of_scan=False, invalidate_pixels=invalidate_pixels)
                
                # Interpolate points to correct for duration of radar scan (about 2.5s)
                if use_gpu:
                    frac_interpolation = points_end[:,0] / dim_x
                    wrap_dn = (points_end[:,0] - points_start[:,0]).lt(-dim_x/2)
                    frac_interpolation.masked_fill_(wrap_dn, 0)
                    frac_interpolation = frac_interpolation.unsqueeze(1).repeat(1,2)
                    points = frac_interpolation * points_start + (1 - frac_interpolation) * points_end              
                    valid = (points_start[:,0].ge(0) & points_end[:,0].ge(0) & (points_end[:,0] - points_start[:,0]).lt(dim_x/2))
                    points.masked_fill_((1-valid).unsqueeze(1), -1)
                else:
                    valid = np.logical_and.reduce((points_start[:,0] >= 0, points_end[:,0] >= 0, (points_end[:,0] - points_start[:,0]) < dim_x/2))
                    frac_interpolation = points_end[valid,0] / dim_x
                    wrap_dn = (points_end[valid,0] - points_start[valid,0]) < -dim_x/2
                    frac_interpolation[wrap_dn] = 0
                    points = np.zeros(points_start.shape).astype(np.float64) - 1
                    points[valid,:] = frac_interpolation.reshape(-1,1) * points_start[valid,:] + (1 - frac_interpolation.reshape(-1,1)) * points_end[valid,:]
            else:  
                points = self.transform_points_to_ned(points, pos_end)
                points = self.transform_points_to_radar(t, points, to_sensor_index, start_of_scan=False, invalidate_pixels=invalidate_pixels)
        
        return points
        
        
    def transform_points_from_camera(self, t, points, from_sensor_index, from_subsensor_index, ref_location=None, head_up=False):
        """
        Transform numpy array of camera image pixel coordinates (X,Y) to points in meters in (N,E,D) or heading frame. Assumes that points are in the sea plane.
        The argument 'points' should be an array of shape (n,2), where n is the number of points.
        The argument 'points' can be a numpy array, torch.Tensor or torch.cuda.Tensor. If the type is numpy,
        transformed points are returned as np.float64, else the points are transformed using the GPU and 
        returned as torch.cuda.DoubleTensor.
        The argument 'ref_location' is the offset from the Seapath reference point to the reference.
        point to be used for the returned points in meters and vessel body coordinates (fwd,stb,dwn).
        If 'ref_location' is None, returned points are relative to Seapath reference point.
        The argument 'head_up' decides the frame of the transformed points. If True, the frame is heading frame, else it is (N,E,D).
        The method returns an array of shape (n,3), where n is the number of input points.
        If a point is pointing above the horizon, the coordinates for this points are set to (0,0,0).
        """
        use_gpu = gpu and torch.is_tensor(points)
        points = self.transfer_gpu(points, use_gpu)
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
        dist_to_horizon = np.sqrt(2 * EARTH_RADIUS * height_above_sea_level + height_above_sea_level**2) 
        
        # Convert pixel coordinates to normalized coordinates by applying camera calibration
        yn, zn = self.camera_p2m(points[:,1], points[:,0], camera_matrix, dist_coeffs) 
        
        if use_gpu:
            R_ma = self.to_gpu(R_ma)
            R_att = self.to_gpu(R_att)
            xn = torch.cuda.DoubleTensor(points.size(0)).fill_(1)
            pn = torch.stack((xn, yn, zn), dim=1)
            pn = R_ma.matmul(pn.transpose(0,1) + 0)
            pn_geog = R_att.matmul(pn + 0).transpose(0,1)
            not_valid = pn_geog[:,2].lt(0)
            
            # Calculate distance x to sea plane
            x = height_above_sea_level / pn_geog[:,2]
            p = torch.stack((x, x*yn, x*zn), dim=1)
            
            # Rotate and translate to reference point
            p_transformed = R_att.matmul(R_ma.matmul(p.transpose(0,1) + 0) + self.to_gpu(T)).transpose(0,1)
            
            # Invalidate points above horizon
            dst = np.sqrt(p_transformed[:,0]**2 + p_transformed[:,1]**2)
            p_transformed.masked_fill_(dst.gt(dist_to_horizon).unsqueeze(1), 0)
            p_transformed.masked_fill_(not_valid.unsqueeze(1), 0)
        else:
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
            
            # Invalidate points above horizon
            dst = np.sqrt(p_geog[:,0]**2 + p_geog[:,1]**2)
            p_geog[dst>dist_to_horizon,:] = 0
            p_transformed[valid,:] = p_geog

        return p_transformed
    
    
    def transform_points_from_radar(self, t, points, from_sensor_index, start_of_scan, ref_location=None, head_up=False):
        """
        Transform numpy array of radar image pixel coordinates (X,Y) to points in meters in (N,E,D) or heading frame. 
        Assumes that points are in the radar (R,P) plane.
        The argument 'points' should be an array of shape (n,2), where n is the number of points.
        The argument 'points' can be a numpy array, torch.Tensor or torch.cuda.Tensor. If the type is numpy,
        transformed points are returned as np.float64, else the points are transformed using the GPU and 
        returned as torch.cuda.DoubleTensor.
        If the argument 'start_of_scan' is True, the transformation will be valid at start of scan.
        If the argument 'start_of_scan' is False, the transformation will be valid at end of scan.
        The argument 'ref_location' is the offset from the Seapath reference point to the reference 
        point to be used for the returned points in meters and vessel body coordinates (fwd,stb,dwn). 
        If 'ref_location' is None, returned points are relative to Seapath reference point.
        The argument 'head_up' decides the frame of the transformed points. If True, the frame is heading frame, else it is (N,E,D).
        The method returns an array of shape (n,3), where n is the number of input points.
        If a point is invalidated, the coordinates for this points are set to (0,0,0).
        """
        use_gpu = gpu and torch.is_tensor(points)
        points = self.transfer_gpu(points, use_gpu)
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
        
        hgt = self.get_spt_height(pva['attitude'], self.sensor_config['Seapath']['navref_height'])
        rng = self.radar_p2m(points[:,1], range_filters, cfg_sensor['m_per_sample'])
        brg = 2 * math.pi * (1 - points[:,0] / dim_x)
        
        if use_gpu:
            x = rng.clone()
            x.mul_(brg.cos())
            y = rng.clone()
            y.mul_(brg.sin())
            z = torch.cuda.DoubleTensor(points.shape[0]).fill_(0)
            p = torch.stack((x, y, z), dim=1)
            p_transformed = self.to_gpu(R_att).matmul(self.to_gpu(R_ma).matmul(p.transpose(0,1) + 0) + self.to_gpu(T)).transpose(0,1)
            p_transformed[:,2] = hgt
            p_transformed.masked_fill_(rng.lt(0).unsqueeze(1), 0)
        else:
            x = rng * np.cos(brg)
            y = rng * np.sin(brg)
            z = np.zeros((points.shape[0], 1))
            p = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)
            p_transformed = R_att.dot(R_ma.dot(p.transpose()) + T).transpose()
            p_transformed[:,2] = hgt
            p_transformed[rng<0,:] = 0
        
        return p_transformed
    
    
    def transform_points_to_camera(self, t, points, to_sensor_index, to_subsensor_index, ref_location=None, head_up=False, invalidate_pixels=True):
        """
        Transform points in meters in (N,E,D) or heading frame to pixels coordinates (X,Y) in camera image.
        The argument 'points' should be an array of shape (n,3), where n is the number of points.
        The argument 'points' can be a numpy array, torch.Tensor or torch.cuda.Tensor. If the type is numpy,
        transformed points are returned as np.float64, else the points are transformed using the GPU and 
        returned as torch.cuda.DoubleTensor.
        The argument 'ref_location' is the offset from the Seapath reference point to the reference 
        point used for the input points in meters and vessel body coordinates (fwd,stb,dwn). If None, the offset is 
        assumed to be (0,0,0).
        The argument 'head_up' decides the coordinate frame of the input points. If True, the frame is heading frame, else it is (N,E,D).
        If the argument 'invalidate_pixels' is True, coordinates for points outside the image boundaries are set to -1.
        The method returns an array of shape (n,2), where n is the number of input points.
        """  
        use_gpu = gpu and torch.is_tensor(points)
        points = self.transfer_gpu(points, use_gpu)
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
              
        # Translate and rotate target pos vector to sensor frame
        if use_gpu:
            p_vessel = self.to_gpu(R_att).matmul(points.transpose(0,1) + 0).transpose(0,1)
            p_sensor = self.to_gpu(R_ma).matmul(p_vessel.transpose(0,1) - self.to_gpu(T)).transpose(0,1)
        else:
            p_vessel = R_att.dot(points.transpose()).transpose()
            p_sensor = R_ma.dot(p_vessel.transpose() - T).transpose()
        
        # Convert to pixels by applying camera calibration
        y, x = self.camera_m2p(p_sensor[:,1], p_sensor[:,2], p_sensor[:,0], camera_matrix, dist_coeffs)
        if use_gpu:
            if invalidate_pixels:
                # Get sensor point height above sea level given attitude
                pos_sensor = self.get_sensor_pos(meta['own_vessel'], T, self.sensor_config['Seapath']['navref_height'])
                height_above_sea_level = pos_sensor[2]
                # Invalidate points above horizon and ouside image boundaries
                dist_to_horizon = np.sqrt(2 * EARTH_RADIUS * height_above_sea_level + height_above_sea_level**2)
                rng = torch.sqrt(p_sensor[:,0]**2 + p_sensor[:,1]**2 + p_sensor[:,2]**2)
                not_valid = (p_sensor[:,0].le(0) | rng.ge(dist_to_horizon) | x.lt(0) | x.ge(dim_x) | y.lt(0) | y.ge(dim_y))
                x.masked_fill_(not_valid, -1)
                y.masked_fill_(not_valid, -1)
            p_transformed = torch.stack((x,y),dim=1)
        else:
            p_transformed = np.zeros((points.shape[0], 2)).astype(np.float64) - 1
            if invalidate_pixels:
                # Get sensor point height above sea level given attitude
                pos_sensor = self.get_sensor_pos(meta['own_vessel'], T, self.sensor_config['Seapath']['navref_height'])
                height_above_sea_level = pos_sensor[2]
                # Invalidate points above horizon and ouside image boundaries
                dist_to_horizon = np.sqrt(2 * EARTH_RADIUS * height_above_sea_level + height_above_sea_level**2)
                rng = np.sqrt(p_sensor[:,0]**2 + p_sensor[:,1]**2 + p_sensor[:,2]**2)
                valid = np.logical_and.reduce((p_sensor[:,0] > 0, rng < dist_to_horizon, x >= 0, x < dim_x, y >= 0, y < dim_y))
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
        The argument 'points' should be an array of shape (n,3), where n is the number of points.
        The argument 'points' can be a numpy array, torch.Tensor or torch.cuda.Tensor. If the type is numpy,
        transformed points are returned as np.float64, else the points are transformed using the GPU and 
        returned as torch.cuda.DoubleTensor.
        If the argument 'start_of_scan' is True, the transformation will be valid at start of scan.
        If the argument 'start_of_scan' is False, the transformation will be valid at end of scan.
        The argument 'ref_location' is the offset from the Seapath reference point to the reference 
        point used for the input points in meters and vessel body coordinates (fwd,stb,dwn). If None, the offset is 
        assumed to be (0,0,0).
        The argument 'head_up' decides the coordinate frame of the input points. If True, the frame is heading frame, else it is (N,E,D).
        If the argument 'invalidate_pixels' is True, coordinates for points outside the image y boundary are set to -1.
        The method returns a numpy array of shape (n,2), where n is the number of input points.
        """
        use_gpu = gpu and torch.is_tensor(points)
        points = self.transfer_gpu(points, use_gpu)
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
        if use_gpu:
            points[:,2] = 0
            p_vessel = self.to_gpu(R_att).matmul(points.transpose(0,1) + 0).transpose(0,1)
            p_sensor = self.to_gpu(R_ma).matmul(p_vessel.transpose(0,1) - self.to_gpu(T)).transpose(0,1)
            
            # Calc bearing and range
            rng = np.sqrt(p_sensor[:,0]**2 + p_sensor[:,1]**2)
            brg = torch.atan2(p_sensor[:,1].type(torch.cuda.FloatTensor), p_sensor[:,0].type(torch.cuda.FloatTensor)).type(torch.cuda.DoubleTensor)
            
            x = brg
            x.mul_(-dim_x/(2*math.pi))
            x.add_(dim_x+0.5)
            x.fmod_(dim_x)
            y = self.radar_m2p(rng, range_filters, cfg_sensor['m_per_sample'])
            
            if invalidate_pixels:
                not_valid = y.ge(dim_y)
                x.masked_fill_(not_valid, -1)
                y.masked_fill_(not_valid, -1)

            p_transformed = torch.stack((x,y), dim=1)
        else:
            points[:,2] = 0
            p_vessel = R_att.dot(points.transpose()).transpose()
            p_sensor = R_ma.dot(p_vessel.transpose() - T).transpose()
        
            # Calc bearing and range
            rng = np.sqrt(p_sensor[:,0]**2 + p_sensor[:,1]**2)
            brg = np.arctan2(p_sensor[:,1], p_sensor[:,0])
            p_transformed = np.zeros((points.shape[0], 2)).astype(np.float64)
            p_transformed[:,0] = dim_x - dim_x * (brg / (2*math.pi)) + 0.5
            p_transformed[p_transformed[:,0] < 0] += dim_x
            p_transformed[p_transformed[:,0] >= dim_x] -= dim_x
            p_transformed[:,1] = self.radar_m2p(rng, range_filters, cfg_sensor['m_per_sample'])
            
            if invalidate_pixels:
                p_transformed[p_transformed[:,1] >= dim_y,:] = -1
            
        return p_transformed
        
    
    def transform_image(self, t, from_sensor_type, to_sensor_type, from_sensor_index, to_sensor_index, image=[], from_subsensor_index=None, to_subsensor_index=None, 
                        use_gpu=True, return_gpu_tensor=False):
        """
        Transform image from one sensor frame to another. Assumes that all points are in the sea plane.
        The argument 't' is a datetime object with second resolution for the image transformation.
        The argument 'from_sensor_type' can be either TYPE_CAMERA or TYPE_RADAR.
        The argument 'to_sensor_type' can be either TYPE_CAMERA or TYPE_RADAR.
        The argument 'from_sensor_index' is the index of the sensor to transform from.
        The argument 'to_sensor_index' is the index of the sensor to transform to.
        The argument 'image' is the image to transform as an array of shape (height, width, n), where n is the number of color channels.
        The argument 'image' can be of type np.uint8, torch.ByteTensor or torch.cuda.ByteTensor.
        Height and width of 'image' has to match original sensor image. If 'image' is empty, the original sensor image is used as input.
        The argument 'from_subsensor_index' is only required if 'from_sensor_type' is TYPE_CAMERA.
        The argument 'to_subsensor_index' is only required if 'to_sensor_type' is TYPE_CAMERA.
        If the argument 'use_gpu' is True, the GPU is used for transformation, if pytorch and GPU is available.
        If the argument 'return_gpu_tensor' is True, the type of the returned image is torch.cuda.ByteTensor instead of np.uint8, 
        if pytorch and GPU is available.
        The method returns an array of shape (height,width,n), where n is the number of color channels in input image.
        Height and width matches the dimension of the original sensor image.
        If a pixel value is undetermined, the color channels are set to 0.
        """
        use_gpu = gpu and use_gpu
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
            p = self.transfer_gpu(p, use_gpu)
            px = self.transform_points(t, p, to_sensor_type, from_sensor_type, to_sensor_index, from_sensor_index, to_subsensor_index, from_subsensor_index)
            
            if use_gpu:
                image_transformed = self.grid_sample_gpu(image, px - 0.5, (dim_x_dst, dim_y_dst))
                image_return = self.transfer_gpu(image_transformed, return_gpu_tensor)
            else:
                valid = np.logical_and.reduce((px[:,0] >= 0, px[:,1] >= 0))
                px = px.astype(np.int16)
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
    
    
    def transform_image_from_sensor(self, t, sensor_type, sensor_index, origin, subsensor_index=None, dim=2000, scale=1, image=[],
                                    use_gpu=True, return_gpu_tensor=False):
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
        The argument 'image' is the image to transform and should be an array of shape (height, width, n), where n is the number of color channels.
        The argument 'image' can be of type np.uint8, torch.ByteTensor or torch.cuda.ByteTensor.
        If the argument 'use_gpu' is True, the GPU is used for transformation, if pytorch and GPU is available.
        If the argument 'return_gpu_tensor' is True, the type of the returned image is torch.cuda.ByteTensor instead of np.uint8, 
        if pytorch and GPU is available.
        Height and width of 'image' has to match original sensor image. If 'image' is empty, the original sensor image is used as input.
        The method returns an array of shape (dim,dim,n), where n is the number of color channels in input image.
        If a point is outside sensor field of view, the color channels are set to 0.
        """
        use_gpu = gpu and use_gpu
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
            p = self.transfer_gpu(p, use_gpu)
            origin = (origin[0], origin[1], 0)
            p = self.transform_points_from_ned(p, origin)
            if sensor_type==self.TYPE_RADAR:
                p = (p, p.clone()) if use_gpu else (p, p.copy())
            px = self.transform_points_to_sensor(t, p, sensor_type, sensor_index, subsensor_index)

            if use_gpu:
                image_transformed = self.grid_sample_gpu(image, px, (dim, dim))
                image_return = self.transfer_gpu(image_transformed, return_gpu_tensor)
            else:
                valid = np.logical_and.reduce((px[:,0] >= 0, px[:,1] >= 0)) 
                px = px.astype(np.int16)             
                
                if image.ndim == 3:
                    image_transformed = np.zeros((dim * dim, image.shape[2])).astype(np.uint8)
                    image_transformed[valid,:] = image[px[valid,0], px[valid,1], :]
                    image_transformed = np.reshape(image_transformed,(dim, dim, image.shape[2]))
                else:
                    image_transformed = np.zeros((dim * dim)).astype(np.uint8)
                    image_transformed[valid] = image[px[valid,0], px[valid,1]]
                    image_transformed = np.reshape(image_transformed, (dim, dim))
                        
                image_return = image_transformed.swapaxes(0,1)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')

        return image_return
    
    
    def transform_image_to_sensor(self, t, sensor_type, sensor_index, image, origin, subsensor_index=None, scale=1,
                                  use_gpu=True, return_gpu_tensor=False):
        """
        Transform a geographic image with Seapath reference point as origin to a sensor image.
        The origin in the input image is assumed to be located in the center (dim/2,dim/2) of the transformed image.
        The argument 't' is a datetime object with second resolution for the transformed image.
        The argument 'sensor_type' can be either TYPE_CAMERA or TYPE_RADAR.
        The argument 'sensor_index' is the index of the sensor to transform to.
        The argument 'image' is a numpy array of shape (dim,dim,n), where n is the number of color channels.
        The argument 'image' can be of type np.uint8, torch.ByteTensor or torch.cuda.ByteTensor.
        The argument 'origin' is the origin of the input image in degrees (lat,lon) as a tuple or numpy array.
        The argument 'subsensor_index' is the sub index of the sensor to transform to. Only used if 'sensor_type' is TYPE_CAMERA.
        The argument 'scale' is the number of meters pr. pixel in the input image.
        If the argument 'use_gpu' is True, the GPU is used for transformation, if pytorch and GPU is available.
        If the argument 'return_gpu_tensor' is True, the type of the returned image is torch.cuda.ByteTensor instead of np.uint8, 
        if pytorch and GPU is available.
        The method returns an array of shape (height,width,n), where n is the number of color channels in input image.
        Height and width matches the dimension of the original sensor image.
        If a pixel value is undetermined, the color channels are set to 0.
        """
        use_gpu = gpu and use_gpu
        image_return = []
        try:
            meta = self.get_metadata(t, sensor_type, sensor_index, subsensor_index) 
            dim_x, dim_y = meta['image_dim'][0], meta['image_dim'][1]
            
            x_dest, y_dest = np.meshgrid(np.arange(dim_x), np.arange(dim_y))
            p = np.stack([x_dest.reshape(-1), y_dest.reshape(-1)], axis=1)
            p = self.transfer_gpu(p, use_gpu)
            if sensor_type==self.TYPE_RADAR:
                p = (p, p.clone()) if use_gpu else (p, p.copy())
            p = self.transform_points_from_sensor(t, p, sensor_type, sensor_index, subsensor_index)
            origin = (origin[0], origin[1], 0)
            px = self.transform_points_to_ned(p, origin)
            
            if use_gpu:
                not_valid = px[:,0].eq(0) & px[:,1].eq(0) & px[:,2].eq(0)
                px = px / scale
                px[:,0] = -px[:,0] + image.shape[0]/2
                px[:,1] += image.shape[1]/2
                px.masked_fill_(not_valid.unsqueeze(1), -1)
                image_transformed = self.grid_sample_gpu(image, px, (dim_x, dim_y))
                image_return = self.transfer_gpu(image_transformed, return_gpu_tensor)
            else:    
                px_outside = np.logical_and.reduce((px[:,0] == 0, px[:,1] == 0, px[:,2] == 0))
                px = px / scale
                px[:,0] = -px[:,0] + image.shape[0]/2
                px[:,1] += image.shape[1]/2 
                px = px.astype(np.int16)
        
                valid = np.logical_and.reduce((px[:,0] >= 0, px[:,0] < image.shape[0], px[:,1] >= 0, px[:,1] < image.shape[1], px_outside == False))
                if image.ndim == 3:
                    image_transformed = np.zeros((dim_x * dim_y, image.shape[2])).astype(np.uint8)
                    image_transformed[valid,:] = image[px[valid,0], px[valid,1], :]
                    image_transformed = np.reshape(image_transformed, (dim_y, dim_x, image.shape[2]))
                else:
                    image_transformed = np.zeros((dim_x * dim_y)).astype(np.uint8)
                    image_transformed[valid] = image[px[valid,0], px[valid,1]]
                    image_transformed = np.reshape(image_transformed, (dim_y, dim_x))
                        
                image_return = image_transformed.swapaxes(0,1)
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')

        return image_return
        
    
    def transform_images_to_panorama(self, t, sensor_type=0, sensors=[(2,0),(2,1),(2,2),(0,0),(0,1),(0,2)], 
                                    translation=(-10,0.1,-5.5), dim=(480,3840), vfov=(-15,10),
                                    use_gpu=True, return_gpu_tensor=False):
        """
        Transform sensor images to a panoramic image. All radar points are assumed to be in the sea plane. 
        Camera points below horizon is assumed to be in the sea plane. Camera points above the horizon is assumed to lie on a hemisphere.
        The argument 't' is a datetime object with second resolution for the image to transform.
        The argument 'sensor_type' can be either TYPE_CAMERA or TYPE_RADAR.
        For TYPE_CAMERA, the argument 'sensors' is a tuple (sensor_index, subsensor_index) or a list of tuples of sensors defining the images to be
        transformed and the order which they are added.
        For TYPE_RADAR, the argument 'sensors' is a sensor index or a list of sensor indexes defining the image to be transformed. 
        If it's a list with more than one element, the first sensor with an image for this second is used.
        The argument 'dim' is the height and width of the transformed image in pixels.
        The argument 'vfov' is the vertical field of view in degrees.
        If the argument 'use_gpu' is True, the GPU is used for transformation, if pytorch and GPU is available.
        If the argument 'return_gpu_tensor' is True, the type of the returned image is torch.cuda.ByteTensor instead of np.uint8, 
        if pytorch and GPU is available.
        The method returns an array of shape (dim[0],dim[1],n), where n is the number of color channels in input images.
        If a point is outside sensor field of view, the color channels are set to 0.
        """
        use_gpu = gpu and use_gpu
        image_return = []
        try:
            pva = self.get_seapath_data(t)
            spt_hgt = self.sensor_config['Seapath']['navref_height']
            origin = self.get_sensor_pos(pva, translation, spt_hgt)
            dist_to_horizon = np.sqrt(2 * EARTH_RADIUS * origin[2] + origin[2] ** 2)
            
            # Draw points below horizon on the sea surface
            head = pva['attitude'][2] * math.pi/180
            brg = np.linspace(-math.pi + head, math.pi + head, dim[1])
            el = np.linspace(vfov[1] * math.pi / 180, vfov[0] * math.pi / 180, dim[0])
            dist = 0*el - 1
            visible_sea_surface = el<0
            dist[visible_sea_surface] = origin[2] / np.tan(-el[visible_sea_surface])
            dist_dest, brg_dest = np.meshgrid(dist, brg)
            x_dest = dist_dest * np.cos(brg_dest)
            y_dest = dist_dest * np.sin(brg_dest)
            z_dest = np.zeros(dim[0]*dim[1]) + origin[2]
            p = np.stack([x_dest.reshape(-1), y_dest.reshape(-1), z_dest.reshape(-1)], axis=1)

            # Draw points above horizon on a hemisphere
            head = pva['attitude'][2]
            brg = np.linspace(360 - head, -head, dim[1])
            el = np.linspace(vfov[1], vfov[0], dim[0])
            el_dest, brg_dest = np.meshgrid(el, brg)
            rad_dest = np.zeros(dim[0]*dim[1]) - EARTH_RADIUS + dist_to_horizon - 1000
            p_sphere = np.stack([el_dest.reshape(-1), brg_dest.reshape(-1), rad_dest.reshape(-1)], axis=1)
            p_sphere = self.geog2cart(p_sphere)
            p_sphere[:,0] = -p_sphere[:,0]
            p_sphere[:,2] = -p_sphere[:,2]
            
            if sensor_type == self.TYPE_CAMERA:
                mask = np.logical_or.reduce((dist_dest.reshape(-1)<=0, dist_dest.reshape(-1)>=(dist_to_horizon - 1000)))
                p[mask,:] = p_sphere[mask,:]
            else:
                mask = dist_dest.reshape(-1)<=0
                p[mask,:] = 100000
                
            p = self.transfer_gpu(p, use_gpu)
            p = self.transform_points_from_ned(p, origin)
                 
            if sensor_type == self.TYPE_RADAR:
                if not isinstance(sensors, list):
                    sensors = [sensors]
                
                for sensor in sensors:
                    image = self.load_image(t, self.TYPE_RADAR, sensor)
                    if image != []:
                        break
                    
                if image == []:
                    return []
                
                px = self.transform_points_to_sensor(t, (p, p.clone()) if use_gpu else (p, p.copy()), self.TYPE_RADAR, sensor)
                if use_gpu:
                    image_return = self.grid_sample_gpu(image, px, dim)
                else:
                    valid = np.logical_and.reduce((px[:,0] >= 0, px[:,1] >= 0)) 
                    px = px.astype(np.int16)            
                    image_transformed = np.zeros((dim[0] * dim[1])).astype(np.uint8)
                    image_transformed[valid] = image[px[valid,0], px[valid,1]]
                    image_return = np.reshape(image_transformed, (dim[1], dim[0])).swapaxes(0,1) 
                
            elif sensor_type == self.TYPE_CAMERA:
                if use_gpu:
                    image_return = torch.cuda.ByteTensor(dim[0], dim[1], 3)
                    image_return.zero_()
                else:
                    image_return = np.zeros((dim[0], dim[1], 3), dtype=np.uint8)
                image_aug = np.zeros((1920, 2560, 4)) 
                
                if not isinstance(sensors, list):
                    sensors = [sensors]

                for sensor in sensors:
                    sensor_index = sensor[0]
                    subsensor_index = sensor[1]
                    
                    image = self.load_image(t, sensor_type, sensor_index, subsensor_index)
                    if image == []:
                        return []
                    
                    image_aug[:,:,:3] = image
                    image_aug[:,:,3] = 1
                    px = self.transform_points_to_sensor(t, p.clone() if use_gpu else p.copy(), sensor_type, sensor_index, subsensor_index)

                    if use_gpu:
                        image_transformed = self.grid_sample_gpu(image_aug, px, dim)
                        mask = image_transformed[:,:,3].unsqueeze(2)
                        image_transformed[:,:,:3].mul_(mask)
                        image_return.mul_(1-mask)
                        image_return.add_(image_transformed[:,:,:3])
                    else:
                        valid = np.logical_and.reduce((px[:,0] >= 0, px[:,1] >= 0)) 
                        px = px.astype(np.int16)                                     
                        image_transformed = np.zeros((dim[0] * dim[1], 4)).astype(np.uint8)
                        image_transformed[valid,:] = image_aug[px[valid,0], px[valid,1], :]
                        image_transformed = np.reshape(image_transformed,(dim[1], dim[0], 4)).swapaxes(0,1)                        
                        mask = image_transformed[:,:,3].reshape(dim[0],dim[1],1)
                        image_return = (1 - mask) * image_return + mask * image_transformed[:,:,:3]
        except:
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')

        if use_gpu:
            image_return = self.transfer_gpu(image_return, return_gpu_tensor)
            
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
        use_gpu = gpu and torch.is_tensor(xm)
        if use_gpu:
            xm = xm.type(torch.cuda.DoubleTensor)
            ym = ym.type(torch.cuda.DoubleTensor)
            zm = zm.type(torch.cuda.DoubleTensor)
            
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
        xp = xp * fx + ux + 0.5
        yp = yp * fy + uy + 0.5
          
        return xp, yp
    
    
    def camera_p2m(self, xp, yp, camera_matrix, dist_coeffs):
        """
        Function for converting pixel coordinates in camera image to cartesian points.
        """
        use_gpu = gpu and torch.is_tensor(xp)
        if use_gpu:
            xp = xp.type(torch.cuda.DoubleTensor)
            yp = yp.type(torch.cuda.DoubleTensor)
            
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
        For ranges larger than defined by range filters, pixel value 10000 is returned.
        """
        use_gpu = gpu and torch.is_tensor(m)
        if use_gpu:
            m = m.type(torch.cuda.DoubleTensor)
            p = torch.cuda.DoubleTensor(m.size(0)).fill_(10000)
        else:
            m = np.array(m)
            p = np.zeros(len(m)).astype(np.float64) + 10000
            
        num_f = len(range_filters)
        dst_acc = 0
        px_acc = 0
        for j in range(num_f):
            if range_filters[j][0] == 0:
                break
            dst_acc_prev = dst_acc
            dst_acc += range_filters[j][1] * m_per_sample * range_filters[j][0]
            if use_gpu:
                mask = (m.gt(dst_acc_prev) & m.le(dst_acc))
                source = px_acc + (m - dst_acc_prev) / (m_per_sample * range_filters[j][0])
                p[mask] = source[mask]
            else:
                ind = np.logical_and.reduce((m > dst_acc_prev, m <= dst_acc))
                p[ind] = px_acc + (m[ind] - dst_acc_prev) / (m_per_sample * range_filters[j][0])
            px_acc += range_filters[j][1]
        
        return p
  
    
    def radar_p2m(self, px, range_filters, m_per_sample):
        """
        Function for converting radar range in pixels to meters according to range filters.
        """
        use_gpu = gpu and torch.is_tensor(px)
        if use_gpu:
            px = px.type(torch.cuda.DoubleTensor).add_(0.5)
            m = torch.cuda.DoubleTensor(px.size(0)).fill_(-1)
        else:
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
            if use_gpu:
                mask = (px.gt(px_acc_prev) & px.le(px_acc))
                source = dst_acc + (px - px_acc_prev) * range_filters[j][0] * m_per_sample
                m[mask] = source[mask]
            else:
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
    
    
    def geog2cart(self, g):
        use_gpu = gpu and torch.is_tensor(g)
        
        r = EARTH_RADIUS + g[:,2]
        arg1 = 0.5 * math.pi - g[:,0] * math.pi / 180
        arg2 = g[:,1] * math.pi / 180
        x = r * np.sin(arg1) * np.cos(arg2)
        y = r * np.sin(arg1) * np.sin(arg2)
        z = r * np.cos(arg1)
        if use_gpu:
            c = torch.stack((x,y,z), dim=1)
        else:
            c = np.stack((x,y,z), axis=1)
            
        return c
    
    
    def cart2geog(self, c):
        use_gpu = gpu and torch.is_tensor(c)
        
        r = np.sqrt(c[:,0]**2 + c[:,1]**2 + c[:,2]**2)
        if use_gpu:
            lat = 0.5 * math.pi - torch.acos(c[:,2]/r)
            lon = torch.atan(c[:,1]/c[:,0])
            hgt = r - EARTH_RADIUS
            g = torch.stack((lat * 180 / math.pi, lon * 180 / math.pi, hgt), dim=1)
        else:
            lat = 0.5 * math.pi - np.arccos(c[:,2]/r)
            lon = np.arctan(c[:,1]/c[:,0])
            hgt = r - EARTH_RADIUS
            g = np.stack((lat * 180 / math.pi, lon * 180 / math.pi, hgt), axis=1) 
            
        return g


    @staticmethod
    def dlat2m(dlat):
        """
        Function for converting difference in latitude (radians) to meters.
        Only accurate for small distances.
        """
        return EARTH_RADIUS * dlat
     
        
    @staticmethod   
    def m2dlat(m):
        """
        Function for converting meters to difference in latitude (radians).
        Only accurate for small distances.
        """
        return m / EARTH_RADIUS
    
    
    @staticmethod
    def dlon2m(dlon, lat):
        """
        Function for converting difference in longitude (radians) to meters.
        Only accurate for small distances.
        """
        return EARTH_RADIUS * dlon * math.cos(lat)
    
    
    @staticmethod
    def m2dlon(m, lat):
        """
        Function for converting meters to difference in longitude (radians).
        Only accurate for small distances.
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
  
    
    @staticmethod
    def rot_matrix_from_geog(g):
        """
        Function for creating a rotation matrix lat,lon. Used to rotate vector between ECEF and NED.
        """
        lat_rad = g[0] * math.pi / 180
        lon_rad = g[1] * math.pi / 180
        R = np.zeros((3,3))
        R[0,0] = -math.sin(lat_rad) * math.cos(lon_rad)
        R[0,1] = -math.sin(lat_rad) * math.sin(lon_rad)
        R[0,2] = math.cos(lat_rad)
        R[1,0] = -math.sin(lon_rad)
        R[1,1] = math.cos(lon_rad)
        R[1,2] = 0
        R[2,0] = -math.cos(lat_rad) * math.cos(lon_rad)
        R[2,1] = -math.cos(lat_rad) * math.sin(lon_rad)
        R[2,2] = -math.sin(lat_rad)

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
        
        return ref_hgt[0]
        
    
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
     
    
    def to_gpu(self, tensor):
        """
        Move a tensor to torch and GPU if not already there
        """
        if gpu:
            if not torch.is_tensor(tensor):
                tensor = torch.from_numpy(tensor)
            if not tensor.is_cuda:
                tensor = tensor.cuda().type(torch.cuda.DoubleTensor)

        return tensor
  
    
    def from_gpu(self, tensor):
        """
        Move a tensor to CPU and numpy if not already there
        """
        if gpu:
            if torch.is_tensor(tensor):
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                tensor = tensor.numpy()

        return tensor


    def transfer_gpu(self, tensor, use_gpu):
        """
        Move a to CPU or GPU based on argument 'use_gpu' if not already there
        """
        if use_gpu:
            return self.to_gpu(tensor)
        else:
            return self.from_gpu(tensor)

    
    def grid_sample_gpu(self, image, points, dim_dst):
        """
        Resample an image using pytoch method grid_sample().
        This method is differentiable.
        """
        image = self.to_gpu(image.copy())
        points = self.to_gpu(points)
        ndim = image.dim()
        grid_x = (2/image.size(0)) * points[:,0] - 1
        grid_y = (2/image.size(1)) * points[:,1] - 1
        grid_x = grid_x.resize_(dim_dst[1], dim_dst[0])
        grid_y = grid_y.resize_(dim_dst[1], dim_dst[0])
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0)
        del points, grid_x, grid_y
        
        if ndim==2:
            image = image.unsqueeze(2)
        image = image.permute(2, 1, 0).unsqueeze(0).type(torch.cuda.DoubleTensor)
        image = Variable(image)
        grid = Variable(grid, requires_grad=False)
        image_resampled = F.grid_sample(image, grid).data.add(0.5).type(torch.cuda.ByteTensor).squeeze(0).permute(2, 1, 0)
        if ndim==2:
            image_resampled = image_resampled.squeeze(2)
            
        return image_resampled
            