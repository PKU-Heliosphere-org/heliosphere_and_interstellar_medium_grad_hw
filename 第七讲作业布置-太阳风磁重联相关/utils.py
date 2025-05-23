'''BY ZIQI WU @ 2025/05/22'''
import os,re
import numpy as np
from datetime import datetime,timedelta
import imageio

def interp_epoch(value, from_epoch, to_epoch):
    return np.interp(np.array((to_epoch - to_epoch[0]) / timedelta(days=1), dtype='float64'),
                     np.array((from_epoch - to_epoch[0]) / timedelta(days=1), dtype='float64'), value)

def get_all_filenames(folder_path):
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filenames.append(file)
    return filenames

def folder_to_movie(folder_path, filename_pattern='a(.+)b', time_format='%Y%m%dT%H%M%S', export_pathname='a',
                    video_format='.mp4' ,**kwargs):

    filenames = np.array(get_all_filenames(folder_path))

    pattern = re.compile(filename_pattern)
    dt_strs=np.zeros_like(filenames)
    epoch = np.zeros_like(filenames)
    for i in range(len(filenames)):
        try:
            dt_strs[i] = pattern.findall(filenames[i])[0] #np.array([pattern.findall(name_)[0] for name_ in filenames])
            epoch[i] = datetime.strptime(dt_strs[i],time_format)
        except:
            print('Bad Name: ',filenames[i])
            dt_strs[i] = ''
            filenames[i] = ''
            epoch[i] = ''

    filenames = filenames[filenames!='']
    epoch = epoch[epoch!='']
    sort_arg = np.argsort(epoch)
    epoch = epoch[sort_arg]
    filenames = filenames[sort_arg]
    print(epoch)
    print(filenames)
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(folder_path + filename))
    imageio.mimsave(export_pathname + video_format, frames, **kwargs)
    print('Writing movie to ' + export_pathname + video_format + ' from imgs ' + filename_pattern + ' in ' + folder_path)

