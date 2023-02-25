'''
@author jasperan

This file will distribute both train.csv and test.csv into different folders, to prepare for training.
'''

import pandas as pd
import shutil
import os

# default path for my dataset H:\WORK\reto_nuwe
# We read the dataset from the local file

base_path = """H:\\WORK\\reto_nuwe\\reto"""
df_train = pd.read_csv('{}\\train.csv'.format(base_path), sep=',', engine='python')
df_test = pd.read_csv('{}\\test.csv'.format(base_path), sep=',', engine='python')

# .iloc[i] to get ith row
print('Dataset Lengths: {} train | {} test'.format(len(df_train), len(df_test)))

print(df_train.iloc[0]['path_img'])


for x in range(len(df_train)):
    current_img = df_train.iloc[x]['path_img']
    assert 'all_imgs/' in current_img
    original_file_path = '{}\\{}'.format(base_path, current_img)
    print('[TRAIN] {}: {}'.format(original_file_path, os.path.isfile(original_file_path)))
    shutil.copy(original_file_path, "H:\\WORK\\reto_nuwe\\reto\\train\\{}".format(df_train.iloc[x]['label']))


for x in range(len(df_test)):
    current_img = df_test.iloc[x]['path_img']
    assert 'all_imgs/' in current_img
    original_file_path = '{}\\{}'.format(base_path, current_img)
    print('[TEST] {}: {}'.format(original_file_path, os.path.isfile(original_file_path)))
    shutil.copy(original_file_path, "H:\\WORK\\reto_nuwe\\reto\\test\\{}".format(df_test.iloc[x]['label'])) # error identified


#shutil.move("H:\\WORK\\reto_nuwe\\reto\\all_imgs\\1.txt", "H:\\WORK\\reto_nuwe\\reto\\test\\1.txt")

#shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
