from ultralytics import YOLO
import json
import pandas as pd
import torch
import argparse
import os
import platform
import sys
from pathlib import Path


model = YOLO("H:/models/nuwe.pt") # load pretrained model (recommended for training)
base_path = """H:\\WORK\\reto_nuwe\\reto"""


if __name__ == '__main__':
    final_object = {}
    df_test = pd.read_csv('{}\\test.csv'.format(base_path), sep=',', engine='python')

    # .iloc[i] to get ith row
    print('Test Data Length:{} test'.format(len(df_test)))


    for x in range(len(df_test)):
        current_img = df_test.iloc[x]['path_img']
        # Run Model Prediction
        original_file_path = '{}\\{}'.format(base_path, current_img)
        print(original_file_path)
        results = model(original_file_path)  # predict on an image
        print(results)

        print('[TEST] {}: {}'.format(original_file_path, os.path.isfile(original_file_path)))



    test_file = pd.read_csv('./test.csv')
    results = model.val()


{
    "target": {
        "0": 1,
    }
}

