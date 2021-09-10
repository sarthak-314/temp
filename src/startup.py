"""
Startup script to run in the beggining of every Jupyter Notebook
- Import common libraries
- Import common functions & classes
- Define important paths & constants
"""
# Commonly Used Libraries
from IPython.display import clear_output 
from functools import partial
from termcolor import colored
from tqdm.auto import tqdm
from pathlib import Path 
from time import time
import pandas as pd
import numpy as np
import pickle
import random
import json
import cv2
import gc
import os

import tensorflow as tf
import torch
from src.utils import *

# Uncommonly Used Libraries
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, asdict
from distutils.dir_util import copy_tree
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import warnings
import shutil
import math
import glob
import sys

# Competition Specific Constants
COMP_NAME = 'chaii-hindi-and-tamil-question-answering'
DRIVE_DIR = Path('/content/drive/MyDrive/Chai')

# Jupyter Notebook Setup
def _setup_jupyter_notebook(): 
    from IPython.core.interactiveshell import InteractiveShell
    from IPython import get_ipython
    InteractiveShell.ast_node_interactivity = "all"
    ipython = get_ipython()
    try: 
        ipython.magic('matplotlib inline')
        ipython.magic('load_ext autoreload')
        ipython.magic('autoreload 2')
    except: 
        print('could not load ipython magic extensions')
_setup_jupyter_notebook()


# Solve Environment, Hardware & Online Status
def _solve_env(): 
    if 'KAGGLE_CONTAINER_NAME' in os.environ: 
        return 'Kaggle'
    elif Path('/content/').exists(): 
        return 'Colab'
    else: 
        return 'Local'
def _solve_hardware(): 
    if torch.cuda.is_available(): 
        print('GPU Device:', colored(torch.cuda.get_device_name(0), 'green'))
        return 'GPU'
    elif 'TPU_NAME' in os.environ: 
        return 'TPU'
    else: 
        return 'CPU'
def _solve_online_status(): 
    try: 
        os.system('pip install wandb')
        return True
    except: 
        return False
    
ENV = _solve_env()
HARDWARE = _solve_hardware()
IS_ONLINE = _solve_online_status()
print('Notebook running on', colored(ENV, 'blue'), 'on ', colored(HARDWARE, 'blue'))


# Useful Paths for each environment
KAGGLE_INPUT_DIR = Path('/kaggle/input')
if ENV == 'Colab': 
    WORKING_DIR = Path('/content')
    TMP_DIR = Path('/content/tmp')
elif ENV == 'Kaggle': 
    WORKING_DIR = Path('/kaggle/working')
    TMP_DIR = Path('/kaggle/tmp')
else: 
    WORKING_DIR = Path('C:/Users/sarth/Desktop/chai')
    TMP_DIR = WORKING_DIR / 'tmp'

# Check if repo is loaded correctly
def _ensure_repo_dir_is_correct(): 
    if ENV == 'Kaggle': 
        assert Path('/kaggle/working/temp').exists(), red('Wrong Repo Directory')
    elif ENV == 'Colab': 
        assert Path('/content/temp').exists(), red('Wrong Repo Directory')
_ensure_repo_dir_is_correct()
    
# Mount Drive in Colab
def _mount_drive(): 
    from google.colab import drive
    drive.mount('/content/drive')
if ENV == 'Colab': 
    _mount_drive()


# Startup Notebook Functions
def ignore_warnings(should_ignore): 
    if should_ignore: 
        warnings.filterwarnings('ignore')
    else: 
        warnings.filterwarnings("ignore", category=DeprecationWarning) 

REPO_PATH = 'https://github.com/sarthak-314/chai'
def sync(): 
    'Sync Notebook with VS Code'
    os.chdir(WORKING_DIR/'temp')
    subprocess.run(['git', 'pull'])
    sys.path.append(str(WORKING_DIR/'temp'))
    os.chdir(WORKING_DIR)

def clone_repo(repo_url): 
    repo_name = repo_url.split('/')[-1].replace('-', '_')
    clone_dir = str(WORKING_DIR/repo_name)
    subprocess.run(['git', 'clone', repo_url, clone_dir])
    os.chdir(clone_dir)
    sys.path.append(clone_dir)
    print(f'Repo {repo_url} cloned')    