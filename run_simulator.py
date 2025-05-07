import os
import gc
import sys
import json
import yaml
import time
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append('./src/')

from ops import *
from tiled_ops import *
from modules import *
from attention_cim import AttentionCim
from fc1_cim import Fc1Cim
from fc2_cim import Fc2Cim
from buffer import Buffer
from accelerator import *
from simulator import *
from utils import *
from dict2ops import main as dict2ops

def main(model_dict: dict, config: dict, constants: dict, logs_dir: str, debug=False, plot_steps=100):
	
	#print(f"logs_dir: {logs_dir}, type: {type(logs_dir)}")
	simulate(model_dict, config, constants, logs_dir, debug, plot_steps)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Input parameters for conversion',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model_dict_path',
		metavar='',
		type=str,
		default='./model_dicts/model-opt-125m.json',
		help='path where the model dictionary file is stored')
	parser.add_argument('--config_path',
		metavar='',
		type=str,
		default='./config/config_opt125m_32_32_TT.yaml',
		help='path to the accelerator configuration file')
	parser.add_argument('--constants_path',
		metavar='',
		type=str,
		default='./constants/constants.yaml',
		help='path to the accelerator constants file')
	parser.add_argument('--logs_dir',
		metavar='',
		type=str,
		default='./logs/',
		help='directory to store results')
	parser.add_argument('--debug',
		dest='debug',
		help='print debugging statements',
		action='store_true')

	args = parser.parse_args()

	if os.path.exists(args.model_dict_path):
		model_dict = json.load(open(args.model_dict_path, 'r'))
	else:
		raise FileNotFoundError(f'Couldn\'t find JSON file for given path: {args.model_dict_path}')

	if os.path.exists(args.config_path):
		config = yaml.safe_load(open(args.config_path, 'r'))
	else:
		raise FileNotFoundError(f'Couldn\'t find YAML file for given path: {args.config_path}')

	if os.path.exists(args.constants_path):
		constants = yaml.safe_load(open(args.constants_path, 'r'))
	else:
		raise FileNotFoundError(f'Couldn\'t find YAML file for given path: {args.constants_path}')
	
	if os.path.exists(args.logs_dir): shutil.rmtree(args.logs_dir)
	os.makedirs(os.path.join(args.logs_dir, 'metrics'))
	
	main(model_dict, config, constants, args.logs_dir, args.debug, plot_steps=100)
