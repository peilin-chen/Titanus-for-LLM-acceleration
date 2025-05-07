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
import math

from ops import *
from tiled_ops import *
from modules import *
from attention_cim import AttentionCim
from fc1_cim import Fc1Cim
from fc2_cim import Fc2Cim
from buffer import Buffer
from accelerator import *
from utils import *
from dict2ops import main as dict2ops

DO_LOGGING = True
gc.disable()

def get_op_list(ops, op_idx, batch_size):
	assert type(op_idx) == list

	if op_idx[0] is None:
		return [None]
	elif type(ops[op_idx[0]]) == list:
		assert len(op_idx[1]) != 0
		ops_list = []
		for head_idx, head_ops in enumerate(ops[op_idx[0]]):
			if op_idx[1][head_idx] is None: ops_list.append(None)
			elif batch_size == 1:
				ops_list.append(head_ops[op_idx[1][head_idx]])
			else:
				ops_list.append([head_ops[i] for i in range(op_idx[1][head_idx], 
					min(len(head_ops), op_idx[1][head_idx] + batch_size)) if type(head_ops[i]) == type(head_ops[op_idx[1][head_idx]])])
		return ops_list
	else:
		if batch_size == 1:
			if op_idx[0] == None: return []
			return [ops[op_idx[0]]]
		else:
			ops_list = []
			for i in range(op_idx[0], op_idx[0] + batch_size):
				if i < len(ops) and type(ops[i]) != list and type(ops[i]) == type(ops[op_idx[0]]):
					ops_list.append(ops[i])
				else:
					break
			return [ops_list]

def get_last_compute_op(head_op, head_idx, memory_op_idx, memory_ops, compute_ops):
	last_compute_op = None
	compute_op_found = False

	if memory_op_idx[1] != []:
		num_lists = len([idx for idx in range(memory_op_idx[0]) if type(memory_ops[idx]) == list]) + 1
		list_reached = 0
		for idx in range(len(compute_ops)):
			if type(compute_ops[idx]) == list:
				list_reached += 1
			if list_reached == num_lists: 
				head_ops = compute_ops[idx][head_idx]
				break
	else:
		head_ops = compute_ops

	for i, compute_op in enumerate(head_ops):
		if type(compute_op) == list: continue
		#print("compute_op.op_name:")
		#print(compute_op.op_name)
		#print("head_op.op_name[:-2]")
		#print(head_op.op_name[:-2])
		if compute_op.op_name.startswith(head_op.op_name[:-4]):
			last_compute_op_idx = i
			compute_op_found = True
			if i == len(head_ops) - 1: return compute_op
		elif compute_op_found:
			return head_ops[last_compute_op_idx]

def prev_memory_op_done(head_op, head_idx, memory_op_idx, memory_ops):
	if memory_op_idx[1] != []:
		curr_idx = memory_op_idx[1][head_idx]
		head_ops = memory_ops[memory_op_idx[0]][head_idx]
	else:
		curr_idx = memory_op_idx[0]
		head_ops = memory_ops

	for prev_op_idx in range(curr_idx - 1, -1, -1):
		if type(head_ops[prev_op_idx]) == list: continue
		if head_ops[prev_op_idx].data_type == head_op.data_type:
			head_ops[prev_op_idx].done = True
			break

def update_op_idx(ops, op_idx, stall_list, batch_size, ops_done):
	if op_idx[0] is None:
		return [None, []], 0

	for head_idx, stall in enumerate(stall_list):
		if stall:
			pass
		else:
			ops_done += batch_size
			if len(stall_list) > 1:
				if op_idx[1][head_idx] is not None and op_idx[1][head_idx] < len(ops[op_idx[0]][head_idx]) - batch_size:
					op_idx[1][head_idx] += batch_size
				else:
					op_idx[1][head_idx] = None
			else:
				for idx in range(op_idx[0] + 1, op_idx[0] + batch_size + 1):
					if idx >= len(ops) - 1 or type(ops[idx]) == list: break
				op_idx[0] = idx
				if op_idx[0] > len(ops) - 1: break
				elif type(ops[op_idx[0]]) == list:
					op_idx[1] = [0] * len(ops[op_idx[0]])
				else:
					op_idx[1] = []

	if len(stall_list) > 1 and all([op_idx[1][head_idx] is None for head_idx in range(len(stall_list))]):
		op_idx[0] += 1
		if op_idx[0] < len(ops) and type(ops[op_idx[0]]) == list:
			op_idx[1] = [0] * len(ops[op_idx[0]])
		else:
			op_idx[1] = []

	if op_idx[0] > len(ops) - 1:
		op_idx[0] = None; op_idx[1] = []

	return op_idx, ops_done

def get_ops_done(memory_ops, compute_ops):
	ops_done = 0
	for op in memory_ops:
		if type(op) == list:
			for head_ops in op:
				for head_op in head_ops:
					if head_op.done: 
						ops_done += 1
					else:
						break
		else:
			if op.done: ops_done += 1

	for op in compute_ops:
		if type(op) == list:
			for head_ops in op:
				for head_op in head_ops:
					if head_op.done: 
						ops_done += 1
					else:
						break
		else:
			if op.done: ops_done += 1

	return ops_done

def get_utilization(accelerator):
	
	num_computing_engine_qk_free, num_computing_engine_qk = accelerator.num_computing_engines_qk_free()
	num_computing_engine_sv_free, num_computing_engine_sv = accelerator.num_computing_engines_sv_free()
	num_softmax_free, num_softmax = accelerator.num_softmaxs_free()
	num_layer_norm_pre_free, num_layer_norm_pre = accelerator.num_layer_norm_pre_free()
	num_layer_norm_aft_free, num_layer_norm_aft = accelerator.num_layer_norm_aft_free()
	num_attention_cim_q_free, num_attention_cim_q = accelerator.num_attention_cim_q_free()
	num_attention_cim_k_free, num_attention_cim_k = accelerator.num_attention_cim_k_free()
	num_attention_cim_v_free, num_attention_cim_v = accelerator.num_attention_cim_v_free()
	num_attention_cim_out_free, num_attention_cim_out = accelerator.num_attention_cim_out_free()
	num_fc1_cim_free, num_fc1_cim = accelerator.num_fc1_cim_free()
	num_fc2_cim_free, num_fc2_cim = accelerator.num_fc2_cim_free()
	num_relu_free, num_relu = accelerator.num_relu_free()
	num_pruning_unit_free, num_pruning_unit = accelerator.num_pruning_unit_free()
	num_quant_unit_free, num_quant_unit = accelerator.num_quant_unit_free()
	num_dequant_unit_free, num_dequant_unit = accelerator.num_dequant_unit_free()
	
	computing_engine_qk_util = (num_computing_engine_qk - num_computing_engine_qk_free) / num_computing_engine_qk
	computing_engine_sv_util = (num_computing_engine_sv - num_computing_engine_sv_free) / num_computing_engine_sv
	softmax_util = (num_softmax - num_softmax_free) / num_softmax
	layernorm_pre_util = (num_layer_norm_pre - num_layer_norm_pre_free) / num_layer_norm_pre
	layernorm_aft_util = (num_layer_norm_aft - num_layer_norm_aft_free) / num_layer_norm_aft
	attention_cim_q_util = (num_attention_cim_q - num_attention_cim_q_free) / num_attention_cim_q
	attention_cim_k_util = (num_attention_cim_k - num_attention_cim_k_free) / num_attention_cim_k
	attention_cim_v_util = (num_attention_cim_v - num_attention_cim_v_free) / num_attention_cim_v
	attention_cim_out_util = (num_attention_cim_out - num_attention_cim_out_free) / num_attention_cim_out
	fc1_cim_util = (num_fc1_cim - num_fc1_cim_free) / num_fc1_cim
	fc2_cim_util = (num_fc2_cim - num_fc2_cim_free) / num_fc2_cim
	relu_util = (num_relu - num_relu_free) / num_relu
	pruning_unit_util = (num_pruning_unit - num_pruning_unit_free) / num_pruning_unit
	quant_unit_util = (num_quant_unit - num_quant_unit_free) / num_quant_unit
	dequant_unit_util = (num_dequant_unit - num_dequant_unit_free) / num_dequant_unit
	global_buffer_util = accelerator.global_buffer.used / accelerator.global_buffer.buffer_size
	sz_buffer_util = accelerator.sz_buffer.used / accelerator.sz_buffer.buffer_size

	return computing_engine_qk_util, computing_engine_sv_util, softmax_util, layernorm_pre_util, layernorm_aft_util, attention_cim_q_util, attention_cim_k_util, attention_cim_v_util, attention_cim_out_util, fc1_cim_util, fc2_cim_util, relu_util, pruning_unit_util, quant_unit_util, dequant_unit_util, global_buffer_util, sz_buffer_util

def log_metrics(logs, total_energy, global_buffer_energy, sz_buffer_energy, computing_engine_qk_energy, computing_engine_sv_energy, softmax_energy, layer_norm_pre_energy, 
				layer_norm_aft_energy, attention_cim_q_energy, attention_cim_k_energy, attention_cim_v_energy, attention_cim_out_energy, fc1_cim_energy, fc2_cim_energy, relu_energy, 
				pruning_unit_energy, quant_unit_energy, dequant_unit_energy, stalls, logs_dir, accelerator, plot_steps=100):
	"""Log energy values for every cycle"""
	if 'cycle' in logs:
		last_cycle = logs['cycle'][-1]
	else:
		last_cycle = 0
		for log_file in os.listdir(os.path.join(logs_dir, 'metrics')):
			last_cycle = max(last_cycle, int(log_file.split('_')[1].split('.')[0]))

		logs = {'cycle': [], 'total_energy': [], 'global_buffer_energy': [], 'sz_buffer_energy': [], 'computing_engine_qk_energy': [], 'computing_engine_sv_energy': [], 'softmax_energy': [], 
			'layer_norm_pre_energy': [], 'layer_norm_aft_energy': [], 'attention_cim_q_energy': [], 'attention_cim_k_energy': [], 'attention_cim_v_energy': [], 'attention_cim_out_energy': [], 
			'fc1_cim_energy': [], 'fc2_cim_energy': [], 'relu_energy': [], 'pruning_unit_energy': [], 'quant_unit_energy': [], 'dequant_unit_energy': [], 'global_buffer_util': [], 
			'sz_buffer_util': [], 'computing_engine_qk_util': [], 'computing_engine_sv_util': [], 'softmax_util': [], 'layernorm_pre_util': [], 'layernorm_aft_util': [], 'attention_cim_q_util': [], 'attention_cim_k_util': [], 
			'attention_cim_v_util': [], 'attention_cim_out_util': [], 'fc1_cim_util': [], 'fc2_cim_util': [], 'relu_util': [], 'pruning_unit_util': [], 'quant_unit_util': [], 'dequant_unit_util': [], 'stalls': []}

	cycle_difference = accelerator.cycle - last_cycle
	assert cycle_difference > 0 
	for c in range(last_cycle + 1, accelerator.cycle + 1):
		logs['cycle'].append(c)
		logs['total_energy'].append((total_energy[0] / cycle_difference, total_energy[1] / cycle_difference))
		logs['global_buffer_energy'].append((global_buffer_energy[0] / cycle_difference, global_buffer_energy[1] / cycle_difference))
		logs['sz_buffer_energy'].append((sz_buffer_energy[0] / cycle_difference, sz_buffer_energy[1] / cycle_difference))
		logs['computing_engine_qk_energy'].append((computing_engine_qk_energy[0] / cycle_difference, computing_engine_qk_energy[1] / cycle_difference))
		logs['computing_engine_sv_energy'].append((computing_engine_sv_energy[0] / cycle_difference, computing_engine_sv_energy[1] / cycle_difference))
		logs['softmax_energy'].append((softmax_energy[0] / cycle_difference, softmax_energy[1] / cycle_difference))
		logs['layer_norm_pre_energy'].append((layer_norm_pre_energy[0] / cycle_difference, layer_norm_pre_energy[1] / cycle_difference))
		logs['layer_norm_aft_energy'].append((layer_norm_aft_energy[0] / cycle_difference, layer_norm_aft_energy[1] / cycle_difference))
		logs['attention_cim_q_energy'].append((attention_cim_q_energy[0] / cycle_difference, attention_cim_q_energy[1] / cycle_difference))
		logs['attention_cim_k_energy'].append((attention_cim_k_energy[0] / cycle_difference, attention_cim_k_energy[1] / cycle_difference))
		logs['attention_cim_v_energy'].append((attention_cim_v_energy[0] / cycle_difference, attention_cim_v_energy[1] / cycle_difference))
		logs['attention_cim_out_energy'].append((attention_cim_out_energy[0] / cycle_difference, attention_cim_out_energy[1] / cycle_difference))
		logs['fc1_cim_energy'].append((fc1_cim_energy[0] / cycle_difference, fc1_cim_energy[1] / cycle_difference))
		logs['fc2_cim_energy'].append((fc2_cim_energy[0] / cycle_difference, fc2_cim_energy[1] / cycle_difference))
		logs['relu_energy'].append((relu_energy[0] / cycle_difference, relu_energy[1] / cycle_difference))
		logs['pruning_unit_energy'].append((pruning_unit_energy[0] / cycle_difference, pruning_unit_energy[1] / cycle_difference))
		logs['quant_unit_energy'].append((quant_unit_energy[0] / cycle_difference, quant_unit_energy[1] / cycle_difference))
		logs['dequant_unit_energy'].append((dequant_unit_energy[0] / cycle_difference, dequant_unit_energy[1] / cycle_difference))
		
		(computing_engine_qk_util, 
		computing_engine_sv_util, 
		softmax_util, 
		layernorm_pre_util, 
		layernorm_aft_util, 
		attention_cim_q_util, 
		attention_cim_k_util, 
		attention_cim_v_util, 
		attention_cim_out_util,
		fc1_cim_util,
		fc2_cim_util,
		relu_util,
		pruning_unit_util,
		quant_unit_util,
		dequant_unit_util,
		global_buffer_util,
		sz_buffer_util) = get_utilization(accelerator)
		
		logs['global_buffer_util'].append(global_buffer_util)
		logs['sz_buffer_util'].append(sz_buffer_util)
		logs['computing_engine_qk_util'].append(computing_engine_qk_util)
		logs['computing_engine_sv_util'].append(computing_engine_sv_util)
		logs['softmax_util'].append(softmax_util)
		logs['layernorm_pre_util'].append(layernorm_pre_util)
		logs['layernorm_aft_util'].append(layernorm_aft_util)
		logs['attention_cim_q_util'].append(attention_cim_q_util)
		logs['attention_cim_k_util'].append(attention_cim_k_util)
		logs['attention_cim_v_util'].append(attention_cim_v_util)
		logs['attention_cim_out_util'].append(attention_cim_out_util)
		logs['fc1_cim_util'].append(fc1_cim_util)
		logs['fc2_cim_util'].append(fc2_cim_util)
		logs['relu_util'].append(relu_util)
		logs['pruning_unit_util'].append(pruning_unit_util)
		logs['quant_unit_util'].append(quant_unit_util)
		logs['dequant_unit_util'].append(dequant_unit_util)
		
		logs['stalls'].append(stalls)

	if accelerator.cycle % plot_steps == 0:
		json.dump(logs, open(os.path.join(logs_dir, 'metrics', f'logs_{accelerator.cycle}.json'), 'w+'))
		del logs; gc.collect()
		logs = {}

	return logs

def extract_execution_cycles(logs_dir, constants):
	logs_metrics = {'cycle': [], 'total_energy': [], 'global_buffer_energy': [], 'sz_buffer_energy': [], 'computing_engine_qk_energy': [], 'computing_engine_sv_energy': [], 'softmax_energy': [], 
			'layer_norm_pre_energy': [], 'layer_norm_aft_energy': [], 'attention_cim_q_energy': [], 'attention_cim_k_energy': [], 'attention_cim_v_energy': [], 'attention_cim_out_energy': [], 
			'fc1_cim_energy': [], 'fc2_cim_energy': [], 'relu_energy': [], 'pruning_unit_energy': [], 'quant_unit_energy': [], 'dequant_unit_energy': [], 'global_buffer_util': [], 
			'sz_buffer_util': [], 'computing_engine_qk_util': [], 'computing_engine_sv_util': [], 'softmax_util': [], 'layernorm_pre_util': [], 'layernorm_aft_util': [], 'attention_cim_q_util': [], 'attention_cim_k_util': [], 
			'attention_cim_v_util': [], 'attention_cim_out_util': [], 'fc1_cim_util': [], 'fc2_cim_util': [], 'relu_util': [], 'pruning_unit_util': [], 'quant_unit_util': [], 'dequant_unit_util': [], 'stalls': []}
	
	log_files = os.listdir(os.path.join(logs_dir, 'metrics'))
	
	log_files = sorted(log_files, key=lambda log_file: int(log_file.split('_')[1].split('.')[0]))
	
	for log_file in log_files:
		logs_temp = json.load(open(os.path.join(logs_dir, 'metrics', log_file)))
		
		for key in logs_metrics.keys():
			logs_metrics[key].extend(logs_temp[key])
	
	execution_cycles = {}
	modules = {
		'LayerNorm Pre': 'layernorm_pre_util',
		'AttCIM Q': 'attention_cim_q_util',
		'AttCIM K': 'attention_cim_k_util',
		'AttCIM V': 'attention_cim_v_util',
		'Pruning Unit': 'pruning_unit_util',
		'Quant Unit': 'quant_unit_util',
		'Dequant Unit': 'dequant_unit_util',
		'CE QK': 'computing_engine_qk_util',
		'Softmax': 'softmax_util',
		'CE SV': 'computing_engine_sv_util',
		'AttCIM Out': 'attention_cim_out_util',
		'LayerNorm Aft': 'layernorm_aft_util',
		'FC1 CIM': 'fc1_cim_util',
		'ReLU': 'relu_util',
		'FC2 CIM': 'fc2_cim_util'
	}
	
	for module_name, key in modules.items():
		util_values = logs_metrics[key]
		cycles = logs_metrics['cycle']
		
		execution_cycles[module_name] = sum(1 for util in util_values if util > 0)
	
	#for module, cycles in execution_cycles.items():
	#	print(f"{module}: {cycles} cycles")
	
	return execution_cycles
	
def calculate_performance_metrics(logs_dir, accelerator, config, constants, GB_weight_access, GB_KV_cache_access, GB_other_access, execution_cycles):
	logs_metrics = {
		'total_energy': [],
		'global_buffer_energy': [],
		'sz_buffer_energy': []
	}
	
	log_files = os.listdir(os.path.join(logs_dir, 'metrics'))
	log_files = sorted(log_files, key=lambda log_file: int(log_file.split('_')[1].split('.')[0]))

	for log_file in log_files:
		logs_temp = json.load(open(os.path.join(logs_dir, 'metrics', log_file)))
		for key in logs_metrics.keys():
			logs_metrics[key].extend(logs_temp[key])
			
	compute_energy_joules = sum([sum(energy) for energy in logs_metrics['total_energy']]) * config['layer_num'] * ((config['prefill_seq']+config['decode_seq'])/16)
	global_buffer_joules = sum([sum(energy) for energy in logs_metrics['global_buffer_energy']]) * config['layer_num']
	percent_gb_weight_access = (GB_weight_access/(GB_weight_access+GB_KV_cache_access+GB_other_access)) * 100
	percent_gb_kv_cache_access = (GB_KV_cache_access/(GB_weight_access+GB_KV_cache_access+GB_other_access)) * 100
	percent_gb_other_access = (GB_other_access/(GB_weight_access+GB_KV_cache_access+GB_other_access)) * 100
	gb_weight_access_joules = global_buffer_joules * (percent_gb_weight_access/100)
	gb_kv_cache_access_joules = global_buffer_joules * (percent_gb_kv_cache_access/100)
	gb_other_access_joules = global_buffer_joules * (percent_gb_other_access/100)
	sz_buffer_joules = sum([sum(energy) for energy in logs_metrics['sz_buffer_energy']]) * config['layer_num']
	buffer_energy_joules = global_buffer_joules + sz_buffer_joules
	total_energy_joules = compute_energy_joules + buffer_energy_joules
	
	token_count = config['prefill_seq'] + config['decode_seq']
	total_cycle = accelerator.cycle * ((config['prefill_seq']+config['decode_seq'])/16)
	
	layernorm_pre_cycle = execution_cycles.get('LayerNorm Pre')
	attcim_q_cycle = execution_cycles.get('AttCIM Q')
	attcim_k_cycle = execution_cycles.get('AttCIM K')
	attcim_v_cycle = execution_cycles.get('AttCIM V')
	pruning_unit_cycle = execution_cycles.get('Pruning Unit')
	quant_unit_cycle = execution_cycles.get('Quant Unit')
	dequant_unit_cycle = execution_cycles.get('Dequant Unit')
	ce_qk_cycle = execution_cycles.get('CE QK')
	softmax_cycle = execution_cycles.get('Softmax')
	ce_sv_cycle = execution_cycles.get('CE SV')
	attcim_o_cycle = execution_cycles.get('AttCIM Out')
	layernorm_aft_cycle = execution_cycles.get('LayerNorm Aft')
	fc1_cim_cycle = execution_cycles.get('FC1 CIM')
	relu_cycle = execution_cycles.get('ReLU')
	fc2_cim_cycle = execution_cycles.get('FC2 CIM')
	
	cycles_per_token = math.ceil(total_cycle / token_count)
	
	total_module_cycle = layernorm_pre_cycle + attcim_q_cycle + attcim_k_cycle + attcim_v_cycle + pruning_unit_cycle + quant_unit_cycle + dequant_unit_cycle + ce_qk_cycle + softmax_cycle + ce_sv_cycle + attcim_o_cycle + layernorm_aft_cycle + fc1_cim_cycle + relu_cycle + fc2_cim_cycle
	
	layernorm_pre_cycle_percent = layernorm_pre_cycle / total_module_cycle
	attcim_q_cycle_percent = attcim_q_cycle / total_module_cycle
	attcim_k_cycle_percent = attcim_k_cycle / total_module_cycle
	attcim_v_cycle_percent = attcim_v_cycle / total_module_cycle
	pruning_unit_cycle_percent = pruning_unit_cycle / total_module_cycle
	quant_unit_cycle_percent = quant_unit_cycle / total_module_cycle
	dequant_unit_cycle_percent = dequant_unit_cycle / total_module_cycle
	ce_qk_cycle_percent = ce_qk_cycle / total_module_cycle
	softmax_cycle_percent = softmax_cycle / total_module_cycle
	ce_sv_cycle_percent = ce_sv_cycle / total_module_cycle
	attcim_o_cycle_percent = attcim_o_cycle / total_module_cycle
	layernorm_aft_cycle_percent = layernorm_aft_cycle / total_module_cycle
	fc1_cim_cycle_percent = fc1_cim_cycle / total_module_cycle
	relu_cycle_percent = relu_cycle / total_module_cycle
	fc2_cim_cycle_percent = fc2_cim_cycle / total_module_cycle
	
	layernorm_pre_per_token = math.ceil(cycles_per_token * layernorm_pre_cycle_percent)
	attcim_q_per_token = math.ceil(cycles_per_token * attcim_q_cycle_percent)
	attcim_k_per_token = math.ceil(cycles_per_token * attcim_k_cycle_percent)
	attcim_v_per_token = math.ceil(cycles_per_token * attcim_v_cycle_percent)
	pruning_unit_per_token = math.ceil(cycles_per_token * pruning_unit_cycle_percent)
	quant_unit_per_token = math.ceil(cycles_per_token * quant_unit_cycle_percent)
	dequant_unit_per_token = math.ceil(cycles_per_token * dequant_unit_cycle_percent)
	ce_qk_per_token = math.ceil(cycles_per_token * ce_qk_cycle_percent)
	softmax_per_token = math.ceil(cycles_per_token * softmax_cycle_percent)
	ce_sv_per_token = math.ceil(cycles_per_token * ce_sv_cycle_percent)
	attcim_o_per_token = math.ceil(cycles_per_token * attcim_o_cycle_percent)
	layernomr_aft_per_token = math.ceil(cycles_per_token * layernorm_aft_cycle_percent)
	fc1_cim_per_token = math.ceil(cycles_per_token * fc1_cim_cycle_percent)
	relu_per_token = math.ceil(cycles_per_token * relu_cycle_percent)
	fc2_cim_per_token = math.ceil(cycles_per_token * fc2_cim_cycle_percent)
	
	pipeline_stage = 7
	
	pipeline_s1 = layernorm_pre_per_token
	pipeline_s2 = attcim_q_per_token + attcim_k_per_token + attcim_v_per_token + pruning_unit_per_token
	pipeline_s3 = quant_unit_per_token
	pipeline_s4 = dequant_unit_per_token + ce_qk_per_token + softmax_per_token
	pipeline_s5 = ce_sv_per_token
	pipeline_s6 = attcim_o_per_token + layernomr_aft_per_token
	pipeline_s7 = fc1_cim_per_token + relu_per_token + fc2_cim_per_token
	
	pipeline_values = [
				pipeline_s1,
				pipeline_s2,
				pipeline_s3,
				pipeline_s4,
				pipeline_s5,
				pipeline_s6,
				pipeline_s7,
			]
	
	max_pipeline_stage_cycle = max(pipeline_values)
	
	#print(f'pipeline_s1: {pipeline_s1}')
	#print(f'pipeline_s2: {pipeline_s2}')
	#print(f'pipeline_s3: {pipeline_s3}')
	#print(f'pipeline_s4: {pipeline_s4}')
	#print(f'pipeline_s5: {pipeline_s5}')
	#print(f'pipeline_s6: {pipeline_s6}')
	#print(f'pipeline_s7: {pipeline_s7}')
	#
	#print(f'max_pipeline_stage_cycle: {max_pipeline_stage_cycle}')
	
	#print(f'layernorm_pre_cycle: {layernorm_pre_cycle}')
	#print(f'attcim_q_cycle: {attcim_q_cycle}')
	#print(f'attcim_k_cycle: {attcim_k_cycle}')
	#print(f'attcim_v_cycle: {attcim_v_cycle}')
	#print(f'pruning_unit_cycle: {pruning_unit_cycle}')
	#print(f'quant_unit_cycle: {quant_unit_cycle}')
	#print(f'dequant_unit_cycle: {dequant_unit_cycle}')
	#print(f'ce_qk_cycle: {ce_qk_cycle}')
	#print(f'softmax_cycle: {softmax_cycle}')
	#print(f'ce_sv_cycle: {ce_sv_cycle}')
	#print(f'attcim_o_cycle: {attcim_o_cycle}')
	#print(f'layernorm_aft_cycle: {layernorm_aft_cycle}')
	#print(f'fc1_cim_cycle: {fc1_cim_cycle}')
	#print(f'relu_cycle: {relu_cycle}')
	#print(f'fc2_cim_cycle: {fc2_cim_cycle}')
	
	intra_pipeline = pipeline_stage*max_pipeline_stage_cycle + (config['prefill_seq']-1)*max_pipeline_stage_cycle
	total_time_seconds_intra = ((intra_pipeline + (cycles_per_token*config['decode_seq']))*config['layer_num']) / (constants['clock_frequency'] * 1e6)
	
	intra_pipe_inter_paral = pipeline_stage*max_pipeline_stage_cycle*config['layer_num'] + (config['prefill_seq']-1)*max_pipeline_stage_cycle
	total_time_seconds_intra_inter = (intra_pipe_inter_paral + (cycles_per_token*config['decode_seq']*config['layer_num'])) / (constants['clock_frequency'] * 1e6)
	
	total_time_seconds = (total_cycle * config['layer_num']) / (constants['clock_frequency'] * 1e6)

	# metrics
	energy_efficiency_token_per_j = token_count / ((compute_energy_joules+sz_buffer_joules+gb_kv_cache_access_joules+gb_other_access_joules) * 1e-9) 
	#print("((compute_energy_joules+sz_buffer_joules+gb_kv_cache_access_joules+gb_other_access_joules) * 1e-3):")
	#print(((compute_energy_joules+sz_buffer_joules+gb_kv_cache_access_joules+gb_other_access_joules) * 1e-3))
	#print("compute_energy_joules:")
	#print(compute_energy_joules)
	#print("sz_buffer_joules:")
	#print(sz_buffer_joules)
	#print("gb_kv_cache_access_joules:")
	#print(gb_kv_cache_access_joules)
	#print("gb_other_access_joules:")
	#print(gb_other_access_joules)
	energy_efficiency_j_per_token = ((compute_energy_joules+sz_buffer_joules+gb_kv_cache_access_joules+gb_other_access_joules) * 1e-3) / token_count 
	throughput = token_count / total_time_seconds  
	latency = total_time_seconds * 1e3  # ms
	
	throughput_intra = token_count / total_time_seconds_intra
	latency_intra = total_time_seconds_intra * 1e3  # ms
	
	throughput_speedup_intra_original = throughput_intra / throughput
	latency_reduction_intra_original = ((latency - latency_intra) / latency) * 100 
	
	throughput_intra_inter = token_count / total_time_seconds_intra_inter
	latency_intra_inter = total_time_seconds_intra_inter * 1e3  # ms
	
	throughput_speedup_inter_intra = throughput_intra_inter / throughput_intra
	latency_reduction_inter_intra = ((latency_intra - latency_intra_inter) / latency_intra) * 100

	print("------------------Energy------------------")
	print(f"Compute energy: {compute_energy_joules:.6f} nJ")
	print(f"Memory energy: {buffer_energy_joules:.6f} nJ")
	print(f"- global buffer energy: {global_buffer_joules:.6f} nJ")
	print(f"---- weight access energy: {gb_weight_access_joules:.6f} nJ ({percent_gb_weight_access:.3f}%)")
	print(f"---- KV cache access energy: {gb_kv_cache_access_joules:.6f} nJ ({percent_gb_kv_cache_access:.3f}%)")
	print(f"---- other access energy: {gb_other_access_joules:.6f} nJ ({percent_gb_other_access:.3f}%)")
	print(f"- sz buffer energy: {sz_buffer_joules:.6f} nJ")
	print(f"Total energy: {total_energy_joules:.6f} nJ")
	print("------------------Performance------------------")
	print(f"Energy Efficiency (Token/J): {energy_efficiency_token_per_j:.2f} Token/J")
	print(f"Energy Efficiency (uJ/Token): {energy_efficiency_j_per_token:.2f} uJ/Token")
	print(f"Throughput (original): {throughput:.2f} Token/s")
	print(f"Latency (original): {latency:.6f} ms")
	print(f"Throughput (intra-pipeline): {throughput_intra:.2f} Token/s (speedup compared to original: {throughput_speedup_intra_original:.2f})")
	print(f"Latency (intra-pipeline): {latency_intra:.6f} ms (reduction compared to original: {latency_reduction_intra_original:.3f}%)")
	print(f"Throughput (intra-pipeline+inter-parallelism): {throughput_intra_inter:.2f} Token/s (speedup compared to intra-pipeline: {throughput_speedup_inter_intra:.2f})")
	print(f"Latency (intra-pipeline+inter-parallelism): {latency_intra_inter:.6f} ms (reduction compared to intra-pipeline: {latency_reduction_inter_intra:.3f}%)")
	
def simulate(model_dict: dict, config: dict, constants: dict, logs_dir: str, debug=False, plot_steps=100):
	accelerator = Accelerator(config, constants)
	print(f"{color.BOLD}Accelerator area (one core): {accelerator.total_area / 1e6 : 0.04f} mm\u00b2 (Total area: core_area×core_num ->{accelerator.total_area / 1e6 : 0.04f} × {config['layer_num']}){color.ENDC}")
	print(f"{color.BOLD}Accelerator frequency: {constants['clock_frequency']} MHz{color.ENDC}")
	print(f"{color.BOLD}Pruning enable: {config['pruning']}{color.ENDC}")
	print(f"{color.BOLD}Quantization enable: {config['quantization']}{color.ENDC}")
	
	area_components = {
		"Buffer Area": accelerator.buffer_area,
		"DCIM Area": accelerator.dcim_area,
		"Computing Engine Area": accelerator.computing_engine_area,
		"Pruning Unit Area": accelerator.pruning_area,
		"Quantization Unit Area": accelerator.quant_area,
		"Dequantization Unit Area": accelerator.dequant_area,
		"Softmax Area": accelerator.softmax_area,
		"LayerNorm Area": accelerator.layernorm_area,
		"ReLU Area": accelerator.relu_area,
	}
	
	power_components = {
		"Total Power": accelerator.total_power,
		"Buffer Power": accelerator.buffer_power,
		"DCIM Power": accelerator.dcim_power,
		"Computing Engine Power": accelerator.computing_engine_power,
		"Pruning Unit Power": accelerator.pruning_power,
		"Quantization Unit Power": accelerator.quant_power,
		"Dequantization Unit Power": accelerator.dequant_power,
		"Softmax Power": accelerator.softmax_power,
		"LayerNorm Power": accelerator.layernorm_power,
		"ReLU Power": accelerator.relu_power,
	}

	print("------------------Core-level Component-wise Area Breakdown------------------")
	for name, area in area_components.items():
		percentage = (area / accelerator.total_area) * 100 if accelerator.total_area != 0 else 0
		if name == 'Buffer Area':
			print(f"- {name}: {area / 1e6 : 0.04f} mm\u00b2 ({percentage:.3f}%) -> (global buffer: {config['global_buffer_size'] * 1000}KB, sz buffer: {config['sz_buffer_size'] * 1000}KB)")
		else:
			print(f"- {name}: {area / 1e6 : 0.04f} mm\u00b2 ({percentage:.3f}%)")
	
	print("------------------Core-level Component-wise Power Breakdown------------------")
	for name, power in power_components.items():
		percentage = (power / accelerator.total_power) * 100 if accelerator.total_power != 0 else 0
		if name == 'Total Power':
			print(f"- {name}: {power : 0.04f} mW")
		else:
			print(f"- {name}: {power : 0.04f} mW ({percentage:.3f}%)")
	
	print("------------------LLM configurations------------------")
	print(f"Prefill stage sequence length: {config['prefill_seq']}")
	print(f"Decode stage sequence length: {config['decode_seq']}")
	print(f"Layer number: {config['layer_num']}")
	print(f"Attention head number: {config['head_num']}")
	print(f"Hidden size: {config['hidden_size']}")
	print(f"FFN dimension: {config['ff_dim']}")
	# Get tiled ops from model dictionary
	memory_ops, compute_ops, num_ops = dict2ops(model_dict, config, tile_compute_ops=config['compute_ops_tiled'], tile_memory_ops=config['memory_ops_tiled'], debug=debug)
	#print('type(memory_ops):')
	#print(type(memory_ops))
	#print('type(compute_ops):')
	#print(type(compute_ops))
	#print('memory_ops:')
	#print(memory_ops)
	#print('compute_ops:')
	#print(compute_ops)
	#print(f"num_ops: {num_ops}")
	#print(num_ops)
	
	memory_op_idx, compute_op_idx, ops_done = [0, []], [0, []], 0
	
	compute_ops_batch_size = 1
	memory_ops_batch_size = 1
	
	# Get operations
	memory_op = get_op_list(memory_ops, memory_op_idx, memory_ops_batch_size)
	compute_op = get_op_list(compute_ops, compute_op_idx, compute_ops_batch_size)
	#print("memory_op:")
	#print(memory_op)
	#print("compute_op:")
	#print(compute_op)
	
	logs = {'area': accelerator.total_area / 1e6}

	sp_char = "\n\t"
	last_compute_ops = {}
	stalls = [0] * 4
	
	GB_weight_access = 0
	GB_KV_cache_access = 0
	GB_other_access = 0
	
	while not compute_ops[-1].done:

		print(f"Simulating accelerator at cycle: {accelerator.cycle*((config['prefill_seq']+config['decode_seq'])/16)}", end='\r')

		if debug: tqdm.write(f'{color.GREEN}Cycle: {accelerator.cycle + 1}{color.ENDC}')

		memory_stall, compute_stall = False, False
		new_stalls = [0] * 4
		
		if debug: 
			tqdm.write(f'{color.HEADER}Running memory operation(s) with name(s):\n\t{f"{sp_char}".join([f"- {op.op_name}" for op in memory_op if op])}\nand compute operation(s) with name(s):\n\t{f"{sp_char}".join(["- " + (f"{op.op_name}" if compute_ops_batch_size == 1 else f"{op[0].op_name} + " + str( compute_ops_batch_size - 1) + " more") for op in compute_op if op])}{color.ENDC}')
			
		if memory_op:
			#print("memory_op:")
			#print(memory_op)
			memory_stall = [None] * len(memory_op)
			debug_output = []

			head_ids = [i for i in range(len(memory_op))]

			for head_idx in head_ids:
				head_op = memory_op[head_idx]
				if head_op is None: continue

				data = head_op.convert_to_data()
				#print(f"data: {data.data_name}")

				last_compute_done, store_op = True, False
				if isinstance(head_op, (MemoryStoreOp, MemoryStoreTiledOp)): 
					if head_op.op_name in last_compute_ops:
						last_compute_op = last_compute_ops[head_op.op_name]
					else:
						last_compute_op = get_last_compute_op(head_op, head_idx, memory_op_idx, memory_ops, compute_ops)
						#print("last_compute_op:")
						#print(last_compute_op)
						#print("last_compute_op.done:")
						#print(last_compute_op.done)						
						#break
						last_compute_ops[head_op.op_name] = last_compute_op
					last_compute_done = last_compute_op.done
					
					store_op = True

				buffer = getattr(accelerator, f'{data.data_type}_buffer')

				if buffer.ready:
					prev_memory_op_done(head_op, head_idx, memory_op_idx, memory_ops)

					if buffer.can_store(data) and last_compute_done:
						memory_stall[head_idx] = False
					else:
						memory_stall[head_idx] = True
				else:
					memory_stall[head_idx] = True

				if memory_stall[head_idx] and debug:
					op_debug_output = []
					if not buffer.ready: 
						op_debug_output.append(f'{color.WARNING}Memory stall{f" for head {head_idx + 1}" if len(memory_op) > 1 else ""}: {buffer.buffer_type} buffer not ready{color.ENDC}')
						new_stalls[0] = max(1, new_stalls[0] + 1)
					if not last_compute_done: 
						op_debug_output.append(f'{color.WARNING}Memory stall{f" for head {head_idx + 1}" if len(memory_op) > 1 else ""}: waiting for last compute operation "{get_last_compute_op(head_op, head_idx, memory_op_idx, memory_ops, compute_ops).op_name}"{color.ENDC}')
						new_stalls[1] = max(1, new_stalls[1] + 1)
					debug_output.append("\n".join(op_debug_output))

				if not memory_stall[head_idx]:
					if store_op:
						if head_op.op_name.endswith('kv_quant-sm'):
							GB_KV_cache_access = GB_KV_cache_access + data.data_size
							buffer.store_in_main_memory(data)
						elif head_op.op_name.endswith('kv_quant-sim'):
							GB_KV_cache_access = GB_KV_cache_access + data.data_size
							buffer.store_in_main_memory(data)
						elif head_op.op_name.endswith('kv_quant-slm'):
							GB_KV_cache_access = GB_KV_cache_access + data.data_size
							buffer.store_in_main_memory(data)
						elif head_op.op_name.endswith('kv_prune-sm'):
							GB_KV_cache_access = GB_KV_cache_access + data.data_size
							buffer.store_in_main_memory(data)
						elif head_op.op_name.endswith('kv_prune-sim'):
							GB_KV_cache_access = GB_KV_cache_access + data.data_size
							buffer.store_in_main_memory(data)
						elif head_op.op_name.endswith('v-kvm'):
							GB_KV_cache_access = GB_KV_cache_access + data.data_size
							buffer.store_in_main_memory(data)
						else:
							removed_old_data = buffer.store(data)
					else:
						removed_old_data = buffer.load(data)
						if head_op.op_name.endswith('input'):
							GB_other_access = GB_other_access + data.data_size
							#print(f'input size: {data.data_size}')
						if head_op.op_name.endswith('q-l'):
							GB_weight_access = GB_weight_access + data.data_size
							#print(f'q-l size: {data.data_size}')
						elif head_op.op_name.endswith('k-l'):
							GB_weight_access = GB_weight_access + data.data_size
							#print(f'k-l size: {data.data_size}')
						elif head_op.op_name.endswith('v-l'):
							GB_weight_access = GB_weight_access + data.data_size
							#print(f'v-l size: {data.data_size}')
						elif head_op.op_name.endswith('o-l'):
							GB_weight_access = GB_weight_access + data.data_size
							#print(f'o-l size: {data.data_size}')
						elif head_op.op_name.endswith('f1-l'):
							GB_weight_access = GB_weight_access + data.data_size
							#print(f'f1-l size: {data.data_size}')
						elif head_op.op_name.endswith('f2-l'):
							GB_weight_access = GB_weight_access + data.data_size
							#print(f'f2-l size: {data.data_size}')
						elif head_op.op_name.endswith('kv_quant-l'):
							GB_KV_cache_access = GB_KV_cache_access + data.data_size
						elif head_op.op_name.endswith('kv_quant-li'):
							GB_KV_cache_access = GB_KV_cache_access + data.data_size
						elif head_op.op_name.endswith('kv_quant-ll'):
							GB_KV_cache_access = GB_KV_cache_access + data.data_size
						elif head_op.op_name.endswith('kv_prune-l'):
							GB_KV_cache_access = GB_KV_cache_access + data.data_size
						elif head_op.op_name.endswith('kv_prune-li'):
							GB_KV_cache_access = GB_KV_cache_access + data.data_size
						elif head_op.op_name.endswith('v-kvl'):
							GB_KV_cache_access = GB_KV_cache_access + data.data_size
						
			if debug and debug_output:
				tqdm.write("\n".join([debug_output[i] for i in head_ids if i < len(debug_output)]))
				
		ops_to_set_required = []
		if compute_op:
			compute_stall = [None] * len(compute_op)
			for head_idx, head_ops in enumerate(compute_op):
				if head_ops is None: continue

				if type(head_ops) != list:
					head_ops = [head_ops]

				if not accelerator.can_assign(head_ops):
					compute_stall[head_idx] = True
					if debug: 
						tqdm.write(f'{color.WARNING}Compute stall{f" for head {head_idx + 1}" if len(compute_op) > 1 else ""}: all resources are busy{color.ENDC}')
						new_stalls[2] = max(1, new_stalls[2] + 1)

				required_in_buffer_stall = False
				for head_op in head_ops:
					for data_name in head_op.required_in_buffer:
						if not accelerator.global_buffer.data_in_buffer(data_name) and not accelerator.sz_buffer.data_in_buffer(data_name):
							compute_stall[head_idx] = True
							required_in_buffer_stall = True
							new_stalls[3] = max(1, new_stalls[3] + 1)
							break
				
				if debug and required_in_buffer_stall: tqdm.write(f'{color.WARNING}Compute stall{f" for head {head_idx + 1}" if len(compute_op) > 1 else ""}: {data_name} required in buffer{color.ENDC}')
				
				#for head_op in head_ops:
				#	print("head_op.op_name:")
				#	print(head_op.op_name)
				
				if not compute_stall[head_idx]:
					for head_op in head_ops:
						assigned_op = accelerator.assign_op(head_op)
						assert assigned_op is True
						ops_to_set_required.append(head_op)
							
		(total_energy,
		global_buffer_energy,
		sz_buffer_energy,
		computing_engine_qk_energy,
		computing_engine_sv_energy,
		softmax_energy,
		layer_norm_pre_energy,
		layer_norm_aft_energy,
		attention_cim_q_energy,
		attention_cim_k_energy,
		attention_cim_v_energy,
		attention_cim_out_energy,
		fc1_cim_energy,
		fc2_cim_energy,
		relu_energy,
		pruning_unit_energy,
		quant_unit_energy,
		dequant_unit_energy) = accelerator.process_cycle(memory_ops, compute_ops, ops_to_set_required + compute_op)
		
		accelerator.cycle += 1

		stalls = [stalls[i] + new_stalls[i] for i in range(4)]
		
		if DO_LOGGING: logs = log_metrics(logs, 
										total_energy, 
										global_buffer_energy, 
										sz_buffer_energy, 
										computing_engine_qk_energy, 
										computing_engine_sv_energy, 
										softmax_energy, 
										layer_norm_pre_energy, 
										layer_norm_aft_energy, 
										attention_cim_q_energy, 
										attention_cim_k_energy, 
										attention_cim_v_energy, 
										attention_cim_out_energy, 
										fc1_cim_energy, 
										fc2_cim_energy, 
										relu_energy, 
										pruning_unit_energy, 
										quant_unit_energy, 
										dequant_unit_energy, 
										stalls, 
										logs_dir, 
										accelerator, 
										plot_steps=100)

		if debug:
			(computing_engine_qk_util, 
			computing_engine_sv_util, 
			softmax_util, 
			layernorm_pre_util, 
			layernorm_aft_util, 
			attention_cim_q_util, 
			attention_cim_k_util, 
			attention_cim_v_util, 
			attention_cim_out_util,
			fc1_cim_util,
			fc2_cim_util,
			relu_util,
			pruning_unit_util,
			quant_unit_util,
			dequant_unit_util,
			global_buffer_util,
			sz_buffer_util) = get_utilization(accelerator)

			tqdm.write(f'Global Buffer used: {global_buffer_util * 100.0 : 0.3f}%')
			tqdm.write(f'SZ Buffer used: {sz_buffer_util * 100.0 : 0.3f}%')
			tqdm.write(f'Computing Engine QK used: {computing_engine_qk_util * 100.0 : 0.3f}%')
			tqdm.write(f'Computing Engine SV used: {computing_engine_sv_util * 100.0 : 0.3f}%')
			tqdm.write(f'Attention CIM Q used: {attention_cim_q_util * 100.0 : 0.3f}%')
			tqdm.write(f'Attention CIM K used: {attention_cim_k_util * 100.0 : 0.3f}%')
			tqdm.write(f'Attention CIM V used: {attention_cim_v_util * 100.0 : 0.3f}%')
			tqdm.write(f'Attention CIM Out used: {attention_cim_out_util * 100.0 : 0.3f}%')
			tqdm.write(f'FC1 CIM used: {fc1_cim_util * 100.0 : 0.3f}%')
			tqdm.write(f'FC2 CIM used: {fc2_cim_util * 100.0 : 0.3f}%')
			tqdm.write(f'Pruning Unit used: {pruning_unit_util * 100.0 : 0.3f}%')
			tqdm.write(f'Quant Unit used: {quant_unit_util * 100.0 : 0.3f}%')
			tqdm.write(f'Dequant Unit used: {dequant_unit_util * 100.0 : 0.3f}%')
			tqdm.write(f'Softmax used: {softmax_util * 100.0 : 0.3f}%')
			tqdm.write(f'Layernorm Pre used: {layernorm_pre_util * 100.0 : 0.3f}%')
			tqdm.write(f'Layernorm Aft used: {layernorm_aft_util * 100.0 : 0.3f}%')
			tqdm.write(f'Relu used: {relu_util * 100.0 : 0.3f}%')

		memory_op_idx, ops_done = update_op_idx(memory_ops, memory_op_idx, memory_stall, memory_ops_batch_size, ops_done)
		compute_op_idx, ops_done = update_op_idx(compute_ops, compute_op_idx, compute_stall, compute_ops_batch_size, ops_done)

		memory_op, compute_op = get_op_list(memory_ops, memory_op_idx, memory_ops_batch_size), get_op_list(compute_ops, compute_op_idx, compute_ops_batch_size)

	#print(f"GB_weight_access: {GB_weight_access}")
	#print(f"GB_KV_cache_access: {GB_KV_cache_access}")
	#print(f"GB_other_access: {GB_other_access}")
	execution_cycles = extract_execution_cycles(logs_dir, constants)
	calculate_performance_metrics(logs_dir, accelerator, config, constants, GB_weight_access, GB_KV_cache_access, GB_other_access, execution_cycles)
	print(f'{color.BOLD}Finish simulation!{color.ENDC}')