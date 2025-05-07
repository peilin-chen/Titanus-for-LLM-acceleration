import os
import sys
import math
import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from ops import *
from tiled_ops import *
from modules import *
from attention_cim import AttentionCim
from fc1_cim import Fc1Cim
from fc2_cim import Fc2Cim
from buffer import Buffer

class Accelerator(object):
	"""Accelerator class
	
	Attributes:
		computing_engines_qk (list): list of ComputingEngine_QK objects
		computing_engines_sv (list): list of ComputingEngine_SV objects
		softmaxs (list): list of Softmax objects
		layer_norm_pre (Layernorm): Layernorm class object for layernorm_pre
		layer_norm_aft (Layernorm): Layernorm class object for layernorm_aft
		attention_cim_q (AttentionCim): AttentionCim class object for q
		attention_cim_k (AttentionCim): AttentionCim class object for k
		attention_cim_v (AttentionCim): AttentionCim class object for v
		attention_cim_out (AttentionCim): AttentionCim class object for output
		fc1_cim (Fc1Cim): Fc1Cim class object for fc1
		fc2_cim (Fc2Cim): Fc1Cim class object for fc2
		relu (ReLu): ReLu class object for relu
		global_buffer (Buffer): Buffer class object for global
		sz_buffer (Buffer): Buffer class object for sz
		pruning_unit (PruningUnit): PruningUnit class object for pruning
		quant_unit (QuantizationUnit): QuantizationUnit class object for quantization
		dequant_unit (DequantizationUnit): DequantizationUnit class object for dequantization
	"""
	def __init__(self, config, constants):
		self.config = config
		self.constants = constants

		self.computing_engines_qk = []
		for i in range(self.config['computing_engine_num']):
			self.computing_engines_qk.append(ComputingEngine_QK(f'ce_qk{i+1}', config, constants))
			
		self.computing_engines_sv = []
		for i in range(self.config['computing_engine_num']):
			self.computing_engines_sv.append(ComputingEngine_SV(f'ce_sv{i+1}', config, constants))

		self.softmaxs = []
		for i in range(self.config['softmax_num']):
			self.softmaxs.append(Softmax(f'softmax{i+1}', config, constants))
		
		self.layer_norm_pres = []
		for i in range(self.config['layernorm_parallelism']):
			self.layer_norm_pres.append(Layernorm(f'pre_norm{i+1}', config, constants))
		
		self.layer_norm_afts = []
		for i in range(self.config['layernorm_parallelism']):
			self.layer_norm_afts.append(Layernorm(f'after_norm{i+1}', config, constants))
		
		self.attention_cim_q = AttentionCim('attention_cim_q', config, constants)
		
		self.attention_cim_k = AttentionCim('attention_cim_k', config, constants)
		
		self.attention_cim_v = AttentionCim('attention_cim_v', config, constants)

		self.attention_cim_out = AttentionCim('attention_cim_out', config, constants)

		self.fc1_cim = Fc1Cim('fc1_cim', config, constants)

		self.fc2_cim = Fc2Cim('fc2_cim', config, constants)

		self.relu = ReLu('relu', config, constants)
		
		self.pruning_units = []
		for i in range(self.config['pruning_unit_parallelism']):
			self.pruning_units.append(PruningUnit(f'pruning_unit{i+1}', config, constants))
		
		self.quant_units = []
		for i in range(self.config['quantization_unit_parallelism']):
			self.quant_units.append(QuantizationUnit(f'quant_unit{i+1}', config, constants))
		
		self.dequant_units = []
		for i in range(self.config['dequantization_unit_parallelism']):
			self.dequant_units.append(DequantizationUnit(f'dequant_unit{i+1}', config, constants))
		
		self.global_buffer = Buffer('global', config, constants)

		self.sz_buffer = Buffer('sz', config, constants)

		self.total_area = 0
		self.dcim_area = 0
		self.layernorm_area = 0
		self.softmax_area = 0
		self.relu_area = 0
		self.buffer_area = 0
		self.computing_engine_area = 0
		self.pruning_area = 0
		self.quant_area = 0
		self.dequant_area = 0
		
		self.total_power = 0
		self.dcim_power = 0
		self.layernorm_power = 0
		self.softmax_power = 0
		self.relu_power = 0
		self.buffer_power = 0
		self.computing_engine_power = 0
		self.pruning_power = 0
		self.quant_power = 0
		self.dequant_power = 0
		
		for ceqk in self.computing_engines_qk:
			self.computing_engine_area += ceqk.area
			self.computing_engine_power += (ceqk.dynamic_power+ceqk.leakage_power)
		
		for cesv in self.computing_engines_sv:
			self.computing_engine_area += cesv.area
			self.computing_engine_power += (cesv.dynamic_power+cesv.leakage_power)
		
		for sftm in self.softmaxs:
			self.softmax_area += sftm.area
			self.softmax_power += (sftm.dynamic_power+sftm.leakage_power)
		
		for ln_pre in self.layer_norm_pres:
			self.layernorm_area += ln_pre.area
			self.layernorm_power += (ln_pre.dynamic_power+ln_pre.leakage_power)
		
		for ln_aft in self.layer_norm_afts:
			self.layernorm_area += ln_aft.area
			self.layernorm_power += (ln_aft.dynamic_power+ln_aft.leakage_power)
			
		for pu in self.pruning_units:
			self.pruning_area += pu.area
			self.pruning_power += (pu.dynamic_power + pu.leakage_power)
			
		for qu in self.quant_units:
			self.quant_area += qu.area
			self.quant_power += (qu.dynamic_power + qu.leakage_power)
			
		for dqu in self.dequant_units:
			self.dequant_area += dqu.area
			self.dequant_power += (dqu.dynamic_power + dqu.leakage_power)
		
		self.dcim_area = self.dcim_area + self.attention_cim_q.area + self.attention_cim_k.area + self.attention_cim_v.area 
		self.dcim_power = self.dcim_power + self.attention_cim_q.power + self.attention_cim_k.power + self.attention_cim_v.power 
		
		self.dcim_area = self.dcim_area + self.attention_cim_out.area + self.fc1_cim.area + self.fc2_cim.area
		self.dcim_power = self.dcim_power + self.attention_cim_out.power + self.fc1_cim.power + self.fc2_cim.power
		
		self.relu_area = self.relu_area + self.relu.area
		self.relu_power = self.relu_power + self.relu.dynamic_power + self.relu.leakage_power
		
		self.buffer_area = self.buffer_area + self.global_buffer.area + self.sz_buffer.area 
		self.buffer_power = self.buffer_power + self.global_buffer.dynamic_power + self.global_buffer.leakage_power 
		self.buffer_power = self.buffer_power + self.sz_buffer.dynamic_power + self.sz_buffer.leakage_power
		
		#self.pruning_area = self.pruning_area + self.pruning_unit.area
		#self.pruning_power = self.pruning_power + self.pruning_unit.dynamic_power + self.pruning_unit.leakage_power
		
		#self.quant_area = self.quant_area + self.quant_unit.area
		#self.quant_power = self.quant_power + self.quant_unit.dynamic_power + self.quant_unit.leakage_power
		
		#self.dequant_area = self.dequant_area + self.dequant_unit.area
		#self.dequant_power = self.dequant_power + self.dequant_unit.dynamic_power + self.dequant_unit.leakage_power

		self.total_area = self.computing_engine_area+self.softmax_area+self.layernorm_area+self.dcim_area+self.relu_area+self.buffer_area+self.pruning_area+self.quant_area+self.dequant_area
		self.total_power = self.computing_engine_power+self.softmax_power+self.layernorm_power+self.dcim_power+self.relu_power+self.buffer_power+self.pruning_power+self.quant_power+self.dequant_power
		
		self.cycle = 0
		self.idx_done = 0

	def set_required(self, compute_op):
		for data_name in compute_op.required_in_buffer:
			if self.global_buffer.data_in_buffer(data_name):
				self.global_buffer.get_data(data_name).required_in_buffer = True
			if self.sz_buffer.data_in_buffer(data_name):
				self.sz_buffer.get_data(data_name).required_in_buffer = True

	def set_not_required(self, compute_op):
		assert compute_op.done is True
		for data_name in compute_op.required_in_buffer:
			if self.global_buffer.data_in_buffer(data_name):
				self.global_buffer.get_data(data_name).required_in_buffer = False
			if self.sz_buffer.data_in_buffer(data_name):
				self.sz_buffer.get_data(data_name).required_in_buffer = False
	
	def all_resources_free(self):
		for ceqk in self.computing_engines_qk:
			if not ceqk.ready: return False
		
		for cesv in self.computing_engines_sv:
			if not cesv.ready: return False
			
		for sftm in self.softmaxs:
			if not sftm.ready: return False
		
		for ln_pre in self.layer_norm_pres:
			if not ln_pre.ready: return False
		
		for ln_aft in self.layer_norm_afts:
			if not ln_aft.ready: return False
		
		# check attention_cim_q
		for cim_macro_q in self.attention_cim_q.cim_macros:
			if not cim_macro_q.ready: return False
		for accumulator_q in self.attention_cim_q.accumulators:
			if not accumulator_q.ready: return False
			
		# check attention_cim_k
		for cim_macro_k in self.attention_cim_k.cim_macros:
			if not cim_macro_k.ready: return False
		for accumulator_k in self.attention_cim_k.accumulators:
			if not accumulator_k.ready: return False
			
		# check attention_cim_v
		for cim_macro_v in self.attention_cim_v.cim_macros:
			if not cim_macro_v.ready: return False
		for accumulator_v in self.attention_cim_v.accumulators:
			if not accumulator_v.ready: return False
	
		# check attention_cim_out
		for cim_macro_out in self.attention_cim_out.cim_macros:
			if not cim_macro_out.ready: return False
		for accumulator_out in self.attention_cim_out.accumulators:
			if not accumulator_out.ready: return False
	
		# check fc1_cim
		for cim_macro_fc1 in self.fc1_cim.cim_macros:
			if not cim_macro_fc1.ready: return False
		for accumulator_fc1 in self.fc1_cim.accumulators:
			if not accumulator_fc1.ready: return False
			
		# check fc2_cim
		for cim_macro_fc2 in self.fc2_cim.cim_macros:
			if not cim_macro_fc2.ready: return False
		for accumulator_fc2 in self.fc2_cim.accumulators:
			if not accumulator_fc2.ready: return False
		
		for pu in self.pruning_units:
			if not pu.ready: return False
		
		for qu in self.quant_units:
			if not qu.ready: return False
		
		for dqu in self.dequant_units:
			if not dqu.ready: return False
		
		if not self.relu.ready: return False
		
		return True
		
	def num_computing_engines_qk_free(self):
		num_computing_engines_qk, num_free = 0, 0
		
		for ceqk in self.computing_engines_qk:
			num_computing_engines_qk += 1
			if ceqk.ready: 
				num_free += 1
				
		return num_free, num_computing_engines_qk

	def num_computing_engines_sv_free(self):
		num_computing_engines_sv, num_free = 0, 0
		
		for cesv in self.computing_engines_sv:
			num_computing_engines_sv += 1
			if cesv.ready: 
				num_free += 1
				
		return num_free, num_computing_engines_sv
		
	def num_softmaxs_free(self):
		num_softmaxs, num_free = 0, 0
		
		for sftm in self.softmaxs:
			num_softmaxs += 1
			if sftm.ready: 
				num_free += 1
				
		return num_free, num_softmaxs
		
	def num_layer_norm_pre_free(self):
		num_layer_norm_pre, num_free = 0, 0
		
		for ln_pre in self.layer_norm_pres:
			num_layer_norm_pre += 1
			if ln_pre.ready:
				num_free += 1
			
		return num_free, num_layer_norm_pre
	
	def num_layer_norm_aft_free(self):
		num_layer_norm_aft, num_free = 0, 0
		
		for ln_aft in self.layer_norm_afts:
			num_layer_norm_aft += 1
			if ln_aft.ready:
				num_free += 1
				
		return num_free, num_layer_norm_aft
	
	def num_attention_cim_q_free(self):
		num_attention_cim_q, num_free = 0, 0
		
		for cim_macro_q in self.attention_cim_q.cim_macros:
			if not cim_macro_q.ready:
				num_attention_cim_q, num_free = 1, 0
				return num_free, num_attention_cim_q

		for accumulator_q in self.attention_cim_q.accumulators:
			if not accumulator_q.ready:
				num_attention_cim_q, num_free = 1, 0
				return num_free, num_attention_cim_q
		
		num_attention_cim_q, num_free = 1, 1
		return num_free, num_attention_cim_q
		
	def num_attention_cim_k_free(self):
		num_attention_cim_k, num_free = 0, 0
		
		for cim_macro_k in self.attention_cim_k.cim_macros:
			if not cim_macro_k.ready:
				num_attention_cim_k, num_free = 1, 0
				return num_free, num_attention_cim_k

		for accumulator_k in self.attention_cim_k.accumulators:
			if not accumulator_k.ready:
				num_attention_cim_k, num_free = 1, 0
				return num_free, num_attention_cim_k
		
		num_attention_cim_k, num_free = 1, 1
		return num_free, num_attention_cim_k
		
	def num_attention_cim_v_free(self):
		num_attention_cim_v, num_free = 0, 0
		
		for cim_macro_v in self.attention_cim_v.cim_macros:
			if not cim_macro_v.ready:
				num_attention_cim_v, num_free = 1, 0
				return num_free, num_attention_cim_v

		for accumulator_v in self.attention_cim_v.accumulators:
			if not accumulator_v.ready:
				num_attention_cim_v, num_free = 1, 0
				return num_free, num_attention_cim_v
		
		num_attention_cim_v, num_free = 1, 1
		return num_free, num_attention_cim_v
		
	def num_attention_cim_out_free(self):
		num_attention_cim_out, num_free = 0, 0
		
		for cim_macro_out in self.attention_cim_out.cim_macros:
			if not cim_macro_out.ready:
				num_attention_cim_out, num_free = 1, 0
				return num_free, num_attention_cim_out

		for accumulator_out in self.attention_cim_out.accumulators:
			if not accumulator_out.ready:
				num_attention_cim_out, num_free = 1, 0
				return num_free, num_attention_cim_out
		
		num_attention_cim_out, num_free = 1, 1
		return num_free, num_attention_cim_out
		
	def num_fc1_cim_free(self):
		num_fc1_cim, num_free = 0, 0
		
		for cim_macro_fc1 in self.fc1_cim.cim_macros:
			if not cim_macro_fc1.ready:
				num_fc1_cim, num_free = 1, 0
				return num_free, num_fc1_cim

		for accumulator_fc1 in self.fc1_cim.accumulators:
			if not accumulator_fc1.ready:
				num_fc1_cim, num_free = 1, 0
				return num_free, num_fc1_cim
		
		num_fc1_cim, num_free = 1, 1
		return num_free, num_fc1_cim
		
	def num_fc2_cim_free(self):
		num_fc2_cim, num_free = 0, 0
		
		for cim_macro_fc2 in self.fc2_cim.cim_macros:
			if not cim_macro_fc2.ready:
				num_fc2_cim, num_free = 1, 0
				return num_free, num_fc2_cim

		for accumulator_fc2 in self.fc2_cim.accumulators:
			if not accumulator_fc2.ready:
				num_fc2_cim, num_free = 1, 0
				return num_free, num_fc2_cim
		
		num_fc2_cim, num_free = 1, 1
		return num_free, num_fc2_cim
		
	def num_relu_free(self):
		num_relu, num_free = 0, 0
		
		if not self.relu.ready: 
			num_relu, num_free = 1, 0
			return num_free, num_relu
				
		num_relu, num_free = 1, 1
		return num_free, num_relu
	
	def num_pruning_unit_free(self):
		num_pruning_unit, num_free = 0, 0
		
		for pu in self.pruning_units:
			num_pruning_unit += 1
			if pu.ready:
				num_free += 1
		
		return num_free, num_pruning_unit
		
	def num_quant_unit_free(self):
		num_quant_unit, num_free = 0, 0
		
		for qu in self.quant_units:
			num_quant_unit += 1
			if qu.ready:
				num_free += 1

		return num_free, num_quant_unit
		
	def num_dequant_unit_free(self):
		num_dequant_unit, num_free = 0, 0
		
		for dqu in self.dequant_units:
			num_dequant_unit += 1
			if dqu.ready:
				num_free += 1
		
		return num_free, num_dequant_unit
	
	def process_cycle(self, memory_ops, compute_ops, ops_to_set_required):
	
		total_computing_engine_qk_energy = [0, 0]
		for ceqk in self.computing_engines_qk:
			computing_engine_qk_energy = ceqk.process_cycle()
			total_computing_engine_qk_energy[0] += computing_engine_qk_energy[0]
			total_computing_engine_qk_energy[1] += computing_engine_qk_energy[1]
			
		total_computing_engine_sv_energy = [0, 0]
		for cesv in self.computing_engines_sv:
			computing_engine_sv_energy = cesv.process_cycle()
			total_computing_engine_sv_energy[0] += computing_engine_sv_energy[0]
			total_computing_engine_sv_energy[1] += computing_engine_sv_energy[1]
			
		total_softmax_energy = [0, 0]
		for sftm in self.softmaxs:
			softmax_energy = sftm.process_cycle()
			total_softmax_energy[0] += softmax_energy[0]
			total_softmax_energy[1] += softmax_energy[1]
		
		total_layer_norm_pre_energy = [0, 0]
		for ln_pre in self.layer_norm_pres:
			layer_norm_pre_energy = ln_pre.process_cycle()
			total_layer_norm_pre_energy[0] += layer_norm_pre_energy[0]
			total_layer_norm_pre_energy[1] += layer_norm_pre_energy[1]
		
		total_layer_norm_aft_energy = [0, 0]
		for ln_aft in self.layer_norm_afts:
			layer_norm_aft_energy = ln_aft.process_cycle()
			total_layer_norm_aft_energy[0] += layer_norm_aft_energy[0]
			total_layer_norm_aft_energy[1] += layer_norm_aft_energy[1]
		
		attention_cim_q_energy = self.attention_cim_q.process_cycle()
		attention_cim_k_energy = self.attention_cim_k.process_cycle()
		attention_cim_v_energy = self.attention_cim_v.process_cycle()
		attention_cim_out_energy = self.attention_cim_out.process_cycle()
		fc1_cim_energy = self.fc1_cim.process_cycle()
		fc2_cim_energy = self.fc2_cim.process_cycle()
		relu_energy = self.relu.process_cycle()
		#pruning_unit_energy = self.pruning_unit.process_cycle()
		#quant_unit_energy = self.quant_unit.process_cycle()
		#dequant_unit_energy = self.dequant_unit.process_cycle()
		
		pruning_unit_energy = [0, 0]
		for pu in self.pruning_units:
			pu_energy = pu.process_cycle()
			pruning_unit_energy[0] += pu_energy[0]
			pruning_unit_energy[1] += pu_energy[1]
			
		quant_unit_energy = [0, 0]
		for qu in self.quant_units:
			qu_energy = qu.process_cycle()
			quant_unit_energy[0] += qu_energy[0]
			quant_unit_energy[1] += qu_energy[1]
			
		dequant_unit_energy = [0, 0]
		for dqu in self.dequant_units:
			dqu_energy = dqu.process_cycle()
			dequant_unit_energy[0] += dqu_energy[0]
			dequant_unit_energy[1] += dqu_energy[1]
		
		total_energy = [0, 0]
		
		total_energy[0] = total_energy[0] + total_computing_engine_qk_energy[0] + total_computing_engine_sv_energy[0] + total_softmax_energy[0]
		total_energy[0] = total_energy[0] + total_layer_norm_pre_energy[0] + total_layer_norm_aft_energy[0] + attention_cim_q_energy[0]
		total_energy[0] = total_energy[0] + attention_cim_k_energy[0] + attention_cim_v_energy[0] + attention_cim_out_energy[0]
		total_energy[0] = total_energy[0] + fc1_cim_energy[0] + fc2_cim_energy[0] + relu_energy[0]
		total_energy[0] = total_energy[0] + pruning_unit_energy[0] + quant_unit_energy[0] + dequant_unit_energy[0]
		
		total_energy[1] = total_energy[1] + total_computing_engine_qk_energy[1] + total_computing_engine_sv_energy[1] + total_softmax_energy[1]
		total_energy[1] = total_energy[1] + total_layer_norm_pre_energy[1] + total_layer_norm_aft_energy[1] + attention_cim_q_energy[1]
		total_energy[1] = total_energy[1] + attention_cim_k_energy[1] + attention_cim_v_energy[1] + attention_cim_out_energy[1]
		total_energy[1] = total_energy[1] + fc1_cim_energy[1] + fc2_cim_energy[1] + relu_energy[1]
		total_energy[1] = total_energy[1] + pruning_unit_energy[1] + quant_unit_energy[1] + dequant_unit_energy[1]
		
		global_buffer_energy = self.global_buffer.process_cycle()
		sz_buffer_energy = self.sz_buffer.process_cycle()
		
		for idx, compute_op in enumerate(compute_ops):
			if idx < self.idx_done:
				continue
			if type(compute_op) == list:
				for head_ops in compute_op:
					for head_idx, head_op in enumerate(head_ops):
						if head_op.done == True: 
							self.set_not_required(head_op)
						else:
							break
			else:
				if compute_op.done == True:
					self.set_not_required(compute_op)
					self.idx_done = idx
				else:
					break

		for op in ops_to_set_required:
			if type(op) == list:
				for head_op in op:
					self.set_required(head_op)
			elif op is not None:
				self.set_required(op)
				
		# All energy in nJ
		return tuple(total_energy), global_buffer_energy, sz_buffer_energy, total_computing_engine_qk_energy, total_computing_engine_sv_energy, total_softmax_energy, total_layer_norm_pre_energy, total_layer_norm_aft_energy, attention_cim_q_energy, attention_cim_k_energy, attention_cim_v_energy, attention_cim_out_energy, fc1_cim_energy, fc2_cim_energy, relu_energy, pruning_unit_energy, quant_unit_energy, dequant_unit_energy

	def can_assign(self, op_list):
		assert type(op_list) == list
		
		num_computing_engines_qk_free, num_computing_engines_qk = self.num_computing_engines_qk_free()
		num_computing_engines_sv_free, num_computing_engines_sv = self.num_computing_engines_sv_free()
		num_softmax_free, num_softmax = self.num_softmaxs_free()
		num_layer_norm_pre_free, num_layer_norm_pre = self.num_layer_norm_pre_free()
		num_layer_norm_aft_free, num_layer_norm_aft = self.num_layer_norm_aft_free()
		num_attention_cim_q_free, num_attention_cim_q = self.num_attention_cim_q_free()
		num_attention_cim_k_free, num_attention_cim_k = self.num_attention_cim_k_free()
		num_attention_cim_v_free, num_attention_cim_v = self.num_attention_cim_v_free()
		num_attention_cim_out_free, num_attention_cim_out = self.num_attention_cim_out_free()
		num_fc1_cim_free, num_fc1_cim = self.num_fc1_cim_free()
		num_fc2_cim_free, num_fc2_cim = self.num_fc2_cim_free()
		num_relu_free, num_relu = self.num_relu_free()
		num_pruning_unit_free, num_pruning_unit = self.num_pruning_unit_free()
		num_quant_unit_free, num_quant_unit = self.num_quant_unit_free()
		num_dequant_unit_free, num_dequant_unit = self.num_dequant_unit_free()
				
		num_computing_engines_qk_to_assign = 0
		num_computing_engines_sv_to_assign = 0
		num_softmax_to_assign = 0
		num_layer_norm_pre_to_assign = 0
		num_layer_norm_aft_to_assign = 0
		num_attention_cim_q_to_assign = 0
		num_attention_cim_k_to_assign = 0
		num_attention_cim_v_to_assign = 0
		num_attention_cim_out_to_assign = 0
		num_fc1_cim_to_assign = 0
		num_fc2_cim_to_assign = 0
		num_relu_to_assign = 0
		num_pruning_unit_to_assign = 0
		num_quant_unit_to_assign = 0
		num_dequant_unit_to_assign = 0
		
		for op in op_list:
			assert op.compute_op is True
			
			if isinstance(op, (QKVVMultOp, QKVVMultTiledOp)):
				num_computing_engines_qk_to_assign += self.config['computing_engine_num']
			elif isinstance(op, (SVVVMultOp, SVVVMultTiledOp)):
				num_computing_engines_sv_to_assign += self.config['computing_engine_num']
			elif isinstance(op, (QAttCimVecMatMultOp, QAttCimVecMatMultTiledOp)):
				num_attention_cim_q_to_assign += 1
			elif isinstance(op, (KAttCimVecMatMultOp, KAttCimVecMatMultTiledOp)):
				num_attention_cim_k_to_assign += 1
			elif isinstance(op, (VAttCimVecMatMultOp, VAttCimVecMatMultTiledOp)):
				num_attention_cim_v_to_assign += 1
			elif isinstance(op, (OAttCimVecMatMultOp, OAttCimVecMatMultTiledOp)):
				num_attention_cim_out_to_assign += 1
			elif isinstance(op, (Fc1CimVecMatMultOp, Fc1CimVecMatMultTiledOp)):
				num_fc1_cim_to_assign += 1
			elif isinstance(op, (Fc2CimVecMatMultOp, Fc2CimVecMatMultTiledOp)):
				num_fc2_cim_to_assign += 1
			elif isinstance(op, (PreLayerNormOp, PreLayerNormTiledOp)):
				num_layer_norm_pre_to_assign += self.config['layernorm_parallelism']
			elif isinstance(op, (AftLayerNormOp, AftLayerNormTiledOp)):
				num_layer_norm_aft_to_assign += self.config['layernorm_parallelism']
			elif isinstance(op, (ReLuOp, ReLuTiledOp)):
				num_relu_to_assign += 1
			elif isinstance(op, (SoftmaxOp, SoftmaxTiledOp)):
				num_softmax_to_assign += self.config['softmax_num']
			elif isinstance(op, (PruningOp, PruningTiledOp)):
				num_pruning_unit_to_assign += self.config['pruning_unit_parallelism']
			elif isinstance(op, (QuantOp, QuantTiledOp)):
				num_quant_unit_to_assign += self.config['quantization_unit_parallelism']
			elif isinstance(op, (DequantOp, DequantTiledOp)):
				num_dequant_unit_to_assign += self.config['dequantization_unit_parallelism']

		if num_computing_engines_qk_free < num_computing_engines_qk_to_assign:
			return False

		if num_computing_engines_sv_free < num_computing_engines_sv_to_assign:
			return False

		if num_attention_cim_q_free < num_attention_cim_q_to_assign or num_attention_cim_k_free < num_attention_cim_k_to_assign:
			return False

		if num_attention_cim_v_free < num_attention_cim_v_to_assign or num_attention_cim_out_free < num_attention_cim_out_to_assign:
			return False

		if num_fc1_cim_free < num_fc1_cim_to_assign or num_fc2_cim_free < num_fc2_cim_to_assign or num_layer_norm_pre_free < num_layer_norm_pre_to_assign:
			return False

		if num_layer_norm_aft_free < num_layer_norm_aft_to_assign or num_relu_free < num_relu_to_assign or num_pruning_unit_free < num_pruning_unit_to_assign:
			return False

		if num_quant_unit_free < num_quant_unit_to_assign or num_dequant_unit_free < num_dequant_unit_to_assign or num_softmax_free < num_softmax_to_assign:
			return False

		return True
	
	def assign_op(self, op):
		assert op.compute_op is True
		assigned_op = False
		
		if isinstance(op, (QKVVMultOp, QKVVMultTiledOp)):
			for ceqk in self.computing_engines_qk:
				if ceqk.ready:
					ceqk.assign_op(op)
					assigned_op = True
		elif isinstance(op, (SVVVMultOp, SVVVMultTiledOp)):
			for cesv in self.computing_engines_sv:
				if cesv.ready:
					cesv.assign_op(op)
					assigned_op = True
		elif isinstance(op, (QAttCimVecMatMultOp, QAttCimVecMatMultTiledOp)):
			assigned_op = self.attention_cim_q.assign_op(op)
		elif isinstance(op, (KAttCimVecMatMultOp, KAttCimVecMatMultTiledOp)):
			assigned_op = self.attention_cim_k.assign_op(op)
		elif isinstance(op, (VAttCimVecMatMultOp, VAttCimVecMatMultTiledOp)):
			assigned_op = self.attention_cim_v.assign_op(op)
		elif isinstance(op, (OAttCimVecMatMultOp, OAttCimVecMatMultTiledOp)):
			assigned_op = self.attention_cim_out.assign_op(op)
		elif isinstance(op, (Fc1CimVecMatMultOp, Fc1CimVecMatMultTiledOp)):
			assigned_op = self.fc1_cim.assign_op(op)
		elif isinstance(op, (Fc2CimVecMatMultOp, Fc2CimVecMatMultTiledOp)):
			assigned_op = self.fc2_cim.assign_op(op)
		elif isinstance(op, (PreLayerNormOp, PreLayerNormTiledOp)):
			for ln_pre in self.layer_norm_pres:
				if ln_pre.ready:
					ln_pre.assign_op(op)
					assigned_op = True
		elif isinstance(op, (AftLayerNormOp, AftLayerNormTiledOp)):
			for ln_aft in self.layer_norm_afts:
				if ln_aft.ready:
					ln_aft.assign_op(op)
					assigned_op = True
		elif isinstance(op, (ReLuOp, ReLuTiledOp)):
			if self.relu.ready:
				self.relu.assign_op(op)
				assigned_op = True
		elif isinstance(op, (SoftmaxOp, SoftmaxTiledOp)):
			for sftm in self.softmaxs:
				if sftm.ready:
					sftm.assign_op(op)
					assigned_op = True
		elif isinstance(op, (PruningOp, PruningTiledOp)):
			for pu in self.pruning_units:
				if pu.ready:
					pu.assign_op(op)
					assigned_op = True
		elif isinstance(op, (QuantOp, QuantTiledOp)):
			for qu in self.quant_units:
				if qu.ready:
					qu.assign_op(op)
					assigned_op = True
		elif isinstance(op, (DequantOp, DequantTiledOp)):
			for dqu in self.dequant_units:
				if dqu.ready:
					dqu.assign_op(op)
					assigned_op = True
		else:
			raise ValueError(f'Invalid operation: {op.op_name} of type: {type(op)}')
			
		return assigned_op

