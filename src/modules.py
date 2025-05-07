import re
import math
import inspect
from ops import *
from tiled_ops import *

class Module(object):
	"""Parent module class
	
	Attributes:
		module_name (str): name of the given module
		dynamic_power (float): dynamic power consumption in mW
		leakage_power (float): leakage power consumption in mW
		area (float): silicon area in um^2
		process_cycles (int): number of cycles to process an input tile
		ready (bool): if the module is ready for new input
	"""
	def __init__(self, module_name, dynamic_power, leakage_power, area, clock_frequency, sparsity):
		self.module_name = module_name
		self.dynamic_power = dynamic_power
		self.leakage_power = leakage_power
		self.area = area
		self.clock_frequency = clock_frequency
		self.sparsity = sparsity
		self.process_cycles = None 
		self.buffer_energy = 0
		self.ready = True

	def process_cycle(self):
		if self.process_cycles is None or self.process_cycles == 0:
			self.ready = True
		else:
			self.process_cycles -= 1
			self.ready = False

		submodule_dynamic_energy, submodule_leakage_energy = 0, 0

		for member in inspect.getmembers(self):
			if isinstance(member, Module):
				dyn, leak = member.process_cycle()
				submodule_dynamic_energy += dyn; submodule_leakage_energy += leak
				if not member.ready: self.ready = False

		if self.ready and self.assigned_op is not None:
			self.assigned_op.done = True

		if self.ready:
			return (0, 0) 
		else:
			return ((1-self.sparsity)*(self.dynamic_power / self.clock_frequency + submodule_dynamic_energy + self.buffer_energy), (self.leakage_power / self.clock_frequency + submodule_leakage_energy)) # unit: nJ
		
class ComputingEngine_QK(Module):
	def __init__(self, module_name, config, constants):
		
		if (config['hidden_size']/config['head_num']) >= config['tile_computing_engine']['tile_y']:
			if config['pruning']:
				sparsity = config['sparsity']['before_quant']
			else:
				sparsity = 0
		else:
			if config['pruning']:
				original_sparsity = (config['tile_computing_engine']['tile_y']-(config['hidden_size']/config['head_num'])) / config['tile_computing_engine']['tile_y']
				sparsity = original_sparsity + (1-original_sparsity)*config['sparsity']['before_quant']
			else:
				sparsity = (config['tile_computing_engine']['tile_y']-(config['hidden_size']/config['head_num'])) / config['tile_computing_engine']['tile_y']
		
		Module.__init__(self, module_name, constants['computing_engine']['dynamic'], constants['computing_engine']['leakage'], constants['computing_engine']['area'], constants['clock_frequency'], sparsity)
		self.assigned_op = None

	def assign_op(self, op):
		self.process_cycles = 3
		self.ready = False

		self.assigned_op = op

class ComputingEngine_SV(Module):
	def __init__(self, module_name, config, constants):
		
		if (config['prefill_seq']+config['decode_seq']) >= config['tile_computing_engine']['tile_y']:
			if config['pruning'] and config['quantization']:
				sparsity = config['sparsity']['after_dequant']*0.5+0.5
			elif config['quantization']:
				sparsity = (config['sparsity']['after_dequant']-config['sparsity']['before_quant'])*0.5+0.5
			elif config['pruning']:
				sparsity = config['sparsity']['before_quant']*0.5+0.5
			else:
				sparsity = 0
		else:
			if config['pruning'] and config['quantization']:
				original_sparsity = (config['tile_computing_engine']['tile_y']-(config['prefill_seq']+config['decode_seq'])) / config['tile_computing_engine']['tile_y']
				sparsity = original_sparsity + (1-original_sparsity)*config['sparsity']['after_dequant']
			elif config['quantization']:
				original_sparsity = (config['tile_computing_engine']['tile_y']-(config['prefill_seq']+config['decode_seq'])) / config['tile_computing_engine']['tile_y']
				sparsity = original_sparsity + (1-original_sparsity)*(config['sparsity']['after_dequant']-config['sparsity']['before_quant'])
			elif config['pruning']:
				original_sparsity = (config['tile_computing_engine']['tile_y']-(config['prefill_seq']+config['decode_seq'])) / config['tile_computing_engine']['tile_y']
				sparsity = original_sparsity + (1-original_sparsity)*config['sparsity']['before_quant']
			else:
				sparsity = (config['tile_computing_engine']['tile_y']-(config['prefill_seq']+config['decode_seq'])) / config['tile_computing_engine']['tile_y']
		
		Module.__init__(self, module_name, constants['computing_engine']['dynamic'], constants['computing_engine']['leakage'], constants['computing_engine']['area'], constants['clock_frequency'], sparsity) 
		self.assigned_op = None

	def assign_op(self, op):
		self.process_cycles = 3
		self.ready = False

		self.assigned_op = op

class PruningUnit(Module):
	def __init__(self, module_name, config, constants):
		Module.__init__(self, module_name, constants['pruning_unit']['dynamic'], constants['pruning_unit']['leakage'], constants['pruning_unit']['area'], constants['clock_frequency'], 0)
		self.assigned_op = None

	def assign_op(self, op):
		self.process_cycles = 2
		self.ready = False

		self.assigned_op = op

class QuantizationUnit(Module):
	def __init__(self, module_name, config, constants):
		Module.__init__(self, module_name, constants['quantization_unit']['dynamic'], constants['quantization_unit']['leakage'], constants['quantization_unit']['area'], constants['clock_frequency'], config['sparsity']['before_quant'])
		self.assigned_op = None

	def assign_op(self, op):
		self.process_cycles = 3
		self.ready = False

		self.assigned_op = op
		
class DequantizationUnit(Module):
	def __init__(self, module_name, config, constants):
		Module.__init__(self, module_name, constants['dequantization_unit']['dynamic'], constants['dequantization_unit']['leakage'], constants['dequantization_unit']['area'], constants['clock_frequency'], config['sparsity']['after_quant'])
		self.assigned_op = None

	def assign_op(self, op):
		self.process_cycles = 2
		self.ready = False

		self.assigned_op = op
		
class Softmax(Module):
	def __init__(self, module_name, config, constants):
		Module.__init__(self, module_name, constants['softmax']['dynamic'], constants['softmax']['leakage'], constants['softmax']['area'], constants['clock_frequency'], 0)
		self.assigned_op = None

	def assign_op(self, op):
		self.process_cycles = 46
		self.ready = False

		self.assigned_op = op

class Layernorm(Module):
	def __init__(self, module_name, config, constants):
		Module.__init__(self, module_name, constants['layernorm']['dynamic'], constants['layernorm']['leakage'], constants['layernorm']['area'], constants['clock_frequency'], 0)
		self.assigned_op = None

	def assign_op(self, op):
		self.process_cycles = 189
		self.ready = False

		self.assigned_op = op

class ReLu(Module):
	def __init__(self, module_name, config, constants):
		Module.__init__(self, module_name, constants['relu']['dynamic'], constants['relu']['leakage'], constants['relu']['area'], constants['clock_frequency'], 0)
		self.assigned_op = None

	def assign_op(self, op):
		self.process_cycles = 2
		self.ready = False

		self.assigned_op = op

class CimMacro(Module):
	def __init__(self, module_name, config, constants):
		Module.__init__(self, module_name, constants['cim_macro']['dynamic'], constants['cim_macro']['leakage'], constants['cim_macro']['area'], constants['clock_frequency'], 0)
		self.assigned_op = None

	def assign_op(self, op):
		self.process_cycles = 9
		self.ready = False

		self.assigned_op = op

class Accumulator(Module):
	def __init__(self, module_name, config, constants):
		Module.__init__(self, module_name, constants['accumulator']['dynamic'], constants['accumulator']['leakage'], constants['accumulator']['area'], constants['clock_frequency'], 0)
		self.assigned_op = None

	def assign_op(self, op):
		self.process_cycles = 2
		self.ready = False

		self.assigned_op = op        
