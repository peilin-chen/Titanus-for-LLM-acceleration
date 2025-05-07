import math
from ops import *
from tiled_ops import *
from modules import *

class Fc1Cim(object):
	"""Fc1Cim class
	
	Attributes:
		Fc1Cim (str): name of the given Fc1Cim
		cim_macros (list): list of CimMacro objects
		accumulator (Accumulator): Accumulator module object
	"""
	def __init__(self, fc1_cim_name, config, constants):
		self.fc1_cim_name = fc1_cim_name

		self.cim_macros = []
		for n in range(math.ceil(config['hidden_size'] / 256) * math.ceil(config['ff_dim'] / 32)): # macro_size: [256, 32]
			self.cim_macros.append(CimMacro(f'{self.fc1_cim_name}_CimMacro{(n + 1)}', config, constants))
		
		self.accumulators = []
		for n in range(config['ff_dim']):
			self.accumulators.append(Accumulator(f'{self.fc1_cim_name}_Accumulator{(n + 1)}', config, constants))
			
		self.area = 0
		self.power = 0
		for cim_macro in self.cim_macros:
			self.area += cim_macro.area
			self.power += (cim_macro.dynamic_power+cim_macro.leakage_power)
		
		for accumulator in self.accumulators:
			self.area += accumulator.area
			self.power += (accumulator.dynamic_power+accumulator.leakage_power)
		
	def process_cycle(self):
		total_energy = [0, 0]
		
		for cim_macro in self.cim_macros:
			cim_macro_energy = cim_macro.process_cycle()
			total_energy[0] += cim_macro_energy[0]
			total_energy[1] += cim_macro_energy[1]
			
		for accumulator in self.accumulators:
			accumulator_energy = accumulator.process_cycle()
			total_energy[0] += accumulator_energy[0]
			total_energy[1] += accumulator_energy[1]

		return tuple(total_energy) # unit: nJ

	def assign_op(self, op):
		assert op.compute_op is True
		assigned_op = False

		for cim_macro in self.cim_macros:
			if cim_macro.ready:
				cim_macro.assign_op(op)
				assigned_op = True
		
		for accumulator in self.accumulators:
			if accumulator.ready:
				accumulator.assign_op(op)
				assigned_op = True		

		return assigned_op

