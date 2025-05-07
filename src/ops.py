import math
from tiled_ops import *

class Op(object):
	def __init__(self, op_name, config):
		self.op_name = op_name
		self.config = config
		self.base_op = False
		self.done = False

		self.required_in_buffer = [] 

	def __repr__(self):
		return self.op_name

	@staticmethod
	def transpose_size(matrix_size):
		return (matrix_size[0], matrix_size[2], matrix_size[1])

class Data(object):
	def __init__(self, data_name, data_size, data_type, overwrite=False):
		self.data_name = data_name
		self.data_size = data_size
		self.data_type = data_type
		self.overwrite = overwrite
		self.required_in_buffer = False

class MemoryLoadOp(Op):
	def __init__(self, op_name, config, input_size, data_type):
		Op.__init__(self, op_name, config)
		self.input_size = input_size
		self.data_type = data_type
		self.compute_op = False
		self.base_op = True

	def convert_to_data(self):
		return Data(data_name=self.op_name, data_size=math.prod(self.input_size), data_type=self.data_type)

	def tile_op(self):
		self.tiled_ops = [MemoryLoadTiledOp(f'{self.op_name}_b{b}_x{x}_y{y}', (self.config['tile_memory']['tile_b'], self.config['tile_memory']['tile_x'], self.config['tile_memory']['tile_y']), self.data_type) for b in range(math.ceil(self.input_size[0] * 1.0 / self.config['tile_memory']['tile_b'])) for x in range(math.ceil(self.input_size[1] * 1.0 / self.config['tile_memory']['tile_x'])) for y in range(math.ceil(self.input_size[2] * 1.0 / self.config['tile_memory']['tile_y']))]

		return self.tiled_ops

class MemoryStoreOp(Op):
	def __init__(self, op_name, config, input_size, data_type, overwrite=False):
		Op.__init__(self, op_name, config)
		self.input_size = input_size
		self.data_type = data_type
		self.overwrite = overwrite
		self.compute_op = False
		self.base_op = True

	def convert_to_data(self):
		return Data(data_name=self.op_name, data_size=math.prod(self.input_size), data_type=self.data_type, overwrite=self.overwrite)

	def tile_op(self):
		self.tiled_ops = [MemoryStoreTiledOp(f'{self.op_name}_b{b}_x{x}_y{y}', (self.config['tile_memory']['tile_b'], self.config['tile_memory']['tile_x'], self.config['tile_memory']['tile_y']), self.data_type) for b in range(math.ceil(self.input_size[0] * 1.0 / self.config['tile_memory']['tile_b'])) for x in range(math.ceil(self.input_size[1] * 1.0 / self.config['tile_memory']['tile_x'])) for y in range(math.ceil(self.input_size[2] * 1.0 / self.config['tile_memory']['tile_y']))]

		return self.tiled_ops
		
class QKVVMultOp(Op):
	def __init__(self, op_name, config, required_in_buffer, input_1_size, input_2_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_1_size = input_1_size # [b, 1, x2]
		self.input_2_size = input_2_size # [b, x2, 1]
		self.compute_op = True
		self.base_op = True

		self.check_input_sizes()
		self.num_muls = input_1_size[0] * input_1_size[1] * input_1_size[2] * input_2_size[2]

	def check_input_sizes(self):
		if self.input_1_size[0] != self.input_2_size[0] or self.input_1_size[2] != self.input_2_size[1]:
			raise ValueError(f'Input matrices of sizes: {self.input_1_size} and {self.input_2_size} can\'t be multiplied')

	def output_size(self):
		return (self.input_1_size[0], self.input_1_size[1], self.input_2_size[2])

	def tile_op(self):
		num_tiles_b = math.ceil(self.input_1_size[0] * 1.0 / self.config['tile_computing_engine']['tile_b'])
		num_tiles_1_x = math.ceil(self.input_1_size[1] * 1.0 / self.config['tile_computing_engine']['tile_x'])
		num_tiles_1_y = math.ceil(self.input_1_size[2] * 1.0 / self.config['tile_computing_engine']['tile_y'])

		tile_size_1 = (self.config['tile_computing_engine']['tile_b'], self.config['tile_computing_engine']['tile_x'], self.config['tile_computing_engine']['tile_y'])
		tile_size_2 = (self.config['tile_computing_engine']['tile_b'], self.config['tile_computing_engine']['tile_y'], self.config['tile_computing_engine']['tile_x'])

		self.tiled_ops = []
		for l0 in range(num_tiles_b):
			for l1 in range(num_tiles_1_x):
				for l2 in range(num_tiles_1_y):
					op_name = f'{self.op_name}_b{l0}l_x{l1}_y{l2}'
					self.tiled_ops.append(QKVVMultTiledOp(op_name, self.required_in_buffer, tile_size_1, tile_size_2))

		return self.tiled_ops

class SVVVMultOp(Op):
	"""Score Value Vector Vector multiplication base operation

	Attributes:
		input_1_size (tuple): size of the input_1 vector
		input_2_size (tuple): size of the input_2 vector
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_1_size, input_2_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_1_size = input_1_size # [b, 1, x2]
		self.input_2_size = input_2_size # [b, x2, 1]
		self.compute_op = True
		self.base_op = True

		self.check_input_sizes()
		self.num_muls = input_1_size[0] * input_1_size[1] * input_1_size[2] * input_2_size[2]

	def check_input_sizes(self):
		"""Check if input vectors can be multiplied
		
		Raises:
			ValueError: if input vectors can't be multiplied
		"""
		if self.input_1_size[0] != self.input_2_size[0] or self.input_1_size[2] != self.input_2_size[1]:
			raise ValueError(f'Input matrices of sizes: {self.input_1_size} and {self.input_2_size} can\'t be multiplied')

	def output_size(self):
		"""Get the size of the output matrix
		
		Returns:
			output_size (tuple): size of the output matrix
		"""
		return (self.input_1_size[0], self.input_1_size[1], self.input_2_size[2])

	def tile_op(self):
		"""Implement tiled vector vector multiplication

		Returns:
			self.tiled_ops (list): list of SVVVMultTiledOps
		"""
		num_tiles_b = math.ceil(self.input_1_size[0] * 1.0 / self.config['tile_computing_engine']['tile_b'])
		num_tiles_1_x = math.ceil(self.input_1_size[1] * 1.0 / self.config['tile_computing_engine']['tile_x'])
		num_tiles_1_y = math.ceil(self.input_1_size[2] * 1.0 / self.config['tile_computing_engine']['tile_y'])

		tile_size_1 = (self.config['tile_computing_engine']['tile_b'], self.config['tile_computing_engine']['tile_x'], self.config['tile_computing_engine']['tile_y'])
		tile_size_2 = (self.config['tile_computing_engine']['tile_b'], self.config['tile_computing_engine']['tile_y'], self.config['tile_computing_engine']['tile_x'])

		self.tiled_ops = []
		for l0 in range(num_tiles_b):
			for l1 in range(num_tiles_1_x):
				for l2 in range(num_tiles_1_y):
					op_name = f'{self.op_name}_b{l0}l_x{l1}_y{l2}'
					self.tiled_ops.append(SVVVMultTiledOp(op_name, self.required_in_buffer, tile_size_1, tile_size_2))

		return self.tiled_ops

class QAttCimVecMatMultOp(Op):
	"""Vector Matrix multiplication base operation (CIM)

	Attributes:
		input_size (tuple): size of the input vector [b, x1, x2] = [b, 1, x2]
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

		self.num_muls = input_size[0] * input_size[1] * input_size[2] * input_size[2]

	def output_size(self):
		"""Get the size of the output matrix
		
		Returns:
			output_size (tuple): size of the output matrix
		"""
		return (self.input_size[0], self.input_size[1], self.input_size[2])

	def tile_op(self):
		"""Implement tiled vector matrix multiplication

		Returns:
			self.tiled_ops (list): list of AttCimVecMatMultTiledOps
		"""
		num_tiles_b = math.ceil(self.input_size[0] * 1.0 / self.config['tile_attention_cim']['tile_b'])
		num_tiles_x = math.ceil(self.input_size[1] * 1.0 / self.config['tile_attention_cim']['tile_x'])
		num_tiles_y = math.ceil(self.input_size[2] * 1.0 / self.config['tile_attention_cim']['tile_y'])

		tile_size = (self.config['tile_attention_cim']['tile_b'], self.config['tile_attention_cim']['tile_x'], self.config['tile_attention_cim']['tile_y'])

		self.tiled_ops = []
		for l0 in range(num_tiles_b):
			for l1 in range(num_tiles_x):
				for l2 in range(num_tiles_y):
					op_name = f'{self.op_name}_b{l0}_x{l1}_y{l2}'
					self.tiled_ops.append(QAttCimVecMatMultTiledOp(op_name, self.required_in_buffer, tile_size))

		return self.tiled_ops

class KAttCimVecMatMultOp(Op):
	"""Vector Matrix multiplication base operation (CIM)

	Attributes:
		input_size (tuple): size of the input vector [b, x1, x2] = [b, 1, x2]
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

		self.num_muls = input_size[0] * input_size[1] * input_size[2] * input_size[2]

	def output_size(self):
		"""Get the size of the output matrix
		
		Returns:
			output_size (tuple): size of the output matrix
		"""
		return (self.input_size[0], self.input_size[1], self.input_size[2])

	def tile_op(self):
		"""Implement tiled vector matrix multiplication

		Returns:
			self.tiled_ops (list): list of AttCimVecMatMultTiledOps
		"""
		num_tiles_b = math.ceil(self.input_size[0] * 1.0 / self.config['tile_attention_cim']['tile_b'])
		num_tiles_x = math.ceil(self.input_size[1] * 1.0 / self.config['tile_attention_cim']['tile_x'])
		num_tiles_y = math.ceil(self.input_size[2] * 1.0 / self.config['tile_attention_cim']['tile_y'])

		tile_size = (self.config['tile_attention_cim']['tile_b'], self.config['tile_attention_cim']['tile_x'], self.config['tile_attention_cim']['tile_y'])

		self.tiled_ops = []
		for l0 in range(num_tiles_b):
			for l1 in range(num_tiles_x):
				for l2 in range(num_tiles_y):
					op_name = f'{self.op_name}_b{l0}_x{l1}_y{l2}'
					self.tiled_ops.append(KAttCimVecMatMultTiledOp(op_name, self.required_in_buffer, tile_size))

		return self.tiled_ops
		
class VAttCimVecMatMultOp(Op):
	"""Vector Matrix multiplication base operation (CIM)

	Attributes:
		input_size (tuple): size of the input vector [b, x1, x2] = [b, 1, x2]
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

		self.num_muls = input_size[0] * input_size[1] * input_size[2] * input_size[2]

	def output_size(self):
		"""Get the size of the output matrix
		
		Returns:
			output_size (tuple): size of the output matrix
		"""
		return (self.input_size[0], self.input_size[1], self.input_size[2])

	def tile_op(self):
		"""Implement tiled vector matrix multiplication

		Returns:
			self.tiled_ops (list): list of AttCimVecMatMultTiledOps
		"""
		num_tiles_b = math.ceil(self.input_size[0] * 1.0 / self.config['tile_attention_cim']['tile_b'])
		num_tiles_x = math.ceil(self.input_size[1] * 1.0 / self.config['tile_attention_cim']['tile_x'])
		num_tiles_y = math.ceil(self.input_size[2] * 1.0 / self.config['tile_attention_cim']['tile_y'])

		tile_size = (self.config['tile_attention_cim']['tile_b'], self.config['tile_attention_cim']['tile_x'], self.config['tile_attention_cim']['tile_y'])

		self.tiled_ops = []
		for l0 in range(num_tiles_b):
			for l1 in range(num_tiles_x):
				for l2 in range(num_tiles_y):
					op_name = f'{self.op_name}_b{l0}_x{l1}_y{l2}'
					self.tiled_ops.append(VAttCimVecMatMultTiledOp(op_name, self.required_in_buffer, tile_size))

		return self.tiled_ops
		
class OAttCimVecMatMultOp(Op):
	"""Vector Matrix multiplication base operation (CIM)

	Attributes:
		input_size (tuple): size of the input vector [b, x1, x2] = [b, 1, x2]
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

		self.num_muls = input_size[0] * input_size[1] * input_size[2] * input_size[2]

	def output_size(self):
		"""Get the size of the output matrix
		
		Returns:
			output_size (tuple): size of the output matrix
		"""
		return (self.input_size[0], self.input_size[1], self.input_size[2])

	def tile_op(self):
		"""Implement tiled vector matrix multiplication

		Returns:
			self.tiled_ops (list): list of AttCimVecMatMultTiledOps
		"""
		num_tiles_b = math.ceil(self.input_size[0] * 1.0 / self.config['tile_attention_cim']['tile_b'])
		num_tiles_x = math.ceil(self.input_size[1] * 1.0 / self.config['tile_attention_cim']['tile_x'])
		num_tiles_y = math.ceil(self.input_size[2] * 1.0 / self.config['tile_attention_cim']['tile_y'])

		tile_size = (self.config['tile_attention_cim']['tile_b'], self.config['tile_attention_cim']['tile_x'], self.config['tile_attention_cim']['tile_y'])

		self.tiled_ops = []
		for l0 in range(num_tiles_b):
			for l1 in range(num_tiles_x):
				for l2 in range(num_tiles_y):
					op_name = f'{self.op_name}_b{l0}_x{l1}_y{l2}'
					self.tiled_ops.append(OAttCimVecMatMultTiledOp(op_name, self.required_in_buffer, tile_size))

		return self.tiled_ops

class Fc1CimVecMatMultOp(Op):
	"""Vector Matrix multiplication base operation (CIM)

	Attributes:
		input_size (tuple): size of the input vector [b, x1, x2] = [b, 1, x2]
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

		self.num_muls = input_size[0] * input_size[1] * input_size[2] * (4 * input_size[2])

	def output_size(self):
		"""Get the size of the output matrix
		
		Returns:
			output_size (tuple): size of the output matrix
		"""
		return (self.input_size[0], self.input_size[1], 4 * self.input_size[2])

	def tile_op(self):
		"""Implement tiled vector matrix multiplication

		Returns:
			self.tiled_ops (list): list of Fc1CimVecMatMultTiledOps
		"""
		num_tiles_b = math.ceil(self.input_size[0] * 1.0 / self.config['tile_fc1_cim']['tile_b'])
		num_tiles_x = math.ceil(self.input_size[1] * 1.0 / self.config['tile_fc1_cim']['tile_x'])
		num_tiles_y = math.ceil(self.input_size[2] * 1.0 / self.config['tile_fc1_cim']['tile_y'])

		tile_size = (self.config['tile_fc1_cim']['tile_b'], self.config['tile_fc1_cim']['tile_x'], self.config['tile_fc1_cim']['tile_y'])

		self.tiled_ops = []
		for l0 in range(num_tiles_b):
			for l1 in range(num_tiles_x):
				for l2 in range(num_tiles_y):
					op_name = f'{self.op_name}_b{l0}_x{l1}_y{l2}'
					self.tiled_ops.append(Fc1CimVecMatMultTiledOp(op_name, self.required_in_buffer, tile_size))

		return self.tiled_ops
		
class Fc2CimVecMatMultOp(Op):
	"""Vector Matrix multiplication base operation (CIM)

	Attributes:
		input_size (tuple): size of the input vector [b, x1, x2] = [b, 1, x2]
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

		self.num_muls = input_size[0] * input_size[1] * input_size[2] * (input_size[2] / 4)

	def output_size(self):
		"""Get the size of the output matrix
		
		Returns:
			output_size (tuple): size of the output matrix
		"""
		return (self.input_size[0], self.input_size[1], self.input_size[2] / 4)

	def tile_op(self):
		"""Implement tiled vector matrix multiplication

		Returns:
			self.tiled_ops (list): list of Fc2CimVecMatMultTiledOps
		"""
		num_tiles_b = math.ceil(self.input_size[0] * 1.0 / self.config['tile_fc2_cim']['tile_b'])
		num_tiles_x = math.ceil(self.input_size[1] * 1.0 / self.config['tile_fc2_cim']['tile_x'])
		num_tiles_y = math.ceil(self.input_size[2] * 1.0 / self.config['tile_fc2_cim']['tile_y'])

		tile_size = (self.config['tile_fc2_cim']['tile_b'], self.config['tile_fc2_cim']['tile_x'], self.config['tile_fc2_cim']['tile_y'])

		self.tiled_ops = []
		for l0 in range(num_tiles_b):
			for l1 in range(num_tiles_x):
				for l2 in range(num_tiles_y):
					op_name = f'{self.op_name}_b{l0}_x{l1}_y{l2}'
					self.tiled_ops.append(Fc2CimVecMatMultTiledOp(op_name, self.required_in_buffer, tile_size))

		return self.tiled_ops

class PreLayerNormOp(Op):
	"""Layer normalization operation
	
	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

	def tile_op(self):
		"""Implement tiled layer normalization

		Returns:
			self.tiled_ops (list): list of LayerNormTiledOps
		"""
		self.tiled_ops = [PreLayerNormTiledOp(f'{self.op_name}_b{b}_x{x}_y{y}', self.required_in_buffer, (self.config['tile_layernorm']['tile_b'], self.config['tile_layernorm']['tile_x'], self.config['tile_layernorm']['tile_y'])) for b in range(math.ceil(self.input_size[0] * 1.0 / self.config['tile_layernorm']['tile_b'])) for x in range(math.ceil(self.input_size[1] * 1.0 / self.config['tile_layernorm']['tile_x'])) for y in range(math.ceil(self.input_size[2] * 1.0 / self.config['tile_layernorm']['tile_y']))]

		return self.tiled_ops
		
class AftLayerNormOp(Op):
	"""Layer normalization operation
	
	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

	def tile_op(self):
		"""Implement tiled layer normalization

		Returns:
			self.tiled_ops (list): list of LayerNormTiledOps
		"""
		self.tiled_ops = [AftLayerNormTiledOp(f'{self.op_name}_b{b}_x{x}_y{y}', self.required_in_buffer, (self.config['tile_layernorm']['tile_b'], self.config['tile_layernorm']['tile_x'], self.config['tile_layernorm']['tile_y'])) for b in range(math.ceil(self.input_size[0] * 1.0 / self.config['tile_layernorm']['tile_b'])) for x in range(math.ceil(self.input_size[1] * 1.0 / self.config['tile_layernorm']['tile_x'])) for y in range(math.ceil(self.input_size[2] * 1.0 / self.config['tile_layernorm']['tile_y']))]

		return self.tiled_ops

class ReLuOp(Op):
	"""ReLu operation
	
	Attributes:
		input_size (tuple): size of the input vector
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

	def tile_op(self):
		"""Implement tiled ReLu

		Returns:
			self.tiled_ops (list): list of ReLuTiledOps
		"""
		self.tiled_ops = [ReLuTiledOp(f'{self.op_name}_b{b}_x{x}_y{y}', self.required_in_buffer, (self.config['tile_relu']['tile_b'], self.config['tile_relu']['tile_x'], self.config['tile_relu']['tile_y'])) for b in range(math.ceil(self.input_size[0] * 1.0 / self.config['tile_relu']['tile_b'])) for x in range(math.ceil(self.input_size[1] * 1.0 / self.config['tile_relu']['tile_x'])) for y in range(math.ceil(self.input_size[2] * 1.0 / self.config['tile_relu']['tile_y']))]

		return self.tiled_ops


class SoftmaxOp(Op):
	"""Softmax operation

	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

	def tile_op(self):
		"""Implement tiled softmax

		Returns:
			self.tiled_ops (list): list of SoftmaxTiledOps
		"""
		self.tiled_ops = [SoftmaxTiledOp(f'{self.op_name}_b{b}_x{x}_y{y}', self.required_in_buffer, (self.config['tile_softmax']['tile_b'], self.config['tile_softmax']['tile_x'], self.config['tile_softmax']['tile_y'])) for b in range(math.ceil(self.input_size[0] * 1.0 / self.config['tile_softmax']['tile_b'])) for x in range(math.ceil(self.input_size[1] * 1.0 / self.config['tile_softmax']['tile_x'])) for y in range(math.ceil(self.input_size[2] * 1.0 / self.config['tile_softmax']['tile_y']))]

		return self.tiled_ops
		
class PruningOp(Op):
	"""Pruning operation

	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

	def tile_op(self):
		"""Implement tiled pruning

		Returns:
			self.tiled_ops (list): list of PruningTiledOps
		"""
		self.tiled_ops = [PruningTiledOp(f'{self.op_name}_b{b}_x{x}_y{y}', self.required_in_buffer, (self.config['tile_pruning']['tile_b'], self.config['tile_pruning']['tile_x'], self.config['tile_pruning']['tile_y'])) for b in range(math.ceil(self.input_size[0] * 1.0 / self.config['tile_pruning']['tile_b'])) for x in range(math.ceil(self.input_size[1] * 1.0 / self.config['tile_pruning']['tile_x'])) for y in range(math.ceil(self.input_size[2] * 1.0 / self.config['tile_pruning']['tile_y']))]

		return self.tiled_ops
		
class QuantOp(Op):
	"""Quantization operation

	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

	def tile_op(self):
		"""Implement tiled quantization

		Returns:
			self.tiled_ops (list): list of QuantTiledOps
		"""
		self.tiled_ops = [QuantTiledOp(f'{self.op_name}_b{b}_x{x}_y{y}', self.required_in_buffer, (self.config['tile_quant']['tile_b'], self.config['tile_quant']['tile_x'], self.config['tile_quant']['tile_y'])) for b in range(math.ceil(self.input_size[0] * 1.0 / self.config['tile_quant']['tile_b'])) for x in range(math.ceil(self.input_size[1] * 1.0 / self.config['tile_quant']['tile_x'])) for y in range(math.ceil(self.input_size[2] * 1.0 / self.config['tile_quant']['tile_y']))]

		return self.tiled_ops
		
class DequantOp(Op):
	"""Dequantization operation

	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation (only for base operation)
		required_in_buffer (list): list of data object names required in buffer (only for base operation which is also a compute operation)
	"""
	def __init__(self, op_name, config, required_in_buffer, input_size):
		Op.__init__(self, op_name, config)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		self.base_op = True

	def tile_op(self):
		"""Implement tiled dequantization

		Returns:
			self.tiled_ops (list): list of DequantTiledOps
		"""
		self.tiled_ops = [DequantTiledOp(f'{self.op_name}_b{b}_x{x}_y{y}', self.required_in_buffer, (self.config['tile_dequant']['tile_b'], self.config['tile_dequant']['tile_x'], self.config['tile_dequant']['tile_y'])) for b in range(math.ceil(self.input_size[0] * 1.0 / self.config['tile_dequant']['tile_b'])) for x in range(math.ceil(self.input_size[1] * 1.0 / self.config['tile_dequant']['tile_x'])) for y in range(math.ceil(self.input_size[2] * 1.0 / self.config['tile_dequant']['tile_y']))]

		return self.tiled_ops

class SelfAttentionOp(Op):
	"""Self-attention operation
	
	Attributes:
		input_size (tuple): size of the input vector
		hidden_size (int): hidden size of the attention head
	"""
	def __init__(self, op_name, config, input_size, hidden_size):
		Op.__init__(self, op_name, config)
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.fwd_base_ops = []

	def convert_to_fwd_base_ops(self):
		self.weight_size = (1, self.input_size[2], self.hidden_size) 
		self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_q-l', self.config, self.weight_size, 'global'))
		self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_k-l', self.config, self.weight_size, 'global'))
		self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_v-l', self.config, self.weight_size, 'global'))

		query_op = QAttCimVecMatMultOp(f'{self.op_name}_q', self.config, [], self.input_size)
		key_op = KAttCimVecMatMultOp(f'{self.op_name}_k', self.config, [], self.input_size)
		value_op = VAttCimVecMatMultOp(f'{self.op_name}_v', self.config, [], self.input_size)

		self.fwd_base_ops.extend([query_op, key_op, value_op])

		self.query_size, self.key_size, self.value_size = query_op.output_size(), key_op.output_size(), value_op.output_size()

		self.key_transposed_size = Op.transpose_size(self.key_size)

		self.mult_key_size = self.key_transposed_size
		
		if self.config['pruning']:
			pruning_kv_op = PruningOp(f'{self.op_name}_kv_prune', self.config, [], (self.key_size[0], self.key_size[1], math.ceil((self.key_size[2]+self.value_size[2])/self.config['pruning_unit_parallelism'])))
			self.fwd_base_ops.append(pruning_kv_op)
		
		if self.config['pruning']:
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_prune-s', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil((self.key_size[2]+self.value_size[2])*(1-self.config['sparsity']['before_quant']))), 'global'))
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_prune-si', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil((self.key_size[2]+self.value_size[2]) / self.config['data_width'])), 'global'))

		if self.config['pruning'] and self.config['quantization']:
			quant_kv_op = QuantOp(f'{self.op_name}_kv_quant', self.config, [], (self.key_size[0], self.key_size[1], math.ceil((self.key_size[2]+self.value_size[2])*(1-self.config['sparsity']['before_quant'])/self.config['quantization_unit_parallelism'])))
			self.fwd_base_ops.append(quant_kv_op)
		elif self.config['quantization']:
			quant_kv_op = QuantOp(f'{self.op_name}_kv_quant', self.config, [], (self.key_size[0], self.key_size[1], math.ceil((self.key_size[2]+self.value_size[2])/self.config['quantization_unit_parallelism'])))
			self.fwd_base_ops.append(quant_kv_op)
		
		if self.config['pruning'] and self.config['quantization']:
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_quant-s', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil(((self.key_size[2]+self.value_size[2])*(1-self.config['sparsity']['after_quant'])) / (self.config['data_width'] / ((self.config['quant_bit']['key']+self.config['quant_bit']['value']) / 2)))), 'global'))
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_quant-si', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil((self.key_size[2]+self.value_size[2]) / self.config['data_width'])), 'global'))
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_quant-sl', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil((self.key_size[2]+self.value_size[2])*(1-self.config['sparsity']['before_quant'])/self.config['data_width'])), 'global'))

			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_quant-sm', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil(((self.key_size[2]+self.value_size[2])*(1-self.config['sparsity']['after_quant'])) / (self.config['data_width'] / ((self.config['quant_bit']['key']+self.config['quant_bit']['value']) / 2)))), 'global'))
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_quant-sim', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil((self.key_size[2]+self.value_size[2]) / self.config['data_width'])), 'global'))
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_quant-slm', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil((self.key_size[2]+self.value_size[2])*(1-self.config['sparsity']['before_quant'])/self.config['data_width'])), 'global'))

			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_quant-sp', self.config, (self.key_size[0], self.config['HQE_level_num'], self.key_size[2]+self.value_size[2]), 'sz'))
		elif self.config['quantization']:
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_quant-s', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil(((self.key_size[2]+self.value_size[2])) / (self.config['data_width'] / ((self.config['quant_bit']['key']+self.config['quant_bit']['value']) / 2)))), 'global'))
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_quant-sl', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil((self.key_size[2]+self.value_size[2])/self.config['data_width'])), 'global'))
			
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_quant-sm', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil(((self.key_size[2]+self.value_size[2])) / (self.config['data_width'] / ((self.config['quant_bit']['key']+self.config['quant_bit']['value']) / 2)))), 'global'))
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_quant-slm', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil((self.key_size[2]+self.value_size[2])/self.config['data_width'])), 'global'))
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_quant-sp', self.config, (self.key_size[0], self.config['HQE_level_num'], self.key_size[2]+self.value_size[2]), 'sz'))
		elif self.config['pruning']:
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_prune-sm', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil(((self.key_size[2]+self.value_size[2])*(1-self.config['sparsity']['before_quant'])))), 'global'))
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_kv_prune-sim', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil((self.key_size[2]+self.value_size[2]) / self.config['data_width'])), 'global'))
		else:
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_v-kvs', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil(self.key_size[2]+self.value_size[2])), 'global'))
			self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_v-kvm', self.config, (self.key_size[0], (self.config['prefill_seq']+self.config['decode_seq']), math.ceil(self.key_size[2]+self.value_size[2])), 'global'))
		
		loop_num_0 = self.config['decode_seq']
		
		for i in range(int(loop_num_0)):
			if self.config['pruning'] and self.config['quantization']:
				self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_{i}_kv_quant-l', self.config, (self.key_size[0], (self.config['prefill_seq']+i), math.ceil(((self.key_size[2]+self.value_size[2])*(1-self.config['sparsity']['after_quant'])) / (self.config['data_width'] / ((self.config['quant_bit']['key']+self.config['quant_bit']['value']) / 2)))), 'global'))
				self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_{i}_kv_quant-li', self.config, (self.key_size[0], (self.config['prefill_seq']+i), math.ceil((self.key_size[2]+self.value_size[2]) / self.config['data_width'])), 'global'))
				self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_{i}_kv_quant-ll', self.config, (self.key_size[0], (self.config['prefill_seq']+i), math.ceil((self.key_size[2]+self.value_size[2])*(1-self.config['sparsity']['before_quant'])/self.config['data_width'])), 'global'))
			elif self.config['quantization']:
				self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_{i}_kv_quant-l', self.config, (self.key_size[0], (self.config['prefill_seq']+i), math.ceil(((self.key_size[2]+self.value_size[2])) / (self.config['data_width'] / ((self.config['quant_bit']['key']+self.config['quant_bit']['value']) / 2)))), 'global'))
				self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_{i}_kv_quant-ll', self.config, (self.key_size[0], (self.config['prefill_seq']+i), math.ceil((self.key_size[2]+self.value_size[2])/self.config['data_width'])), 'global'))
			elif self.config['pruning']:
				self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_{i}_kv_prune-l', self.config, (self.key_size[0], (self.config['prefill_seq']+i), math.ceil((self.key_size[2]+self.value_size[2])*(1-self.config['sparsity']['before_quant']))), 'global'))
				self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_{i}_kv_prune-li', self.config, (self.key_size[0], (self.config['prefill_seq']+i), math.ceil((self.key_size[2]+self.value_size[2]) / self.config['data_width'])), 'global'))
			else:
				self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_{i}_v-kvl', self.config, (self.key_size[0], (self.config['prefill_seq']+i), math.ceil(self.key_size[2]+self.value_size[2])), 'global'))
		
		loop_num_1 = 8
		
		if self.config['pruning'] and self.config['quantization']:
			for i in range(int(loop_num_1)):
				dequant_kv_op = DequantOp(f'{self.op_name}_kv_dequant-{i}', self.config, [], (self.key_size[0], (8+i), math.ceil((self.key_size[2]+self.value_size[2])*(1-self.config['sparsity']['before_quant'])/self.config['dequantization_unit_parallelism'])))
				self.fwd_base_ops.append(dequant_kv_op)
		elif self.config['quantization']:
			for i in range(int(loop_num_1)):
				dequant_kv_op = DequantOp(f'{self.op_name}_kv_dequant-{i}', self.config, [], (self.key_size[0], (8+i), math.ceil((self.key_size[2]+self.value_size[2])/self.config['dequantization_unit_parallelism'])))
				self.fwd_base_ops.append(dequant_kv_op)
		
		loop_num_2 = ((8+8)*(8+8+1))/2
				
		for i in range(int(loop_num_2)):
			sdp_1_op = QKVVMultOp(f'{self.op_name}_sdp-qk-{i}', self.config, [], (self.query_size[0], 1, int(self.query_size[2] / self.config['head_num'])), (self.mult_key_size[0], int(self.mult_key_size[1] / self.config['head_num']), 1))
			self.fwd_base_ops.append(sdp_1_op)

		self.sdp_1_size = sdp_1_op.output_size()
		
		loop_num_3 = 8+8
				
		for i in range(loop_num_3):
			self.fwd_base_ops.append(SoftmaxOp(f'{self.op_name}_sftm-{i}', self.config, [], (self.sdp_1_size[0], self.sdp_1_size[1], i+1)))

		loop_num_4 = 8+8
					
		for i in range(loop_num_4):
			for k in range(int(self.value_size[2] / self.config['head_num'])):
				sdp_2_op = SVVVMultOp(f'{self.op_name}_sdp-sv-{i}-{k}', self.config, [], (self.sdp_1_size[0], self.sdp_1_size[1], i+1), (self.value_size[0], i+1, 1))
				self.fwd_base_ops.append(sdp_2_op)

		self.sdp_2_size = sdp_2_op.output_size()
		self.sdp_2_size = (self.sdp_2_size[0], loop_num_4, self.value_size[2])

		self.out_weight_size = (1, self.hidden_size, self.input_size[2])
		self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_o-l', self.config, self.out_weight_size, 'global'))

		out_op = OAttCimVecMatMultOp(f'{self.op_name}_o', self.config, [], self.sdp_2_size)
		self.fwd_base_ops.append(out_op)

		self.output_size = out_op.output_size()
		assert self.output_size == self.input_size

	def tile_op(self, tile_memory_ops=False):
		if not self.fwd_base_ops: self.convert_to_fwd_base_ops()
		base_ops = self.fwd_base_ops

		self.tiled_ops = []
		for op in base_ops:
			if isinstance(op, (MemoryLoadOp, MemoryStoreOp)):
				if tile_memory_ops: 
					self.tiled_ops.extend(op.tile_op())
				else:
					self.tiled_ops.append(op)
			else:
				self.tiled_ops.extend(op.tile_op())

		return self.tiled_ops

class FeedForwardOp(Op):
	"""Feed-forward  layer operation
	
	Attributes:
		input_size (tuple): size of the input vector
		hidden_size (int): hidden size of the attention head
	"""
	def __init__(self, op_name, config, input_size, hidden_size):
		Op.__init__(self, op_name, config)
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.fwd_base_ops = []

	def convert_to_fwd_base_ops(self):
		self.ff1_weight_size = (1, self.input_size[2], self.hidden_size)
		self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_f1-l', self.config, self.ff1_weight_size, 'global'))
		
		self.ff2_weight_size = (1, self.hidden_size, self.input_size[2])
		self.fwd_base_ops.append(MemoryLoadOp(f'{self.op_name}_f2-l', self.config, self.ff2_weight_size, 'global'))

		ff1_op = Fc1CimVecMatMultOp(f'{self.op_name}_f1', self.config, [], self.input_size)
		self.fwd_base_ops.append(ff1_op)

		ff1_size = ff1_op.output_size()
		
		relu_op = ReLuOp(f'{self.op_name}_relu', self.config, [], ff1_size)
		self.fwd_base_ops.append(relu_op)
		
		ff2_op = Fc2CimVecMatMultOp(f'{self.op_name}_f2', self.config, [], ff1_size)
		self.fwd_base_ops.append(ff2_op)

		ff2_size = ff2_op.output_size()

		# Store attenion-head output matrix
		self.output_size = (self.input_size[0], (self.config['prefill_seq']+self.config['decode_seq']), self.input_size[2])
		self.fwd_base_ops.append(MemoryStoreOp(f'{self.op_name}_f-s', self.config, self.output_size, 'global'))

	def tile_op(self, tile_memory_ops=False):
		if not self.fwd_base_ops: self.convert_to_fwd_base_ops()
		base_ops = self.fwd_base_ops

		self.tiled_ops = []
		for op in base_ops:
			if isinstance(op, (MemoryLoadOp, MemoryStoreOp)):
				if tile_memory_ops: 
					self.tiled_ops.extend(op.tile_op())
				else:
					self.tiled_ops.append(op)
			else:
				self.tiled_ops.extend(op.tile_op())

		return self.tiled_ops

