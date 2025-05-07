import math

class TiledOp(object):
	"""Class for a tiled operation"""
	def __init__(self, op_name):
		self.op_name = op_name
		self.done = False

class TiledData(object):
	"""Class for a tiled data block"""
	def __init__(self, data_name, data_size, data_type):
		self.data_name = data_name
		self.data_size = data_size
		self.data_type = data_type
		self.required_in_buffer = False

class MemoryLoadTiledOp(TiledOp):
	"""Memory load (from main memory to buffer) tiled operation
	
	Attributes:
		input_size (tuple): size of the input matrix to be loaded
		data_type (str): type of data to fetch in ['global', 'sz']
		compute_op (bool): if the operation is a compute operation
	"""
	def __init__(self, op_name, input_size, data_type):
		TiledOp.__init__(self, op_name)
		self.input_size = input_size
		self.data_type = data_type
		self.compute_op = False

	def convert_to_data(self):
		return TiledData(data_name=self.op_name, data_size=math.prod(self.input_size), data_type=self.data_type)

class MemoryStoreTiledOp(TiledOp):
	"""Memory store (from PEs to buffer) tiled operation 
	
	Attributes:
		input_size (tuple): size of the input matrix to be loaded
		data_type (str): type of data to fetch in ['global', 'sz']
		compute_op (bool): if the operation is a compute operation
	"""
	def __init__(self, op_name, input_size, data_type):
		TiledOp.__init__(self, op_name)
		self.input_size = input_size
		self.data_type = data_type
		self.compute_op = False

	def convert_to_data(self):
		return TiledData(data_name=self.op_name, data_size=math.prod(self.input_size), data_type=self.data_type)
		
class QKVVMultTiledOp(TiledOp):
	"""Query Key vector-vector multiplication tiled operation
	
	Attributes:
		input_1_size (tuple): size of the input_1 vector [b, x1, x2]=[b, 1, x2]
		input_2_size (tuple): size of the input_2 vector [b, x2, x3]=[b, x2, 1]
		compute_op (bool): if the operation is a compute operation
		required_in_buffer (list): list of data object names required in buffer
	"""
	def __init__(self, op_name, required_in_buffer, input_1_size, input_2_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_1_size = input_1_size
		self.input_2_size = input_2_size
		self.compute_op = True
		
		self.check_input_sizes()
		self.num_muls = input_1_size[0] * input_1_size[1] * input_1_size[2] * input_2_size[2]

	def check_input_sizes(self):
		"""Check if input vector can be multiplied
		
		Raises:
			ValueError: if input vector can't be multiplied
		"""
		if self.input_1_size[0] != self.input_2_size[0] or self.input_1_size[2] != self.input_2_size[1]:
			raise ValueError(f'Input vectors of sizes: {self.input_1_size} and {self.input_2_size} can\'t be multiplied')

	def output_size(self):
		"""Get the size of the output element
		
		Returns:
			output_size (tuple): size of the output element
		"""
		return (self.input_1_size[0], self.input_1_size[1], self.input_2_size[2]) # 1
		
class SVVVMultTiledOp(TiledOp):
	"""Score Value vector-vector multiplication tiled operation
	
	Attributes:
		input_1_size (tuple): size of the input_1 vector [b, x1, x2]=[b, 1, x2]
		input_2_size (tuple): size of the input_2 vector [b, x2, x3]=[b, x2, 1]
		compute_op (bool): if the operation is a compute operation
		required_in_buffer (list): list of data object names required in buffer
	"""
	def __init__(self, op_name, required_in_buffer, input_1_size, input_2_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_1_size = input_1_size
		self.input_2_size = input_2_size
		self.compute_op = True
		
		self.check_input_sizes()
		self.num_muls = input_1_size[0] * input_1_size[1] * input_1_size[2] * input_2_size[2]

	def check_input_sizes(self):
		"""Check if input vector can be multiplied
		
		Raises:
			ValueError: if input vector can't be multiplied
		"""
		if self.input_1_size[0] != self.input_2_size[0] or self.input_1_size[2] != self.input_2_size[1]:
			raise ValueError(f'Input vectors of sizes: {self.input_1_size} and {self.input_2_size} can\'t be multiplied')

	def output_size(self):
		"""Get the size of the output element
		
		Returns:
			output_size (tuple): size of the output element
		"""
		return (self.input_1_size[0], self.input_1_size[1], self.input_2_size[2]) # 1
		
class QAttCimVecMatMultTiledOp(TiledOp):
	"""vector-matrix multiplication tiled operation (CIM)
	
	Attributes:
		input_size (tuple): size of the input vector [b, x1, x2]=[b, 1, x2]
		compute_op (bool): if the operation is a compute operation
		required_in_buffer (list): list of data object names required in buffer
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		
		self.num_muls = input_size[0] * input_size[1] * input_size[2] * input_size[2]

	def output_size(self):
		"""Get the size of the output element
		
		Returns:
			output_size (tuple): size of the output element
		"""
		return (self.input_size[0], self.input_size[1], self.input_size[2])
		
class KAttCimVecMatMultTiledOp(TiledOp):
	"""vector-matrix multiplication tiled operation (CIM)
	
	Attributes:
		input_size (tuple): size of the input vector [b, x1, x2]=[b, 1, x2]
		compute_op (bool): if the operation is a compute operation
		required_in_buffer (list): list of data object names required in buffer
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		
		self.num_muls = input_size[0] * input_size[1] * input_size[2] * input_size[2]

	def output_size(self):
		"""Get the size of the output element
		
		Returns:
			output_size (tuple): size of the output element
		"""
		return (self.input_size[0], self.input_size[1], self.input_size[2])
		
class VAttCimVecMatMultTiledOp(TiledOp):
	"""vector-matrix multiplication tiled operation (CIM)
	
	Attributes:
		input_size (tuple): size of the input vector [b, x1, x2]=[b, 1, x2]
		compute_op (bool): if the operation is a compute operation
		required_in_buffer (list): list of data object names required in buffer
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		
		self.num_muls = input_size[0] * input_size[1] * input_size[2] * input_size[2]

	def output_size(self):
		"""Get the size of the output element
		
		Returns:
			output_size (tuple): size of the output element
		"""
		return (self.input_size[0], self.input_size[1], self.input_size[2])
		
class OAttCimVecMatMultTiledOp(TiledOp):
	"""vector-matrix multiplication tiled operation (CIM)
	
	Attributes:
		input_size (tuple): size of the input vector [b, x1, x2]=[b, 1, x2]
		compute_op (bool): if the operation is a compute operation
		required_in_buffer (list): list of data object names required in buffer
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		
		self.num_muls = input_size[0] * input_size[1] * input_size[2] * input_size[2]

	def output_size(self):
		"""Get the size of the output element
		
		Returns:
			output_size (tuple): size of the output element
		"""
		return (self.input_size[0], self.input_size[1], self.input_size[2])

class Fc1CimVecMatMultTiledOp(TiledOp):
	"""vector-matrix multiplication tiled operation (CIM)
	
	Attributes:
		input_size (tuple): size of the input vector [b, x1, x2]=[b, 1, x2]
		compute_op (bool): if the operation is a compute operation
		required_in_buffer (list): list of data object names required in buffer
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		
		self.num_muls = input_size[0] * input_size[1] * input_size[2] * (4 * input_size[2]) # d_ff=4*d_model

	def output_size(self):
		"""Get the size of the output element
		
		Returns:
			output_size (tuple): size of the output element
		"""
		return (self.input_size[0], self.input_size[1], 4 * self.input_size[2])  

class Fc2CimVecMatMultTiledOp(TiledOp):
	"""vector-matrix multiplication tiled operation (CIM)
	
	Attributes:
		input_size (tuple): size of the input vector [b, x1, x2]=[b, 1, x2]
		compute_op (bool): if the operation is a compute operation
		required_in_buffer (list): list of data object names required in buffer
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		
		self.num_muls = input_size[0] * input_size[1] * input_size[2] * (input_size[2] / 4) # d_ff=4*d_model

	def output_size(self):
		"""Get the size of the output element
		
		Returns:
			output_size (tuple): size of the output element
		"""
		return (self.input_size[0], self.input_size[1], self.input_size[2] / 4)		  

class PreLayerNormTiledOp(TiledOp):
	"""Layer normalization tiled operation
	
	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation 
		required_in_buffer (list): list of data object names required in buffer 
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		
class AftLayerNormTiledOp(TiledOp):
	"""Layer normalization tiled operation
	
	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation 
		required_in_buffer (list): list of data object names required in buffer 
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True

class ReLuTiledOp(TiledOp):
	"""ReLu tiled operation
	
	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation 
		required_in_buffer (list): list of data object names required in buffer 
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True

class SoftmaxTiledOp(TiledOp):
	"""Softmax tiled operation

	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation 
		required_in_buffer (list): list of data object names required in buffer 
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		
class PruningTiledOp(TiledOp):
	"""Pruning unit tiled operation

	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation 
		required_in_buffer (list): list of data object names required in buffer 
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		
class QuantTiledOp(TiledOp):
	"""Quantization unit tiled operation

	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation 
		required_in_buffer (list): list of data object names required in buffer 
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True
		
class DequantTiledOp(TiledOp):
	"""Dequantization unit tiled operation

	Attributes:
		input_size (tuple): size of the input vector
		compute_op (bool): if the operation is a compute operation 
		required_in_buffer (list): list of data object names required in buffer 
	"""
	def __init__(self, op_name, required_in_buffer, input_size):
		TiledOp.__init__(self, op_name)
		self.required_in_buffer = required_in_buffer
		self.input_size = input_size
		self.compute_op = True

