import os
import sys
import math
import json
import argparse
import yaml
from tqdm import tqdm
from ops import *

def get_ops(model_dict, config, debug):
	ops = []
	batch_size = config['batch_size']
	#SEQ_LENGTH = config['prefill_seq']+config['decode_seq']
	SEQ_LENGTH = 16 

	ops.append(MemoryLoadOp('input', config, (batch_size, config['prefill_seq'], model_dict['h'][0]), 'global')) # only need to load prompt tokens at first
	#op = MemoryLoadOp('input', config, (batch_size, SEQ_LENGTH, model_dict['h'][0]), 'activation')
	#print("Operation Name:", op.op_name)
	#print("Input Size:", op.input_size)
	#print("Data Type:", op.data_type)
	#print("Compute Operation:", op.compute_op)
	#print("Base Operation:", op.base_op)
	#print(op)

	for layer in range(model_dict['l']):
		layer_hidden_size = model_dict['h'][layer]
		multihead_ops = []
		ops.append(PreLayerNormOp(f'ln_{layer}_1', config, ['input'], input_size=(batch_size, SEQ_LENGTH, math.ceil(layer_hidden_size / config['layernorm_parallelism']))))

		for attention_head in model_dict['o'][layer]:
			type, param, hidden = attention_head.split('_')

			op_name = 'fused_attention_head' + '_' + str(layer + 1)
			input_size = (batch_size, SEQ_LENGTH, layer_hidden_size)
			
			if type == 'sa':
				multihead_ops.append(SelfAttentionOp(op_name, config, input_size, hidden_size=int(hidden)))

			if debug: print(f'Added operation with name: {op_name}')

		ops.extend(multihead_ops)
		ops.append(AftLayerNormOp(f'ln_{layer}_2', config, [], input_size=(batch_size, SEQ_LENGTH, math.ceil(layer_hidden_size / config['layernorm_parallelism']))))

		last_hidden_size = layer_hidden_size

		for hidden in model_dict['f'][layer]:
			op_name = 'ff' + '_' + str(layer + 1)

			input_size = (batch_size, SEQ_LENGTH, last_hidden_size)

			ops.append(FeedForwardOp(op_name, config, input_size, hidden_size=hidden))

			if debug: print(f'Added operation with name: {op_name}')
	
	#print(ops)
	
	return ops

def get_tiled_ops(ops, tile_compute_ops, tile_memory_ops, debug):
	"""Get tiled operations in forward directions"""
	memory_ops, compute_ops = [], []
	num_ops = 0
	for op in tqdm(ops, desc=f'Converting model to hardware operations'):
		if isinstance(op, list):
			#print("yes")
			memory_multihead_ops, compute_multihead_ops = [], []
			for head_op in op:
				memory_head_ops, compute_head_ops = [], []
				if head_op.base_op:
					if head_op.compute_op:
						compute_head_ops.extend(head_op.tile_op() if tile_compute_ops else [head_op])
					else:
						memory_head_ops.extend(head_op.tile_op() if tile_memory_ops else [head_op])
				else:
					head_op.convert_to_fwd_base_ops()
					for base_op in head_op.fwd_base_ops:
						if base_op.compute_op:
							compute_head_ops.extend(base_op.tile_op() if tile_compute_ops else [base_op])
						else:
							memory_head_ops.extend(base_op.tile_op() if tile_memory_ops else [base_op])
				if memory_head_ops: 
					num_ops += len(memory_head_ops)
					memory_multihead_ops.append(memory_head_ops)
				if compute_head_ops: 
					num_ops += len(compute_head_ops)
					compute_multihead_ops.append(compute_head_ops)
			memory_ops.append(memory_multihead_ops); compute_ops.append(compute_multihead_ops)
		else:
			#print("no")
			if op.base_op:
				#print("base_op:")
				#print(op.op_name)
				if op.compute_op:
					new_ops = op.tile_op() if tile_compute_ops else [op]
					#print("base_op_compute_layernorm:")
					#print(len(new_ops))
					num_ops += len(new_ops)
					compute_ops.extend(new_ops)
				else:
					new_ops = op.tile_op() if tile_memory_ops else [op]
					#print("base_op_memory:")
					#print(len(new_ops))
					num_ops += len(new_ops)
					memory_ops.extend(new_ops)
			else:
				#print("none_base_op:")
				#print(op.op_name)
				op.convert_to_fwd_base_ops()
				#print("op.fwd_base_ops:")
				#print(op.fwd_base_ops)
				for base_op in op.fwd_base_ops:
					if base_op.compute_op:
						new_ops = base_op.tile_op() if tile_compute_ops else [base_op]
						#print("base_op_compute:")
						#print(len(new_ops))
						num_ops += len(new_ops)
						compute_ops.extend(new_ops)
					else:
						new_ops = base_op.tile_op() if tile_memory_ops else [base_op]
						#print("base_op_memory:")
						#print(len(new_ops))
						num_ops += len(new_ops)
						memory_ops.extend(new_ops)

	if debug:
		print(f'Number of operations: {num_ops}')

	return memory_ops, compute_ops, num_ops

def main(model_dict: dict, config: dict, tile_compute_ops=False, tile_memory_ops=False, debug=True):
	"""Convert model dictionary to software compute operations"""
	fwd_ops = get_ops(model_dict, config, debug=debug)

	fwd_memory_ops, fwd_compute_ops, fwd_num_ops = get_tiled_ops(fwd_ops, tile_compute_ops=tile_compute_ops, tile_memory_ops=tile_memory_ops, debug=debug)
	
	#print(fwd_memory_ops)
	#print(fwd_compute_ops)
	#print(fwd_num_ops)
	
	return fwd_memory_ops, fwd_compute_ops, fwd_num_ops

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Input parameters for conversion',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model_dict_path',
		metavar='',
		type=str,
		help='path where the model dictionary file is stored')
	parser.add_argument('--config_path',
		metavar='',
		type=str,
		help='path to the configuration file')
	parser.add_argument('--tile_compute_ops',
		dest='tile_compute_ops',
		help='tile software operations',
		action='store_true')
	parser.add_argument('--tile_memory_ops',
		dest='tile_memory_ops',
		help='tile software operations',
		action='store_true')
	parser.add_argument('--debug',
		dest='debug',
		help='print debugging statements',
		action='store_true')
	parser.set_defaults(debug=False)
	parser.set_defaults(tile_ops=False)

	args = parser.parse_args()

	if os.path.exists(args.model_dict_path):
		model_dict = json.load(open(args.model_dict_path, 'r'))
	else:
		raise FileNotFoundError(f'Couldn\'t find JSON file for given path: {args.model_dict_path}')

	if os.path.exists(args.config_path):
		config = yaml.safe_load(open(args.config_path, 'r'))
	else:
		raise FileNotFoundError(f'Couldn\'t find JSON file for given path: {args.config_path}')

	main(model_dict, config, args.tile_compute_ops, args.tile_memory_ops, args.debug)
