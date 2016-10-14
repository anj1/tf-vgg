"""
Train the net.
"""

import sys
import tensorflow as tf
import gen_batch

inp_data_file = sys.argv[1]
inp_meta_file = sys.argv[2]


batch_op = gen_batch.gen_batch_op(inp_data_file,224,10,10,2,False)

with tf.Session() as sess:
	tf.train.import_meta_graph(inp_meta_file)
