"""
Perform net surgery.
"""

import sys
import h5py
import tensorflow as tf
import gen_batch

input_metagraphdef_file = sys.argv[1]
input_hdf5_file         = sys.argv[2]
output_dir              = sys.argv[3]

model_arch = 'vgg'

with tf.Session() as sess:
	# Load the MetaGraphDef file into the default graph
	new_saver = tf.train.import_meta_graph(input_metagraphdef_file)

	# initialize variables and then load the ones that are available,
	# and who's sizes match.
	sess.run(tf.initialize_all_variables())
	with h5py.File(input_hdf5_file,'r') as f:
		for var in tf.get_collection(tf.GraphKeys.VARIABLES, scope=model_arch):
			h5var = f[var.name[(len(model_arch)+1):-2]][:]
			if var.get_shape() == h5var.shape:
				print var.name + " matched shape, loading."
				sess.run(var.assign(h5var))
			else:
				print var.name + " didn't match shape, initializing."

	# Write to output
	new_saver.save(sess, output_dir, global_step=0)