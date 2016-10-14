"""
Perform net surgery.
"""

import sys
import tensorflow as tf
import gen_batch

inp_data_file = sys.argv[1]
inp_meta_file = sys.argv[2]


batch_op = gen_batch.gen_batch_op(inp_data_file,224,10,10,2,False)


# nimg = img.shape[0]

# patch_size  = int(sys.argv[1])
# batch_size  = 10 #20
# n_testing   = 10
# num_classes = 2
# learning_rate = 1e-5
# momentum = 0.3
# inp_weight_file = sys.argv[2]
# if sys.argv[3] == 'train':
# 	testing = False 
# elif sys.argv[3] == 'test':
# 	testing = True
# else:
# 	print "please specify 'train' or 'test' on command line."
# n_iterations = int(sys.argv[4])

# # TODO: normalize data mean

#saver = tf.train.Saver()

with tf.Session() as sess:
	tf.train.import_meta_graph(inp_meta_file)
	#saver.restore(sess, inp_meta_file)