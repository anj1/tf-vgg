"""
This code generates a VGG model (either vgg16 or vgg19).
This model can then be saved as a GraphDef.

This python file can be either included in another project,
Or run as a standalone script (in which case it just constructs and saves the graph.)
Example usage: vgg16 with input size 224:
	python vgg.py 224 16 1000 models/vgg16_244_1000.meta
or, for instance, vgg19 with input size 64 and 2 output classes:
	python vgg.py 64 19 2 models/vgg19_64_2.meta

Author: Al Nejati
TrainedEye Medical Technologies Ltd.
"""

import sys 
import tensorflow as tf

# A single convolutional layer with in_channels input features and
# out_channels output features.
def conv_layer(x, in_channels, out_channels, name):
	with tf.variable_scope(name):
		filt = tf.get_variable("filters",
			                   shape=[3, 3, in_channels, out_channels],
			                   initializer=tf.truncated_normal_initializer(0.0, 0.001))
		bias = tf.get_variable("biases",
			                   shape=[out_channels],
			                   initializer=tf.truncated_normal_initializer(0.0, 0.001))
		y = tf.nn.conv2d(x, filt, [1, 1, 1, 1], padding='SAME')
		z = tf.nn.bias_add(y, bias)
		return tf.nn.relu(z)

# A single fc (flat e.g. dense) layer
def fc_layer(x, in_channels, out_channels, train_mode, dropout_frac, name):
	with tf.variable_scope(name):
		weights = tf.get_variable("weights",
			                       shape=[in_channels, out_channels],
			                       initializer=tf.truncated_normal_initializer(0.0, 0.001))
		biases  = tf.get_variable("biases",
			                      shape=[out_channels],
			                      initializer=tf.truncated_normal_initializer(0.0, 0.001))
		y = tf.reshape(x, [-1, in_channels])
		fc = tf.nn.bias_add(tf.matmul(y, weights), biases)
		z = tf.nn.relu(fc)
		return tf.cond(train_mode, lambda: tf.nn.dropout(z, dropout_frac), lambda: z)

# A stack of n_convs convolutions followed by a max-pooling operation.
# VGG has five of these blocks.
def conv_block(x, n_convs, n_input_channels, n_output_channels, name):
	with tf.variable_scope(name):
		n_channels = n_input_channels
		for i in range(n_convs):
			x = conv_layer(x, n_channels, n_output_channels, "conv_%d" % (i+1))
			n_channels = n_output_channels
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool")

# The VGG model. This is a stack of five conv_blocks plus three fc_layers.
def vgg(x, train_mode, num_classes, patch_size=224, vgg_type="16"):
	assert(vgg_type=="19" or vgg_type=="16")
	nc = 4 if vgg_type=="19" else 3
	
	with tf.variable_scope("vgg"):
		# TODO: normalize inputs 
		pool1 = conv_block(    x, 2,    3,  64, "block_1")
		pool2 = conv_block(pool1, 2,   64, 128, "block_2")
		pool3 = conv_block(pool2, nc, 128, 256, "block_3")
		pool4 = conv_block(pool3, nc, 256, 512, "block_4")
		pool5 = conv_block(pool4, nc, 512, 512, "block_5")
		fc6   = fc_layer(pool5, ((patch_size/(2**5))**2)*512, 4096, train_mode, 0.5, "fc_6")
		fc7   = fc_layer(fc6,   4096, 4096, train_mode, 0.5, "fc_7")
		fc8   = fc_layer(fc7,   4096, num_classes, tf.constant(False), 0.5, "fc_8")
		return tf.nn.softmax(fc8, name="prob")

if __name__ == "__main__":
	patch_size   =   int(sys.argv[1])
	vgg_type     =       sys.argv[2]
	num_classes  =   int(sys.argv[3])
	out_filename =       sys.argv[4]
	x = tf.placeholder(shape=[None, patch_size, patch_size, 3], dtype=tf.float32, name="input")
	train_mode = tf.placeholder(shape=[], dtype=tf.bool, name="train_mode")
	prob = vgg(x, train_mode, num_classes, patch_size, vgg_type)
	tf.train.export_meta_graph(filename=out_filename,as_text=True)