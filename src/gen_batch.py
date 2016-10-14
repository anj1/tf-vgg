"""
Generate data batches.
"""

# if code gives cuDNN error, use:
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64

import h5py
import numpy as np
import tensorflow as tf


def random_patch(images,annots,patch_size,n_testing,num_classes,testing=False):
	# get image dimensions and 'safe' zone for glimpses
	nimg = tf.shape(images)[0]
	testing=False
	if testing:
		minv = nimg - n_testing
		maxv = nimg
	else:
		minv = 0
		maxv = nimg - n_testing

	h = tf.shape(images)[1]
	w = tf.shape(images)[2]
	n = patch_size/2
	# first, generate a set of random glimpse coordinates
	randi = tf.random_uniform(shape=[],
	                          minval=minv,
	                          maxval=maxv,
	                          dtype=tf.int32)
	randy = tf.random_uniform(shape=[],
		                      minval=0,
		                      maxval=h-patch_size,
		                      dtype=tf.int32)
	randx = tf.random_uniform(shape=[],
		                      minval=0,
		                      maxval=w-patch_size,
		                      dtype=tf.int32)
	ptch = tf.slice(images, #images,
		            begin=[randi, randy, randx, 0],
		            size= [1,patch_size,patch_size,3])
	lbl  = tf.slice(annots, #annots,
		            begin=[randi, randy + n, randx + n, 0],
		            size= [1,1,1,num_classes])
	# shuffle around randomly and drop the testing images
	# TODO
	return (tf.squeeze(ptch,squeeze_dims=[0]),
	        tf.squeeze(lbl,squeeze_dims=[0,1,2]))

def gen_batch_op(h5filename, patch_size, batch_size, n_testing, num_classes, testing):
	f = h5py.File(h5filename,'r')
	img  = (f['img'][...]).astype('float32')/255.0
	anno = (f['anno'][...]).astype('float32')
	f.close()
	anno = np.expand_dims(anno, 3)
	anno = np.concatenate((1.0-anno, anno), axis=3)
	nimg = img.shape[0]

	# create a FIFO queues for images and labels
	q = tf.FIFOQueue(capacity=2*batch_size,
	                 dtypes=[tf.float32, tf.float32],
	                 shapes=[(patch_size, patch_size, 3), (num_classes)])

	# generate data on CPU
	with tf.device('/cpu:0'):
		input_img  = tf.constant(img)
		input_anno = tf.constant(anno)
		enqueue_op = q.enqueue(random_patch(input_img, input_anno, patch_size, n_testing, num_classes, testing))
		# Create a queue runner that will run 1 thread to enqueue examples
		qr = tf.train.QueueRunner(q, [enqueue_op] * 3)

	# Fetch and process on GPU
	with tf.device('/gpu:0'):
		inp_raw = q.dequeue_many(batch_size)

	return inp_raw
