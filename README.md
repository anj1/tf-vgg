This is a 'canonical' TensorFlow implementation of VGG, using TF's own library mechanisms to simplify model creation, saving, and loading. The result is a very compact model. Features:

* Both VGG16 and VGG19 are implemented in the same code (can be selected with `vgg_type` parameter)
* Writes network graph definition as a MetaGraphDef file.
* Loads/saves network weights using the `tf.Saver` mechanism.

The way to use this source code is as follows. First, the MetaGraphDef file has to be generated:

```bash
python vgg.py <patch_size> <16|19> <dropout_fraction> <num_classes> <output_file.meta>
```

`patch_size` is the size of the input patch in pixels. It is best to pick a multiple of 32 smaller than or equal to 224, for example 192. Choose either '16' or '19' to select the vgg type. Select the `dropout_fraction` to be 0.0 if testing and, for instance, 0.5 for training. `num_classes` is the number of output classes. VOC is 1000; for many segmentation purposes it can be 2. Finally, `output_file` is the output MetaGraphDef file to write.

The second script does the 'net surgery' to re-use trained variables from the stock VGG model. If variable shapes match, they are used, otherwise the variables are initialized. The following command does this:

```bash
python src/load_weights.py models/vgg16.meta models/vgg16.h5 models/vgg16_custom
```
It loads the MetaGraphDef file we just generated, along with the stock vgg16.h5 weights, and outputs a vgg16_custom file using the `tf.train.Saver()` mechanism. Make sure that **cache/vgg16.h5** or **cache/vgg19.h5** exist and contain the proper initializions for the VGG model.


Now, we are ready to run the training and/or testing operations:

```bash
python src/train.py cache/colorectal.h5 models/vgg16_custom-0
```
Make sure that the **cache/colorectal.h5** file exists and contains the training data.


***
    TensorFlow is not Python.