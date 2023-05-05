# AudioMNIST on Android
This is the code by Jorge Chavez, Max Song, and Jonathan Chang for Android. 

# Overview

The project takes a TensorFlow Lite model trained on a voice, and predicts what number was spoken. A script for generating a model is given in *'/tensorflow_model'*, but you must provide your own data. The Android app can also be used to curate a dataset as well. The directory called *'MusicClassification/'* contains notebooks and code that run CUDA in Python scipts and Jupyter notebooks, but is not refined and not necessary. It is sufficient to procure your own dataset using the application, and generating a model using the *'tensorflow_model/spoken_digit_recognition_tensorflow_images.ipynb'* notebook. One model is provided in the root directory.
## Dependencies

All included

## Usage

Just open this project with Android Studio and is ready to run, this will work
with x86 and armeabi-v7a architectures.

### How to export my model?

A full example can be seen [here](https://github.com/mari-linhares/mnist-android-tensorflow/blob/master/tensorflow_model/convnet.py)

1. Train your model
2. Keep an in memory copy of eveything your model learned (like biases and weights)
   Example: `_w = sess.eval(w)`, where w was learned from training.
3. Rewrite your model changing the variables for constants with value = in memory copy of learned variables.
   Example: `w_save = tf.constant(_w)`  

   Also make sure to put names in the input and output of the model, this will be needed for the model later.
   Example:  
   `x = tf.placeholder(tf.float32, [None, 1000], name='input')`  
   `y = tf.nn.softmax(tf.matmul(x, w_save) + b_save), name='output')`  
4. Export your model with:  
   `tf.train.write_graph(<graph>, <path for the exported model>, <name of the model>.pb, as_text=False)`

### How to run my model with Android?

You need `tensorflow.aar`, which can be downloaded from [the nightly build artifact of TensorFlow CI](http://ci.tensorflow.org/view/Nightly/job/nightly-android/), here we use [the #124 build](http://ci.tensorflow.org/view/Nightly/job/nightly-android/124/artifact/).

### Interacting with TensorFlow

To interact with TensorFlow you will need an instance of TensorFlowInferenceInterface, you can see more details about it [here](https://github.com/mari-linhares/mnist-android-tensorflow/blob/master/MnistAndroid/app/src/main/java/mariannelinhares/mnistandroid/Classifier.java)

## Credits

Credits for this code go to [mari-linhares](https://github.com/mari-linhares/mnist-android-tensorflow). I've merely created a wrapper to get people started. 
