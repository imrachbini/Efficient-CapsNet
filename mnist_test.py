import tensorflow as tf
from utils import Dataset, plotImages, plotWrongImages
from models import EfficientCapsNet


# some parameters
model_name = 'MNIST'
custom_path = './trained_model/original_capsnet_MNIST_new_train.h5'
plot = False
gpu_no = 1

if gpu_no >= 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu_no], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_no], True)

dataset = Dataset(model_name, config_path='config.json')

if plot:
    n_images = 20 # number of images to be plotted
    plotImages(dataset.X_test[:n_images,...,0], dataset.y_test[:n_images], n_images, dataset.class_names)

# Load Model
model_test = EfficientCapsNet(model_name, mode='test', verbose=True, custom_path=custom_path)
model_test.load_graph_weights() # load graph weights (bin folder)

# Test Model
model_test.evaluate(dataset.X_test, dataset.y_test) # if "smallnorb" use X_test_patch
y_pred = model_test.predict(dataset.X_test)[0] # if "smallnorb" use X_test_patch

if plot:
    n_images = 20
    plotWrongImages(dataset.X_test, dataset.y_test, y_pred, # if "smallnorb" use X_test_patch
                    n_images, dataset.class_names)
