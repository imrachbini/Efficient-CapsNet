import argparse
import tensorflow as tf
from utils import Dataset, plotImages, plotWrongImages
from models import EfficientCapsNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model_path', type=str, default=None)

    return parser.parse_args()


# some parameters
args = parse_args()
model_name = 'MNIST'
custom_path = args.model_path
plot = args.plot
gpu_no = args.gpu

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
