import argparse
import tensorflow as tf
from utils import Dataset, plotImages, plotWrongImages, plotHistory
from models import EfficientCapsNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--gpu', type=int, default=-1)

    return parser.parse_args()


# some parameters
args = parse_args()
model_name = 'MNIST'
plot = args.plot
gpu_no = args.gpu

if gpu_no >= 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu_no], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_no], True)

dataset = Dataset(model_name, config_path='config.json')
print("========================")
print(dataset.X_train.shape)
print(dataset.X_test.shape)
print("========================")

if plot:
    n_images = 20 # number of images to be plotted
    plotImages(dataset.X_test[:n_images,...,0], dataset.y_test[:n_images], n_images, dataset.class_names)

model_train = EfficientCapsNet(model_name, mode='train', verbose=True)
dataset_train, dataset_val = dataset.get_tf_data()

history = model_train.train(dataset, initial_epoch=0)
if plot: plotHistory(history)
