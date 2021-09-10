import tensorflow_addons as tfa
import tensorflow as tf
from termcolor import colored 

AUTO = { 'num_parallel_calls': tf.data.AUTOTUNE }




##### Important Functions #####
def save_weights(model, filepath):
    filepath = str(filepath)
    print('Saving model weights at', colored(filepath, 'blue'))
    model.save_weights(filepath=filepath, options=get_save_locally())

def get_gcs_path(dataset_name): 
    from kaggle_datasets import KaggleDatasets
    return KaggleDatasets().get_gcs_path(dataset_name)
    
def get_save_locally(): 
    return tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

def get_load_locally(): 
    return tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')


##### Factory Functions #####




