try: 
    import tensorflow_hub as hub 
except: 
    print('TFHub not availible')
    _HAS_TFHUB = False
try: 
    import tensorflow_addons as tfa
except: 
    print('Tensorflow Addons not availible')
    _HAS_TFA = False

import tensorflow as tf

def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy

def mixed_precision(): 
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

def jit(): 
    tf.config.optimizer.set_jit(True)