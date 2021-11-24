import argparse
import os
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import numpy as np
import io

STAGE = "creating base model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    ## get the data
    mnist = tf.keras.datasets.mnist

    (X_train_full,y_train_full),(X_test, y_test)=mnist.load_data()
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    
    ## set the seeds
    seed = 2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ## define layers
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300, name="hiddenLayer1"),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.Dense(100, name="hiddenLayer2"),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")
          ]
    
    ## define model
    model =  tf.keras.models.Sequential(LAYERS)

    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate = 1e-3)
    METRICS = ["accuracy"]
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

    def _log_model_summary(full_model):
        with io.StringIO() as stream:
            full_model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

    logging.info(f"full model summary: \n{_log_model_summary(model)}")
    #model.summary()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e