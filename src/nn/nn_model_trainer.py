import tensorflow as tf
import math
import os
# from joblib import dump

import pickle

checkpoint_path = '/Users/hbojja/uiuc/CS445-CP/FinalProject/nn_logs'


def compile_fit(model, train_set, val_set, train_set_length, val_set_length, batch_size):
    """Compile and fit the input model with given training set.
        Arguments:
            model: Fully built model and ready to train
            train_set: training data set
            val_set: validation data set
            train_set_length: length of the training data set
            val_set_length: length of the validation data set
            batch_size: batch size of for iterations
        :returns:
            model: model after fitting the training data. use this to predict values
            model_history: training history for debugging reasons
        """

    steps_per_epoch = 50
    validation_steps = 10
    print("starting the training. Steps/epoch: ", steps_per_epoch, " and validation steps: ", validation_steps)

    # Last good lr = 0.00005
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00008)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'mean_absolute_error'
            # tf.keras.metrics.MeanSquaredError()
        ])

    model_history = model.fit(train_set,
                              epochs=16,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=val_set,
                              validation_steps=validation_steps
                              # callbacks=get_callbacks()
                              )
    return model, model_history


def get_callbacks():
    """Compile and fit the input model with given training set.
            :returns:
                returns the array of call backs required during training.
            """
    check_point = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1)

    return [
        tf.keras.callbacks.EarlyStopping(patience=10),
        # tf.keras.callbacks.TensorBoard(logdir/name),
        # check_point
    ]


def saveModels(model, scaler, vectorizer, target_dir_root):
    """Use this method to save objects used during the training process.
            """
    print("Saving the models to: ", target_dir_root)

    model_dir = target_dir_root + "/model/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save(model_dir)

    with open(target_dir_root + "/scaler", 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    with open(target_dir_root + "/vec", 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

    # pickle.dump(scaler, target_dir_root + "/scaler")
    # pickle.dump(vectorizer, target_dir_root + "/vec")

    print("Finished saving the models.....")
