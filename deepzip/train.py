import time

import tensorflow as tf

def baseline_train(autoencoder, x_train, x_val, epochs, experiment_name):
    directory += str(int(time.time()))

    # Setup checkpoints and logging
    checkpoint_dir = os.path.join(directory, 'checkpoints')
    os.mkdir(checkpoint_dir)
    
    log_dir = os.path.join(directory, 'logs')
    log_dir += time_now
    os.mkdir(log_dir)

    # Create optimizer
    optimizer = tf.train.AdamOptimizer(lr=1e-3)

    # Create model
    autoencoder.compile(optimizer, loss='mse')

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir, write_graph=True, write_images=True, update_freq='epoch'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, monitor='val_acc', save_best_only=True, save_weights_only=False, period=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=2, restore_best_weights=True),
    ]

    return autoencoder.fit(
        x=x_train,
        y=x_train,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
        validation_data=(x_val, x_val),
        shuffle=True,
    )