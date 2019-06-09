
import tensorflow as tf

@tf.function
def train(dataset, autoencoder, epochs, checkpoint_dir, checkpoint_freq, log_dir):
    # Create tf Dataset
    sample_count = dataset.shape[0] #channels last format
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.shuffle(sample_count + 1).batch(64)

    # Create optimizer
    optimizer = tf.train.AdamOptimizer(lr=1e-3)

    # Create model
    model = autoencoder

    # Setup checkpoints and logging
    checkpoint_dir = os.path.dirname(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    log_dir = os.path.dirname(log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_writer = tf.summary.create_file_writer(log_dir)
    log_writer.set_as_default()


    for epoch in range(epochs):
        # Metrics
        avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)

        for x, y in dataset:
            with tf.GradientTape() as tape:
                reconstruction = model(x)
                loss = tf.reduce_mean(tf.square(x - reconstruction))
                avg_loss.update_state(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.get_or_create_global_step())
        
        # Logging
        tf.summary.scaler('loss', avg_loss.result(), step=optimizer.iterations)
        avg_loss.reset_states()
        
        # Checkpointing
        if epoch % checkpoint_freq == 0:
            model.save_weights(os.path.join(checkpoint_dir, f"{epoch_{epoch}"), save_format='tf')
