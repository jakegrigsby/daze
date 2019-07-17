import os

import tensorflow as tf


def checkpoints(interval):
    def _checkpoints(model, **train_dict):
        current_epoch = train_dict["current_epoch"]
        if current_epoch % interval == 0:
            model.save_weights(
                os.path.join(train_dict["checkpoint_dir"], f"epoch{current_epoch}")
            )

    return _checkpoints


def tensorboard():
    def _tensorboard(model, **train_dict):
        log_writer = tf.summary.create_file_writer(train_dict["log_dir"])
        train_loss = train_dict["train_loss"]
        val_loss = train_dict["val_loss"]
        epoch = train_dict["current_epoch"]
        with log_writer.as_default():
            tf.summary.scalar("train_loss", train_loss.result(), step=epoch)
            tf.summary.scalar("val_loss", val_loss.result(), step=epoch)

    return _tensorboard
