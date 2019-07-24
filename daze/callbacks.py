import os
import io

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def checkpoints(interval=1):
    def _checkpoints(model, **info_dict):
        if not info_dict["type"] == "epoch": return
        current_epoch = info_dict["current_epoch"]
        if current_epoch % interval == 0:
            model.save_weights(
                os.path.join(info_dict["checkpoint_dir"], f"epoch{current_epoch}")
            )

    return _checkpoints


def tensorboard_loss_scalars():
    def _tensorboard_loss_scalars(model, **info_dict):
        if not info_dict["type"] == "epoch": return
        log_writer = tf.summary.create_file_writer(info_dict["log_dir"])
        train_loss = info_dict["train_loss"]
        val_loss = info_dict["val_loss"]
        epoch = info_dict["current_epoch"]
        with log_writer.as_default():
            tf.summary.scalar("train_loss", train_loss.result(), step=epoch)
            tf.summary.scalar("val_loss", val_loss.result(), step=epoch)
    
    return _tensorboard_loss_scalars

def tensorboard_grad_histograms(freq=1):
    def _tensorboard_grad_histograms(model, **info_dict):
        if not info_dict["type"] == "batch": return
        current_step = info_dict["current_step"]
        if not current_step % freq == 0:
            return
        gradients = info_dict["gradients"]
        log_writer = tf.summary.create_file_writer(info_dict["log_dir"])
        with log_writer.as_default():
            for idx, grad in enumerate(gradients):
                tf.summary.histogram(f"grad_{idx}", grad, step=current_step)

    return _tensorboard_grad_histograms

def _plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def _adjust_for_imshow(img):
    if img.shape[-1] == 1:
        img = np.squeeze(img, -1)
    return img

def _reconstruction_acc_figure(true, pred):
    fig = plt.figure(figsize=(20,20))
    rows = true.shape[0]
    columns = 2
    f, axarr = plt.subplots(rows, columns)
    for row in range(rows):
        img = _adjust_for_imshow(true[row, ...])
        axarr[row, 0].imshow(img)
        reconstruction = _adjust_for_imshow(pred[row,...])
        axarr[row, 1].imshow(reconstruction)
    return fig

def tensorboard_image_reconstruction(examples):
    def _tensorboard_image_reconstruction(model, **info_dict):
        if not info_dict["type"] == "epoch": return
        pred = model.predict(examples)
        fig_img = _plot_to_image(_reconstruction_acc_figure(examples, pred))
        log_writer = tf.summary.create_file_writer(info_dict["log_dir"])
        current_epoch = info_dict["current_epoch"]
        with log_writer.as_default():
            tf.summary.image('Reconstruction', fig_img, step=current_epoch)
    return _tensorboard_image_reconstruction
