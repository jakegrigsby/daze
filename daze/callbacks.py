import os
import io

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import daze as dz


class EpochCallback:
    def time_to_run(self, **info_dict):
        time_to_run = False
        if info_dict["type"] == "epoch":
            time_to_run = True
        return time_to_run

class SingleUseCallback:
    been_called = True
    def time_to_run(self, **info_dict):
        if self.been_called:
            self.been_called = False
            return True
        return False


class BatchCallback:
    def time_to_run(self, **info_dict):
        time_to_run = False
        if info_dict["type"] == "batch":
            time_to_run = True
        return time_to_run


class Checkpoints(EpochCallback):
    def __init__(self, interval=1):
        self.interval = interval

    def __call__(self, model, **info_dict):
        if not self.time_to_run(**info_dict):
            return
        current_epoch = info_dict["current_epoch"]
        if current_epoch % self.interval == 0:
            model.save_weights(
                os.path.join(info_dict["checkpoint_dir"], f"epoch{current_epoch}")
            )


checkpoints = Checkpoints


class TensorboardLossScalars(EpochCallback):
    def __call__(self, model, **info_dict):
        if not self.time_to_run(**info_dict):
            return
        log_writer = tf.summary.create_file_writer(info_dict["log_dir"])
        train_loss = info_dict["train_loss"]
        val_loss = info_dict["val_loss"]
        epoch = info_dict["current_epoch"]
        with log_writer.as_default():
            tf.summary.scalar("train_loss", train_loss.result(), step=epoch)
            tf.summary.scalar("val_loss", val_loss.result(), step=epoch)


tensorboard_loss_scalars = TensorboardLossScalars


class TensorboardGradientHistograms(BatchCallback):
    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, model, **info_dict):
        if not self.time_to_run(**info_dict):
            return
        current_step = info_dict["current_step"]
        if not current_step % self.frequency == 0:
            return
        gradients = info_dict["gradients"]
        log_writer = tf.summary.create_file_writer(info_dict["log_dir"])
        with log_writer.as_default():
            for idx, grad in enumerate(gradients):
                tf.summary.histogram(f"grad_{idx}", grad, step=current_step)


tensorboard_gradient_histograms = TensorboardGradientHistograms


def _plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
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
    fig = plt.figure(figsize=(20, 20))
    rows = true.shape[0]
    columns = 2
    f, axarr = plt.subplots(rows, columns)
    for row in range(rows):
        img = _adjust_for_imshow(true[row, ...])
        axarr[row, 0].imshow(img)
        reconstruction = _adjust_for_imshow(pred[row, ...])
        axarr[row, 1].imshow(reconstruction)
    return fig


class TensorboardImageReconstruction(EpochCallback):
    def __init__(self, examples):
        self.examples = examples

    def __call__(self, model, **info_dict):
        if not self.time_to_run(**info_dict):
            return
        plt.clf()
        pred = model.predict(self.examples)
        fig = _reconstruction_acc_figure(self.examples, pred)
        img = _plot_to_image(fig)
        log_writer = tf.summary.create_file_writer(info_dict["log_dir"])
        current_epoch = info_dict["current_epoch"]
        with log_writer.as_default():
            tf.summary.image("Reconstruction", img, step=current_epoch)


tensorboard_image_reconstruction = TensorboardImageReconstruction

class TensorboardLatentSpacePlot(EpochCallback):
    def __init__(self, examples):
        self.examples = examples
        self.compatible = True
    
    def __call__(self, model, **info_dict):
        if not self.compatible or not self.time_to_run(**info_dict): return
        plt.clf()
        encodings = model.get_batch_encodings(self.examples)
        fig = None
        if encodings.shape[-1] == 3:
            fig = dz.tools.plot.plot3d(encodings)
        elif encodings.shape[-1] == 2:
            fig = dz.tools.plot.plot2d(encodings)
        elif encodings.shape[-1] == 1:
            fig = dz.tools.plot.plot1d(encodings)
        else:
            print("Warning: the TensorboardLatentSpacePlot callback has no effect "
                  f"with latent space size > 3. Detected latent space of size {encodings.shape[-1]}")
            self.compatible = False
            return
        img = _plot_to_image(fig)
        log_writer = tf.summary.create_file_writer(info_dict["log_dir"])
        current_step = info_dict["current_epoch"]
        with log_writer.as_default():
            tf.summary.image('Latent Space', img, step=current_step)

tensorboard_latent_space_plot = TensorboardLatentSpacePlot

class TensorboardTraceGraph(SingleUseCallback):
    def __init__(self, function, *inputs, graph=True, profiler=True):
        self.func = function
        self.inputs = inputs
        self.graph = graph
        self.profiler = profiler
    
    def __call__(self, model, **info_dict):
        if not self.time_to_run(**info_dict): return
        log_writer = tf.summary.create_file_writer(info_dict["log_dir"])
        tf.summary.trace_on(graph=self.graph, profiler=self.profiler)
        self.func(*self.inputs)
        with log_writer.as_default():
            tf.summary.trace_export(
                name=f"Graph of {self.func.__name__}",
                step=0,
                profiler_outdir=info_dict["log_dir"]
            )

tensorboard_trace_graph = TensorboardTraceGraph


