import math
from matplotlib import pyplot as plt
from collections import deque
from tensorflow.keras.callbacks import Callback


class MetricsHistory(Callback):
    """Tracks and plot accuracy and loss in real-time"""

    def __init__(self):
        # Initialization
        # self.display = RealTimePlot(max_entries=20)
        pass

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("\nStart epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        # Add point to plot
        # self.display.add(x=epoch,
        #                 y_train=logs.get('accuracy'),
        #                 y_validation=logs.get('val_accuracy'))
        plt.pause(0.1)

        keys = list(logs.keys())
        print("\nEnd epoch {} of training; got log keys: {}".format(epoch, keys))


    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("\nEnd epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_train_begin(self, logs=None):
        print("\nStarting training")

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("\nEvaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))
        print("\nEvaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("\nPredicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("\nPredicting: end of batch {}; got log keys: {}".format(batch, keys))


# History
class RealTimePlot(object):
    # Adapted from https://gist.github.com/Uberi/283a13b8a71a46fb4dc8
    def __init__(self, max_entries=50, x_label=r'Epochs', y_label=r'Accuracy'):
        # Store
        self.fig, self.axes = plt.subplots()
        self.max_entries = max_entries
        # axes configuration
        self.axis_x = deque(maxlen=max_entries)  # x-axis
        self.axis_y_train = deque(maxlen=max_entries)  # Training accuracy
        self.lineplot_train, = self.axes.plot([], [], "ro-")
        self.axis_y_validation = deque(maxlen=max_entries)  # Validation accuracy
        self.lineplot_validation, = self.axes.plot([], [], "bo-")
        self.axes.set_xlabel(x_label)  # Set label names
        self.axes.set_ylabel(y_label)
        # Autoscale
        self.axes.set_autoscaley_on(True)

    def add(self, x, y_train, y_validation=None):
        # Add new point
        self.axis_x.append(x)
        self.axis_y_train.append(y_train)
        self.lineplot_train.set_data(self.axis_x, self.axis_y_train)
        # if Validation accuracy is specified
        if y_validation is not None:
            self.axis_y_validation.append(y_validation)
            self.lineplot_validation.set_data(self.axis_x, self.axis_y_validation)
        # Change axis limits
        self.axes.set_xlim(self.axis_x[0], self.axis_x[-1] + 1e-15)
        self.axes.relim();
        self.axes.autoscale_view()  # Rescale the y-axis

    def animate(self, figure, callback, interval=25):
        import matplotlib.animation as animation

        def wrapper(frame_index):
            self.add(*callback(frame_index))
            self.axes.relim();
            self.axes.autoscale_view()  # Rescale the y-axis
            return self.lineplot

        animation.FuncAnimation(figure, wrapper, interval=interval)


# Visualize model history at the end of the training
def plotHistory_val(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc="upper left")
    plt.xlabel('Epoch')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc="upper left")
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')

    plt.show()


def step_decay(epoch):
    initial_lr = 1e-4
    drop = 0.5
    epochs_drop = 2
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr


def plotHistory(history, epochs):
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.legend(loc="upper left")
    plt.xlabel('Epoch')
    plt.title('Training Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.legend(loc="upper left")
    plt.title('Training Loss')
    plt.xlabel('Epoch')

    plt.show()
