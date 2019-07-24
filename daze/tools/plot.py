import argparse

import numpy as np
import matplotlib.pyplot as plt


def maximize_window():
    mng = plt.get_current_fig_manager()
    backend = plt.get_backend()
    if backend == "TkAgg":
        mng.window.state("zoomed")
    elif backend == "wxAgg":
        mng.frame.Maximize(True)
    elif backend == "Qt4Agg":
        mng.window.showMaximized()
    else:
        print(f"maximize_window() unable to cope with backend: {backend}")
        pass


def attempt_switch_from_macosx():
    if not plt.get_backend() == "MacOSX":
        return True
    backend_list = ["Qt4Agg", "wXAgg", "TkAgg"]
    for backend in backend_list:
        try:
            plt.switch_backend(backend)
        except:
            continue
        else:
            return True
    return False

def main():
    attempt_switch_from_macosx()

    parser = argparse.ArgumentParser()
    parser.add_argument("-datafile", type=str)
    args = parser.parse_args()

    data = np.loadtxt(args.datafile)
    rows = data.shape[0]
    cols = data.shape[1]

    if cols == 3:
        plot3d(data)
    elif cols == 2:
        plot2d(data)
    elif cols == 1:
        plot1d(data)
    else:
        raise ValueError(f"plot.py cannot visualize data of dimension: {cols}")

    maximize_window()
    plt.show()


def plot3d(data):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(*np.split(data, 3, axis=-1))
    return fig


def plot2d(data):
    raise NotImplementedError()


def plot1d(data):
    raise NotImplementedError()


if __name__ == "__main__":
    main()
