import argparse

import numpy as np
import umap

import daze as dz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_neighbors', type=int, default=15)
    parser.add_argument('-min_dist', type=float, default=.1)
    parser.add_argument('-n_components', type=int, default=3)
    parser.add_argument("-output", type=str)
    parser.add_argument("-dataset", type=str, choices=["mnist", "cifar", "cifar10"])
    args = parser.parse_args()

    if args.dataset in ["cifar", "cifar10"]:
        dataset = dz.data.cifar10
        flatten_shape = 32*32*3
    elif args.dataset in ["mnist"]:
        dataset = dz.data.mnist
        flatten_shape = 28*28

    data, _ = dataset.load(dtype="f")
    data /= 255.0
    np.random.shuffle(data)
    data = np.reshape(data, (-1, flatten_shape))

    reducer = umap.UMAP(n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist,
                        n_components=args.n_components)
    encodings = reducer.fit_transform(data)
    np.savetxt(args.output, encodings)