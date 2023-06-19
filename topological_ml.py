"""First attempt at doing machine learning on persistence landscapes.

**NOTE: This code is not tested and most likely contains bugs! Do not use it
without thoroughly vetting every line of code!**
"""
import pickle
import os

from fire import Fire
from tqdm import tqdm

import persim.landscapes
import numpy as np


def generate_features(data_dir: str, skip: bool) -> dict:
    """Loads or Generates topology features for downstream machine learning"""

    save_fname = os.path.join(data_dir, "ml_features.pkl")
    if skip:
        save_fname = os.path.join(data_dir, "ml_features_skip.pkl")

    if os.path.exists(save_fname):
        print("Features pkl file found!")
        with open(save_fname, "rb") as f:
            data_dict = pickle.load(f)

    else:
        osa_fname = "pl_data_bands.pkl"
        if skip:
            osa_fname = "pl_data_bands_skip.pkl"

        fname = os.path.join(data_dir, osa_fname)

        with open(fname, "rb") as f:
            data = pickle.load(f)

        channels = ["delta", "theta", "alpha", "beta", "gamma"]
        data_dict = {}

        for channel in tqdm(channels):
            X_raw = []
            y = []

            # Extracting all segment PLs with associated labels
            for k, v in data.items():
                label = False
                if "True" in k:
                    label = True

                for segment in v:
                    X_raw.append(segment[channel])
                    y.append(label)

            # Finding minimum start point and maximum end point
            start = min([p.start for p in X_raw])
            stop = max([p.stop for p in X_raw])
            num_steps = max([p.num_steps for p in X_raw])

            # Putting PLs on same grid
            X_pl = persim.landscapes.snap_pl(
                X_raw, start=start, stop=stop, num_steps=num_steps
            )
            X_pl = [p.values.flatten() for p in X_pl]
            max_size = max([x.shape[0] for x in X_pl])

            X_arr = np.zeros((len(X_pl), max_size))

            for idx, pl_arr in enumerate(X_pl):
                X_arr[idx, : len(pl_arr)] = pl_arr

            data_dict[channel] = {"features": X_arr, "labels": y}

        with open(save_fname, "wb") as f:
            pickle.dump(data_dict, f)

    return data_dict


def main(data_dir: str, skip: bool = False) -> None:
    data_dict = generate_features(data_dir, skip)
    return
