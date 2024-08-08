#!/usr/bin/env python

import os
import imageio
import pathlib

data_set = "spcas9"
image_path = pathlib.Path(os.environ["HOME"]) / "sdc1" / "SX" / data_set / "output" / "images"
for date in os.listdir(image_path):
    for epoch in os.listdir(image_path / date):
        for valid_epoch in os.listdir(image_path / date / epoch):
            for valid_batch in os.listdir(image_path / date / epoch / valid_epoch):
                for in_batch in os.listdir(image_path / date / epoch / valid_epoch / valid_batch):
                    if not os.path.isdir(image_path / date / epoch / valid_epoch / valid_batch / in_batch):
                        continue
                    image_files = os.listdir(image_path / date / epoch / valid_epoch / valid_batch / in_batch)
                    time_steps = max([int(os.path.splitext(image_file)[0]) for image_file in image_files])
                    save_dir = image_path / date / epoch / valid_epoch / valid_batch
                    os.makedirs(save_dir, exist_ok=True)
                    with imageio.get_writer(save_dir / f"{in_batch}.gif", mode='I', duration=300) as writer:
                        for time_step in range(time_steps - 1, -1, -1):
                            writer.append_data(
                                imageio.imread(
                                    image_path / date / epoch / valid_epoch / valid_batch / in_batch / f"{time_step}.png"
                                )
                            )