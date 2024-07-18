from diffusers.models import UNet2DModel
from .config import args, ref1len, ref2len
from .load_data import train_dataloader

for batch in train_dataloader:
    break

if args.model == 'UNet2DModel':
    model = UNet2DModel(
        sample_size=(ref2len, ref1len),
        in_channels=batch["condition"].shape[1],
        out_channels=1,
        center_input_sample=True, # this needs to normalize micro-homology strength to 0~1,
        block_out_channels=(224, 448, 672, 896),
        dropout=0.01
    )
