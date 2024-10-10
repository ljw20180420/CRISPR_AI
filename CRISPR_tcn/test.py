from darts.datasets import WeatherDataset
from darts.models import TCNModel
series = WeatherDataset().load()
# predicting atmospheric pressure
target = series['p (mbar)'][:100]
# optionally, use past observed rainfall (pretending to be unknown beyond index 100)
past_cov = series['rain (mm)'][:100]
# `output_chunk_length` must be strictly smaller than `input_chunk_length`
model = TCNModel(
    input_chunk_length=12,
    output_chunk_length=6,
    n_epochs=20,
)
model.fit(target, past_covariates=past_cov)
pred = model.predict(6)
pred.values()
array([[-80.48476824],
        [-80.47896667],
        [-41.77135603],
        [-41.76158729],
        [-41.76854107],
        [-41.78166819]])
