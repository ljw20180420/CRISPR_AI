from plotnine import ggplot, geom_point, aes
import pandas as pd
from load_data import train_dataloader
for batch in train_dataloader:
    break

ggplot(
    pd.DataFrame({'idx': range(len(batch['count'][0])), 'count': batch['count'][0].cpu()}), aes(x='idx', y='count')
) + geom_point()

