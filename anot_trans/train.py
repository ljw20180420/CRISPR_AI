from .transformer import Encoder, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward

N = 3 # number of EncoderLayer
d_model = 512 # model dimension
h = 8 # head number
d_ff = 2048 # FeedForward intermediate dimension size
dropout = 0.1

encoder = Encoder(
    EncoderLayer(
        size=d_model,
        self_attn=MultiHeadedAttention(h, d_model),
        feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout),
        dropout=dropout
    ),
    N
)