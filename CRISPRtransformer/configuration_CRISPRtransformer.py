from transformers import RoFormerConfig

class CRISPRtransformerConfig(RoFormerConfig):
    model_type = "CRISPR transformer"

    def __init__(
        self,
        vocab_size = 5,
        hidden_size = 512, # model dimension
        num_hidden_layers = 6, # number of EncoderLayer
        num_attention_heads = 8,
        intermediate_size = 2048, # FeedForward intermediate dimension size
        hidden_dropout_prob = 0.1, # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
        attention_probs_dropout_prob = 0.1, # The dropout ratio for the attention probabilities
        max_position_embeddings = 256, # The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).
        output_size = 128 * 128, # size of the output logits, which equals (ref1len + 1) * (ref2len + 1)
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.output_size = output_size
        super().__init__(**kwargs)
