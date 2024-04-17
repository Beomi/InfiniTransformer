from transformers import GemmaConfig as OriginalGemmaConfig


class GemmaConfig(OriginalGemmaConfig):
    def __init__(
        self,
        vocab_size=256000,
        hidden_size=3072,
        intermediate_size=24576,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        hidden_act="gelu_pytorch_tanh",
        hidden_activation=None,
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000,
        attention_bias=False,
        attention_dropout=0,
        segment_size=2048,
        **kwargs
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_act,
            hidden_activation,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            pad_token_id,
            eos_token_id,
            bos_token_id,
            tie_word_embeddings,
            rope_theta,
            attention_bias,
            attention_dropout,
            **kwargs
        )
        self.segment_size = segment_size
