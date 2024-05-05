import warnings
from collections import OrderedDict
from typing import Any, Mapping, Optional

from transformers import PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from transformers.onnx.utils import compute_effective_axis_dimension
from transformers.utils import TensorType, is_torch_available, logging


logger = logging.get_logger(__name__)




class TranExtracConfig(PretrainedConfig):
    model_type = "TranExtrac"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=1,
        encoder_ffn_dim=3072,
        encoder_attention_heads=8,
        decoder_layers=3, 
        decoder_ffn_dim=3072,
        decoder_attention_heads=8,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=768,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        use_cache=True,
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=0,
        is_encoder_decoder=True,
        decoder_start_token_id=0, 
        forced_eos_token_id=0,
        pretrained_model_path=None,
        ext_ff_size=2048,
        ext_heads=8,
        ext_dropout=0.2,
        ext_layers=2,
        max_position_input=512,
        param_init=0,
        param_init_glorot=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.pretrained_model_path = pretrained_model_path

        self.ext_ff_size=ext_ff_size
        self.ext_heads=ext_heads
        self.ext_dropout=ext_dropout
        self.ext_layers=ext_layers
        self.max_position_input=max_position_input
        self.param_init=param_init
        self.param_init_glorot=param_init_glorot
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

        if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
            self.forced_bos_token_id = self.bos_token_id
            warnings.warn(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions. "
                "The config can simply be saved and uploaded again to be fixed."
            )