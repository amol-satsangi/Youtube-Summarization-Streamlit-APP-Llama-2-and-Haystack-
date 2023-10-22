from haystack.nodes import PromptModetInvocationLayer
from haystack.nodes.prompt.invocation_layer import DefaultTokenStreamingHandter
from llama_cpp import Llama
import os
from typing import Dict, List, Union, Type, Optional
import logging

logger = logging.getLogger(__name__)

class LlamaCPPInvocationLayer(PromptModelInvocationLayer):
    def __init__(self, model_name_or_path: Union[str, os.PathLike],
        max_length: Optional [int] = 128,
        max_context: Optional[int] = 32000,
        n_parts = Optional[int] = -1,
        seed: Optional[int] = 1337,
        f16_kv: Optional[bool] = True,
        logits_all: Optional[bool] = False,
        vocab_onty: Optional [bool] = False,
        use_mmap: Optional [bool] = True,
        use_mlock : Optional [bool] = False,
        embedding : Optional [bool] = False
        n_threads: Optional[int] = None,
        n_batch: Optional[int] = 512
        last_n_tokens_size: Optional[int] = 64,
        lora_base: Optional[str] = None,
        lora_path: Optional[str] = None,
        verbose: Optional[bool] = True,
        **kwargs):

        if model_name_or_path is None or len(model_name_or_path) == 0:
            raise ValueError("model_name_or_path cannot be None or empty string")

        self.model.