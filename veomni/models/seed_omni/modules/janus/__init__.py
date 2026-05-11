"""Janus-1.3B OmniModule implementations.

Splits the monolithic JanusForConditionalGeneration into three composable
OmniModule sub-modules that can be independently FSDP-wrapped and assembled
into an OmniModel via config-driven connections.

Sub-modules
-----------
JanusVisionEncoder
    SigLIP vision tower + MLP aligner.  Produces understanding image embeddings
    that are injected into the AR-LLM as ``und_image_embeds``.

JanusVQDecoder
    VQVAE + generation embeddings + generation aligner + generation head.
    During training it encodes images to VQ tokens and computes the VQ
    embedding loss.  During inference it acts as:
      (a) an *encode* step: image patches → VQ token IDs (pre-AR prefix).
      (b) a *decode embed* step: VQ token ID → codebook embed → LLM input.
      (c) a *decode pixels* step: sequence of VQ token IDs → pixel values.

JanusLLM
    LLaMA language model + LM head.  Accepts ``input_ids``, optional
    ``inputs_embeds``, and optional ``und_image_embeds`` / ``gen_image_embeds``
    injection tensors.
"""

from .modeling_janus_llm import JanusLLM
from .modeling_janus_vision_encoder import JanusVisionEncoder
from .modeling_janus_vq_decoder import JanusVQDecoder


__all__ = ["JanusVisionEncoder", "JanusVQDecoder", "JanusLLM"]
