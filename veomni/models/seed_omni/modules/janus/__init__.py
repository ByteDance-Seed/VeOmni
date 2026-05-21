"""Janus-1.3B OmniModule mixins.

Splits the monolithic ``JanusForConditionalGeneration`` into composable
sub-modules that match the SeedOmni V2 graph runtime (:mod:`veomni.models.
seed_omni`).  Each sub-module lives in its own folder under
``janus/<sub_module>/`` and contains short-named files
(``configuration.py``, ``modeling.py``, optional ``processing.py``) —
the folder name carries the namespace, so the file names don't repeat
``janus_<sub_module>`` again.

The classes register themselves with HuggingFace ``AutoConfig`` /
``AutoModel`` so ``from_pretrained`` works against the
``<output_dir>/<sub_module>`` checkpoint folders produced by
``scripts/split_janus.py``.

Sub-modules
-----------
:class:`JanusSiglip` (folder ``siglip/``)
    SigLIP vision tower + MLP aligner (model_type ``janus_siglip``).
    Produces understanding image embeddings (``image_embeds``) projected
    into the LLM hidden space.

:class:`JanusVqvae` (folder ``vqvae/``)
    VQVAE + generation embeddings + generation aligner + generation head
    (model_type ``janus_vqvae``).  Two call-site methods:

    * ``encode``  — pixels → ``gen_embeds`` + ``vq_token_ids``
                    (training teacher-forcing).
    * ``decode``  — unified VQ head; covers training CE
                    (``hidden + gt_token_ids → _loss``) and inference
                    sample-then-lookup (``hidden → vq_token_id + embed``)
                    in one method, dispatched by inputs.

:class:`JanusLlama` (folder ``llama/``)
    LLaMA backbone (model_type ``janus_llama``) — *no* word-token
    embeddings, *no* LM head.  Those vocab-bound layers are owned by the
    sibling :class:`~veomni.models.seed_omni.modules.base.TextEmbed`
    module.  Multi-modal embedding scatter
    (``und_image_embeds`` / ``gen_image_embeds`` →
    ``inputs_embeds`` placeholder positions) lives in :meth:`pre_forward`.

:class:`JanusTextEmbed` (folder ``text_embed/``)
    :class:`TextEmbed` + Janus-specific
    :code:`<begin_of_image>` / :code:`<end_of_image>` boundary-token
    emitters (``emit_image_start`` / ``emit_image_end``).  model_type
    ``janus_text_embed``.  Used for the ``image_vq_start`` /
    ``image_vq_end`` bridge states in the Janus inference FSM —
    boundary tokens are a model concern, not a framework concern.
"""

from .llama import JanusLlama, JanusLlamaConfig
from .siglip import JanusSiglip, JanusSiglipConfig, JanusSiglipProcessor
from .text_embed import JanusTextEmbed, JanusTextEmbedConfig
from .vqvae import JanusVqvae, JanusVqvaeConfig, JanusVqvaeProcessor


__all__ = [
    "JanusLlama",
    "JanusLlamaConfig",
    "JanusSiglip",
    "JanusSiglipConfig",
    "JanusSiglipProcessor",
    "JanusTextEmbed",
    "JanusTextEmbedConfig",
    "JanusVqvae",
    "JanusVqvaeConfig",
    "JanusVqvaeProcessor",
]
