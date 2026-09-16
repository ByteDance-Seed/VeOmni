"""DeepSeek-V4.1-Flash OmniModule family.

Splits the monolithic checkpoint into four modules:

* ``deepseek_v41_vision``       -- DeepSeek-ViT + aligner (0.49B)
* ``deepseek_v41_text_encoder`` -- embed_tokens + lm_head + tokenizer + chat template (1.32B)
* ``deepseek_v41_ngram``        -- the two n-gram memory tables and their hashing (196.6B)
* ``deepseek_v41_llm``          -- the 40-layer CED backbone, the n-gram connection
                                   (``wkv`` + gate), and the DSpark draft stages

The n-gram split follows the lookup: gathering the rows depends only on
``input_ids`` and the image mask, so it runs ahead of the backbone as its own
node, while ``wkv`` and the gate read the mHC residual stream at layers 1 and 14
and stay inside the backbone.
"""

from . import ngram, text_encoder, vision  # noqa: F401
