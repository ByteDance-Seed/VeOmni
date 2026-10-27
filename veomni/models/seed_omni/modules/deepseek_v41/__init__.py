"""DeepSeek-V4.1-Flash OmniModule family.

Splits the monolithic checkpoint into four modules:

* ``deepseek_v41_vision``       -- DeepSeek-ViT + aligner (0.49B)
* ``deepseek_v41_text_encoder`` -- embed_tokens + lm_head + tokenizer + chat template (1.32B)
* ``deepseek_v41_ngram``        -- the two n-gram memory tables and their hashing (196.6B)
* ``deepseek_v41_llm``          -- the 40-layer CED backbone, DSpark draft stages, and the n-gram connection
                                   (``wkv`` + gate) (550.6B, of which 543.4B is the
                                   routed experts)

748.98B in total, which is what the four modules report when built from the
released ``config.json``. DSpark lives under ``mtp.*`` inside ``deepseek_v41_llm``.

The n-gram split follows the lookup: gathering the rows depends only on
``input_ids`` and the image mask, so it runs ahead of the backbone as its own
node, while ``wkv`` and the gate read the mHC residual stream at layers 1 and 14
and stay inside the backbone.
"""

from . import convert_model, hf_layout, llm, ngram, text_encoder, vision  # noqa: F401
