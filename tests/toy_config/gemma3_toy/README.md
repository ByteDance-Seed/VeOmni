# Gemma 3 Text Toy Config

Based on the text configuration of
[`google/gemma-3-270m`](https://huggingface.co/google/gemma-3-270m/blob/main/config.json).

The test fixture keeps Gemma 3's alternating sliding/full attention semantics,
but reduces the model to two layers with hidden size 64, four query heads, two
KV heads, head dimension 16, vocabulary size 128, and maximum sequence length
64. The sliding window is reduced to four tokens so short tests exercise both
mask types. Logit softcapping is disabled because the fixture also covers
VeOmni's generic fused-loss contract.
