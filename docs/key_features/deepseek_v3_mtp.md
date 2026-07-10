# DeepSeek-V3 Multi-Token Prediction

VeOmni can optionally train the Multi-Token Prediction (MTP) modules included
in official DeepSeek-V3 checkpoints. The model's `num_nextn_predict_layers`
field describes checkpoint structure; training is disabled by default and is
controlled independently through training arguments:

```yaml
train:
  enable_mtp: true
  mtp_loss_weight: 0.1
```

The total objective is the normal next-token loss plus the mean MTP loss,
scaled by `train.mtp_loss_weight`. MTP targets respect `IGNORE_INDEX`, packed-sample
boundaries, and sequence-parallel partitions. Setting
`train.enable_mtp: false` does not construct or load the MTP decoder modules;
their checkpoint tensors are consumed by the loader without allocating model,
FSDP, gradient, or optimizer storage. Enabling MTP for a model with
`num_nextn_predict_layers == 0` raises a configuration error.

Official checkpoints repeat the shared token embedding, final norm, and LM
head under the MTP layer keyspace. VeOmni consumes those aliases during load
and keeps a single registered owner for each shared parameter, which avoids
placing one parameter in multiple nested FSDP2 units.
