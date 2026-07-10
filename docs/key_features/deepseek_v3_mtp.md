# DeepSeek-V3 Multi-Token Prediction

VeOmni trains the Multi-Token Prediction (MTP) modules included in official
DeepSeek-V3 checkpoints when `num_nextn_predict_layers` is greater than zero.
The default DeepSeek-V3 671B configuration enables one predictor and uses:

```json
{
  "num_nextn_predict_layers": 1,
  "mtp_loss_weight": 0.1
}
```

The total objective is the normal next-token loss plus the mean MTP loss,
scaled by `mtp_loss_weight`. MTP targets respect `IGNORE_INDEX`, packed-sample
boundaries, and sequence-parallel partitions. Setting
`num_nextn_predict_layers` to zero disables construction and training of the
MTP modules.

Official checkpoints repeat the shared token embedding, final norm, and LM
head under the MTP layer keyspace. VeOmni consumes those aliases during load
and keeps a single registered owner for each shared parameter, which avoids
placing one parameter in multiple nested FSDP2 units.
