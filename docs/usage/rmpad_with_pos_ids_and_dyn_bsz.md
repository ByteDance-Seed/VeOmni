# Assume the dataset has 4 samples:
* S1 = [A B C D] (len 4)
* S2 = [E F] (len 2)
* S3 = [G H I] (len 3)
* S4 = [J] (len 1)

# rmpad_with_pos_ids = True, dyn_bsz = ON
Goal: pack multiple samples into one long sequence and choose how many samples to fit per batch based on token budget.
Example packed batch (one micro‑batch):
* input_ids = [A B C D E F G H I J]
* position_ids = [0 1 2 3 0 1 0 1 2 0]
* (No padding added.)
dyn_bsz decides how many samples to pack so total tokens ~ target budget. If your target is, say, 10 tokens, it packs all 4 here.

# rmpad_with_pos_ids = True, dyn_bsz = OFF
Goal: still pack with position_ids, but batch size is fixed in number of samples, not by tokens.
Example fixed batch size = 2 samples:
* Batch 1: S1 + S2
* input_ids = [A B C D E F]
* position_ids = [0 1 2 3 0 1]
* Batch 2: S3 + S4
* input_ids = [G H I J]
* position_ids = [0 1 2 0]

# So the difference is:
**dyn_bsz ON: batch size varies to hit a token budget.**
**dyn_bsz OFF: batch size is fixed (#samples), but still uses position_ids packing (no padding).**