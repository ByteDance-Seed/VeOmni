# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict


class DynBszBuffer:
    """
    A buffer to store samples for dynamic batch size.
    """

    def __init__(self):
        self._buffer = []
        self._buffer_sample_lens = []
        self.del_idxs = []
        self.cur_idx = 0
        self.all_token_cnt = 0

    def append(self, item: Dict[str, Any]):
        """
        Append a sample to the buffer.
        Args:
            item: a sample to append to the buffer.
                The sample should be a dict with the following keys:
                    - input_ids: torch.Tensor of shape (seq_len, )
                    - attention_mask: torch.Tensor of shape (seq_len, )
        """
        self._buffer.append(item)
        self._buffer_sample_lens.append(item["attention_mask"].sum())
        self.all_token_cnt += self._buffer_sample_lens[-1]

    def get_samples(self, n_token_per_iter: int, force: bool = True):
        """
        get samples from the buffer.
        Args:
            n_token_per_iter: the number of tokens to get.
            force: if True, the first sample will be returned even if it is not full.
        Returns:
            samples: a list of samples.
        """
        cum_seq_len = 0
        samples = []
        while self.cur_idx < len(self._buffer) and cum_seq_len < n_token_per_iter:
            seq_len = self._buffer_sample_lens[self.cur_idx]
            if self.cur_idx not in self.del_idxs and (
                (force is True and cum_seq_len == 0) or (seq_len <= n_token_per_iter - cum_seq_len)
            ):
                cum_seq_len += seq_len
                samples.append(self._buffer[self.cur_idx])
                self.del_idxs.append(self.cur_idx)
            self.cur_idx += 1
        assert len(samples) > 0
        return samples

    def __len__(self):
        return len(self._buffer)

    def flush(self):
        """ "
        Flush the buffer.
        """
        self.cur_idx = 0
        self.all_token_cnt -= sum([self._buffer_sample_lens[idx] for idx in self.del_idxs])
        buffer_len = len(self._buffer)
        self._buffer = [self._buffer[idx] for idx in range(buffer_len) if idx not in self.del_idxs]
        self._buffer_sample_lens = [
            self._buffer_sample_lens[idx] for idx in range(buffer_len) if idx not in self.del_idxs
        ]
        self.del_idxs = []

    def merge(self, buffer_to_merge: "DynBszBuffer"):
        """ "
        Merge the buffer with another buffer.
        Args:
            buffer_to_merge: the buffer to merge.
        """
        self.flush()
        buffer_to_merge.flush()
        for item in buffer_to_merge._buffer:
            self.append(item)


class BaseBatchingStrategy:
    """
    Base class for batching strategy.s
    """

    def is_full_filled(self) -> bool:
        raise NotImplementedError("should implement `is_full_filled`")

    def put_item(self, item: Dict[str, Any]):
        raise NotImplementedError("should implement `put_item`")

    def get_micro_batch(self, step: int) -> Any:
        raise NotImplementedError("should implement `get_micro_batch` ")

    def empty(self) -> bool:
        raise NotImplementedError("should implement `empty`")


class IdentityPacker:
    def __init__(self, token_micro_bsz, bsz_warmup_steps, bsz_warmup_init_mbtoken):
        self.token_micro_bsz = token_micro_bsz
        self.bsz_warmup_steps = bsz_warmup_steps
        self.bsz_warmup_init_mbtoken = bsz_warmup_init_mbtoken

    def __call__(self, samples):
        return samples

    def get_token_num_to_request(self, cur_step, warmup):
        return (
            (self.token_micro_bsz - self.bsz_warmup_init_mbtoken) * cur_step // self.bsz_warmup_steps
            + self.bsz_warmup_init_mbtoken
            if warmup
            else self.token_micro_bsz
        )


class TextBatchingStrategy(BaseBatchingStrategy):
    """ "
    Batching strategy for text data.
    Args:
        token_micro_bsz: the number of tokens to get for each request.
        buffer_size: the size of the buffer.
        bsz_warmup_steps: the number of steps to warm up the batch size.
        bsz_warmup_init_mbtoken: the initial number of tokens to get for each request.
    """

    def __init__(
        self,
        token_micro_bsz,
        buffer_size: int = 500,
        bsz_warmup_steps: int = -1,
        bsz_warmup_init_mbtoken: int = 200,
    ) -> None:
        super().__init__()
        self._step = 0
        self.token_micro_bsz = token_micro_bsz
        self.bsz_warmup_steps = bsz_warmup_steps
        self.buffer_size = buffer_size  # minimum samples in buffer
        self.buffer = DynBszBuffer()
        self.bsz_warmup_init_mbtoken = bsz_warmup_init_mbtoken
        assert self.bsz_warmup_init_mbtoken >= 0

        self.packer = IdentityPacker(
            token_micro_bsz=token_micro_bsz,
            bsz_warmup_steps=bsz_warmup_steps,
            bsz_warmup_init_mbtoken=bsz_warmup_init_mbtoken,
        )

    def is_full_filled(self) -> bool:
        return len(self.buffer) >= self.buffer_size and self.buffer.all_token_cnt >= self.token_micro_bsz

    def put_item(self, item: Dict[str, Any]):
        if len(item["input_ids"]) == 1:
            print("WARNING: EMPTY STRING.")
            return
        self.buffer.append(item)

    def get_token_num_to_request(self):
        if self.packer is not None:
            warmup = self._step <= self.bsz_warmup_steps and self.bsz_warmup_steps > 0
            return self.packer.get_token_num_to_request(self._step, warmup=warmup)
        else:
            return self.get_cur_token_micro_bsz()

    def get_cur_token_micro_bsz(self):
        warmup = self._step <= self.bsz_warmup_steps and self.bsz_warmup_steps > 0
        if warmup:
            return (
                self.token_micro_bsz - self.bsz_warmup_init_mbtoken
            ) * self._step // self.bsz_warmup_steps + self.bsz_warmup_init_mbtoken
        else:
            return self.token_micro_bsz

    def get_micro_batch(self, step) -> Any:
        """
        Get a micro batch from the buffer according to the current step.
        Args:
            step: the current step.
        Returns:
            data: a list of samples.
        """

        self._step = step
        n_token_per_iter = self.get_token_num_to_request()
        cur_token_micro_bsz = self.get_cur_token_micro_bsz()
        assert (
            cur_token_micro_bsz % n_token_per_iter == 0
        ), "The token num to get for each request should be divisible by token micro bsz."
        n_iter = int(cur_token_micro_bsz // n_token_per_iter)
        data = []
        for i in range(n_iter):
            samples = self.buffer.get_samples(n_token_per_iter)
            if self.packer:
                samples = self.packer(samples)  # maybe packed into one sample, but wrapped in list.
            data.extend(samples)
        self.buffer.flush()  # remove the selected samples.
        return data

    def empty(self) -> bool:
        return len(self.buffer) == 0
