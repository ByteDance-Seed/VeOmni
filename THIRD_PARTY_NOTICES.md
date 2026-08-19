# Third-party notices

## MindSpeed context-parallel attention

The Ring context-parallel implementation in
`veomni/distributed/context_parallel/{causal_schedule,ring_attention,ring_p2p}.py`
is adapted from the following Huawei MindSpeed sources:

- `mindspeed/core/context_parallel/ring_context_parallel/ring_context_parallel.py`
- `mindspeed/core/context_parallel/utils.py`

The Open-VeOmni port was introduced in commit
`da6483e2152281dc81fe5abb8fb7289ee326bf76`; the contemporaneous MindSpeed
`master` snapshot used for provenance is
[`1787f81effa43af27ff01b409bbfa9d182126c65`](https://github.com/Ascend/MindSpeed/tree/1787f81effa43af27ff01b409bbfa9d182126c65).
The adapted files retain the upstream attribution and are redistributed under
the BSD 3-Clause License:

> Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
> Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
> 3. Neither the name of the copyright holder nor the names of its contributors
>    may be used to endorse or promote products derived from this software
>    without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.
