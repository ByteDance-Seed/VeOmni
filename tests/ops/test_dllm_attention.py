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
"""Direct tests of ``dllm_attention``, independent of the model wrapper."""

import pytest
import torch

from veomni.ops.kernels.dllm_attention import dllm_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dllm_attention needs a GPU")
class TestDllmAttention:
    def test_packed_segments_match_dense_attention_forward_and_backward(self):
        torch.manual_seed(123)
        device = "cuda"
        dtype = torch.bfloat16
        batch, heads, dim = 1, 4, 64
        q_lens = [7, 4]
        prefix_lens = [5, 13]
        k_lens = [p + q for p, q in zip(prefix_lens, q_lens)]
        total_q, total_prefix = sum(q_lens), sum(prefix_lens)
        scale = 1.0

        q_ref = (0.1 * torch.randn(batch, heads, total_q, dim, device=device, dtype=dtype)).requires_grad_(True)
        prefix_k_ref = (0.1 * torch.randn(batch, heads, total_prefix, dim, device=device, dtype=dtype)).requires_grad_(
            True
        )
        prefix_v_ref = torch.randn_like(prefix_k_ref, requires_grad=True)
        canvas_k_ref = (0.1 * torch.randn(batch, heads, total_q, dim, device=device, dtype=dtype)).requires_grad_(True)
        canvas_v_ref = torch.randn_like(canvas_k_ref, requires_grad=True)
        q_tri = q_ref.detach().clone().requires_grad_(True)
        prefix_k_tri = prefix_k_ref.detach().clone().requires_grad_(True)
        prefix_v_tri = prefix_v_ref.detach().clone().requires_grad_(True)
        canvas_k_tri = canvas_k_ref.detach().clone().requires_grad_(True)
        canvas_v_tri = canvas_v_ref.detach().clone().requires_grad_(True)

        cu_q = torch.tensor([0, q_lens[0], total_q], device=device, dtype=torch.int32)
        cu_prefix = torch.tensor([0, prefix_lens[0], total_prefix], device=device, dtype=torch.int32)
        dense_parts = []
        for segment in range(len(q_lens)):
            qs, qe = cu_q[segment].item(), cu_q[segment + 1].item()
            ps, pe = cu_prefix[segment].item(), cu_prefix[segment + 1].item()
            segment_k = torch.cat([prefix_k_ref[:, :, ps:pe], canvas_k_ref[:, :, qs:qe]], dim=2)
            segment_v = torch.cat([prefix_v_ref[:, :, ps:pe], canvas_v_ref[:, :, qs:qe]], dim=2)
            scores = torch.matmul(q_ref[:, :, qs:qe].float(), segment_k.float().transpose(-2, -1))
            probs = torch.softmax(scores * scale, dim=-1).to(dtype)
            dense_parts.append(torch.matmul(probs, segment_v))
        out_ref = torch.cat(dense_parts, dim=2)

        out_tri = (
            dllm_attention(
                q_tri,
                prefix_k_tri,
                prefix_v_tri,
                canvas_k_tri,
                canvas_v_tri,
                cu_q,
                cu_prefix,
                max_seqlen_q=max(q_lens),
                max_seqlen_k=max(k_lens),
                sm_scale=scale,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        torch.testing.assert_close(out_tri, out_ref, rtol=2e-2, atol=2e-2)

        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out_tri.backward(grad)
        torch.testing.assert_close(q_tri.grad, q_ref.grad, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(prefix_k_tri.grad, prefix_k_ref.grad, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(prefix_v_tri.grad, prefix_v_ref.grad, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(canvas_k_tri.grad, canvas_k_ref.grad, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(canvas_v_tri.grad, canvas_v_ref.grad, rtol=3e-2, atol=3e-2)

    def test_empty_prefix_requires_explicit_scale_and_rejects_a_leaking_row(self):
        q = torch.randn(1, 2, 8, 64, device="cuda", dtype=torch.bfloat16)
        prefix_k = torch.empty(1, 2, 0, 64, device="cuda", dtype=torch.bfloat16)
        prefix_v = torch.empty_like(prefix_k)
        canvas_k = torch.randn_like(q)
        canvas_v = torch.randn_like(q)
        cu_q = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
        cu_prefix = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
        with pytest.raises(ValueError, match="explicit sm_scale"):
            dllm_attention(q, prefix_k, prefix_v, canvas_k, canvas_v, cu_q, cu_prefix)
        out = dllm_attention(q, prefix_k, prefix_v, canvas_k, canvas_v, cu_q, cu_prefix, sm_scale=1.0)
        assert out.shape == (1, 8, 2, 64)
        assert torch.isfinite(out).all()

        bad_prefix = torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="does not cover"):
            dllm_attention(
                q,
                bad_prefix,
                bad_prefix,
                canvas_k,
                canvas_v,
                cu_q,
                cu_prefix,
                sm_scale=1.0,
            )

    def test_packed_segments_do_not_cross_leak(self):
        torch.manual_seed(9)
        q = torch.randn(1, 2, 6, 64, device="cuda", dtype=torch.bfloat16)
        pk = torch.randn(1, 2, 7, 64, device="cuda", dtype=torch.bfloat16)
        pv = torch.randn_like(pk)
        ck = torch.randn_like(q)
        cv = torch.randn_like(q)
        cu_q = torch.tensor([0, 3, 6], device="cuda", dtype=torch.int32)
        cu_p = torch.tensor([0, 2, 7], device="cuda", dtype=torch.int32)
        out = dllm_attention(q, pk, pv, ck, cv, cu_q, cu_p, sm_scale=1.0)
        pk2, pv2, ck2, cv2 = pk.clone(), pv.clone(), ck.clone(), cv.clone()
        pk2[:, :, 2:] += 100
        pv2[:, :, 2:] -= 100
        ck2[:, :, 3:] += 100
        cv2[:, :, 3:] -= 100
        out2 = dllm_attention(q, pk2, pv2, ck2, cv2, cu_q, cu_p, sm_scale=1.0)
        torch.testing.assert_close(out[:, :3], out2[:, :3], rtol=0, atol=0)

    def test_head_dim_256_and_512_forward_backward(self):
        for dim in (256, 512):
            q = (0.05 * torch.randn(1, 2, 4, dim, device="cuda", dtype=torch.bfloat16)).requires_grad_(True)
            pk = (0.05 * torch.randn(1, 2, 3, dim, device="cuda", dtype=torch.bfloat16)).requires_grad_(True)
            pv = torch.randn_like(pk, requires_grad=True)
            ck = (0.05 * torch.randn(1, 2, 4, dim, device="cuda", dtype=torch.bfloat16)).requires_grad_(True)
            cv = torch.randn_like(ck, requires_grad=True)
            q_ref = q.detach().clone().requires_grad_(True)
            pk_ref = pk.detach().clone().requires_grad_(True)
            pv_ref = pv.detach().clone().requires_grad_(True)
            ck_ref = ck.detach().clone().requires_grad_(True)
            cv_ref = cv.detach().clone().requires_grad_(True)
            cu_q = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
            cu_p = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
            out = dllm_attention(q, pk, pv, ck, cv, cu_q, cu_p, sm_scale=1.0)
            ref_k = torch.cat([pk_ref, ck_ref], dim=2)
            ref_v = torch.cat([pv_ref, cv_ref], dim=2)
            ref_probs = torch.softmax(torch.matmul(q_ref.float(), ref_k.float().transpose(-2, -1)), dim=-1).to(
                torch.bfloat16
            )
            ref = torch.matmul(ref_probs, ref_v)
            torch.testing.assert_close(out.permute(0, 2, 1, 3), ref, rtol=3e-2, atol=3e-2)
            grad = torch.randn_like(ref)
            out.backward(grad.permute(0, 2, 1, 3).contiguous())
            ref.backward(grad)
            for actual, expected in (
                (q.grad, q_ref.grad),
                (pk.grad, pk_ref.grad),
                (pv.grad, pv_ref.grad),
                (ck.grad, ck_ref.grad),
                (cv.grad, cv_ref.grad),
            ):
                torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
            for tensor in (out, q.grad, pk.grad, pv.grad, ck.grad, cv.grad):
                assert torch.isfinite(tensor).all(), f"non-finite dllm_attention result for {dim=}"
