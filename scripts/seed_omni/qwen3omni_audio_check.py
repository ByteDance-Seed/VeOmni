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
"""Check whether Qwen3-Omni's talker emits Mimi RVQ codes -- in both directions.

Qwen2.5-Omni issue #219 asks how to obtain the proprietary "qwen-tts-tokenizer"
needed to build (text, audio-code) pairs for talker training. For Qwen3-Omni the
answer turns out to be that no proprietary tokenizer is needed: the talker's RVQ
codes live in Mimi's index space, and Mimi's encoder is public. This script
verifies that claim on our own checkpoint rather than taking it on faith, and it
checks the direction that actually matters for training.

Three phases:

1. **Decode direction.** Tap ``talker_codes`` on its way into Code2Wav and
   decode the same tensor two ways -- Qwen's Code2Wav and ``kyutai/mimi``.
   Matching audio means the two decoders read the indices the same way.

2. **Encode direction** (what talker training needs). Re-encode Qwen's decoded
   audio with Mimi and feed those codes back through Qwen's vocoder. If Mimi's
   encoder can drive Qwen's decoder, targets can be built from raw audio.

3. **The control that keeps phase 2 honest.** Code-level agreement is poor
   (~10% overall), which looks alarming until you run the same round-trip
   through Mimi *alone*: encode -> decode -> encode does not recover its own
   codes either. That makes the loss intrinsic RVQ residual sensitivity, not a
   codec mismatch -- and it means neither this codec nor a talker trained on it
   can be validated by exact code equality beyond the semantic group.

One run on Qwen3-Omni-30B-A3B-Instruct, for orientation. The talker samples a
different utterance each time, so absolute values move between runs; what is
stable is the ordering -- real pairings well below the controls::

    log-STFT L1 vs the talker's own audio (lower is better)
      mimi.encode -> mimi.decode              0.0653
      mimi.encode -> qwen.code2wav            0.1235   <- encode direction works
      talker codes -> mimi.decode             0.1256   <- decode direction (ref)
      shuffled codes -> mimi.decode           0.2344   <- control

    exact code match                        q00      q03+
      qwen.decode -> mimi.encode            0.583    0.022
      mimi.decode -> mimi.encode  (control) 0.854    0.127   <- barely better
      random codes                (control) 0.000    0.000

Usage::

    python scripts/seed_omni/qwen3omni_audio_check.py --out-dir /tmp/omni_check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


QWEN_DEFAULT = "/opt/tiger/models/Qwen3-Omni-30B-A3B-Instruct"
# A hub id rather than a cluster path, matching `convert_model.py`: the encoder
# half is public (https://huggingface.co/kyutai/mimi) and `MimiModel.from_pretrained`
# resolves the id, so anyone outside internal storage can run this check.
MIMI_DEFAULT = "kyutai/mimi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=QWEN_DEFAULT)
    parser.add_argument("--mimi", default=MIMI_DEFAULT)
    parser.add_argument("--out-dir", default="/tmp/omni_check")
    # Single device by default: the talker's sampling loop mixes `pad_token_id`
    # with `next_tokens` without a device cast, so an "auto" split across GPUs
    # dies with a cuda:0/cuda:N mismatch. The 30B-A3B weights fit on one H20.
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--prompt", default="Hello! Please say this sentence out loud so I can hear your voice.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--mimi-only",
        action="store_true",
        help=(
            "Check the encoder/decoder *plumbing* without the Qwen checkpoint: load Mimi, verify its config "
            "against what the codec module expects, and round-trip a tone. This is the part of the verification "
            "that does not need Qwen weights, so `--model` is not read. It does NOT measure the index-space "
            "agreement between the two codecs -- that needs both checkpoints."
        ),
    )
    return parser.parse_args()


def run_mimi_only(args: argparse.Namespace) -> None:
    """Encoder-side checks that need no Qwen checkpoint.

    Answers the question the converter actually depends on -- is the Mimi
    checkpoint we fetch the one this module's config was written against -- and
    nothing more. The two-codec comparison still needs the 30B weights.
    """
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import MimiConfig, MimiModel

    from veomni.models.seed_omni.modules.qwen3omni.codec.configuration import Qwen3OmniCodecConfig

    print(f"[1/3] loading {args.mimi}")
    mimi = MimiModel.from_pretrained(args.mimi, dtype=torch.float32).eval()
    print(f"      {sum(p.numel() for p in mimi.parameters()) / 1e6:.1f}M params")

    print("[2/4] does this checkpoint match what the codec module expects?")
    # The same object the split builds, so the checks below are the ones a real
    # codec runs at construction rather than a reimplementation of them.
    config = Qwen3OmniCodecConfig(mimi_config=MimiConfig.from_pretrained(args.mimi).to_dict())
    print(f"      encoder_from_checkpoint: {config.encoder_from_checkpoint}")
    print(f"      sampling_rate: {config.sampling_rate} Hz   quantizers: {config.num_quantizers}")
    print(
        f"      samples/frame -- decoder {config.decoder_samples_per_frame}, "
        f"encoder {config.encoder_samples_per_frame}"
    )

    print("[3/4] encode -> decode round trip on a 1 s tone")
    rate = config.sampling_rate
    tone = 0.3 * torch.sin(2 * torch.pi * 220 * torch.arange(rate, dtype=torch.float32) / rate)
    with torch.no_grad():
        codes = mimi.encode(tone.view(1, 1, -1), num_quantizers=config.num_quantizers).audio_codes
        decoded = mimi.decode(codes).audio_values
    sf.write(out / "mimi_round_trip.wav", decoded.squeeze().float().cpu().numpy(), rate)
    print(f"      codes {tuple(codes.shape)}  frames/s={codes.shape[-1] / (len(tone) / rate):.1f}")
    print(f"      decoded {decoded.shape[-1]} samples = {decoded.shape[-1] / rate:.2f}s -> {out}")

    # ...and through *this module*, which is the contract the talker is written
    # against: `(num_quantizers, T)`. Built from the real Mimi config with the
    # encoder weights loaded, so this exercises `encode_wav` rather than
    # reimplementing it — without the Qwen decoder half, which `--mimi-only`
    # deliberately does not have.
    print("[4/4] this module's encode_wav on the same tone")
    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY
    from veomni.models.seed_omni.modules.qwen3omni.codec.modeling import Qwen3OmniCodec

    codec_cfg = OMNI_CONFIG_REGISTRY["qwen3omni_codec"]()(
        code2wav_config=None,
        mimi_config=MimiConfig.from_pretrained(args.mimi).to_dict(),
    )
    codec = OMNI_MODEL_REGISTRY["qwen3omni_codec"]()(codec_cfg)
    # The decoder half is absent under `--mimi-only`; nothing in `encode_wav`
    # touches it, but the module builds it, so keep this honest by reporting.
    if not isinstance(codec, Qwen3OmniCodec):
        raise AssertionError(f"unexpected codec class {type(codec).__name__}")
    codec.mimi.load_state_dict(mimi.state_dict(), strict=False)
    codec.eval()
    with torch.no_grad():
        module_codes = codec.encode_wav(tone, sampling_rate=rate)
    print(f"      encode_wav -> {tuple(module_codes.shape)}  (expected ({config.num_quantizers}, T))")
    assert module_codes.shape[0] == config.num_quantizers, module_codes.shape


def spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Log-magnitude STFT L1 distance, robust to a global phase/sample shift."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    win = np.hanning(1024)
    frames = range(0, n - 1024, 256)
    sa = np.stack([np.abs(np.fft.rfft(a[i : i + 1024] * win)) for i in frames])
    sb = np.stack([np.abs(np.fft.rfft(b[i : i + 1024] * win)) for i in frames])
    return float(np.abs(np.log1p(sa) - np.log1p(sb)).mean())


def main() -> None:
    args = parse_args()
    if args.mimi_only:
        run_mimi_only(args)
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

    print(f"[1/6] loading {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": args.device}
    )
    model.eval()

    # Tap talker_codes on the way into Code2Wav. This is the tensor the issue
    # calls the "RVQ code": (batch, num_quantizers, frames).
    captured: dict[str, torch.Tensor] = {}
    original_decode = model.code2wav.chunked_decode

    def tapped_decode(codes, *fn_args, **fn_kwargs):
        captured["codes"] = codes.detach().clone()
        return original_decode(codes, *fn_args, **fn_kwargs)

    model.code2wav.chunked_decode = tapped_decode

    print("[2/6] generating with audio output")
    conversation = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        text_ids, wav = model.generate(
            **inputs,
            speaker=args.speaker,
            return_audio=True,
            thinker_max_new_tokens=args.max_new_tokens,
            talker_do_sample=False,
        )

    text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
    print(f"      text: {text.strip()[:200]}")

    # Deliberately not read off code2wav_config: it declares no sampling rate.
    # The rate comes from the encoder below, which does.
    qwen_wav = wav.squeeze().float().cpu().numpy()

    codes = captured["codes"]
    np.save(out / "talker_codes.npy", codes.cpu().numpy())
    print(f"[3/6] talker codes {tuple(codes.shape)} dtype={codes.dtype} range=[{codes.min()}, {codes.max()}]")

    print(f"[4/6] decoding the same codes through {args.mimi}")
    from transformers import AutoFeatureExtractor, MimiModel

    mimi = MimiModel.from_pretrained(args.mimi, dtype=torch.float32).eval()
    feat = AutoFeatureExtractor.from_pretrained(args.mimi)
    sr = feat.sampling_rate
    sf.write(out / "qwen_output.wav", qwen_wav, sr)
    print(f"      wrote {out / 'qwen_output.wav'}  ({len(qwen_wav) / sr:.2f}s @ {sr}Hz)")
    with torch.no_grad():
        mimi_wav = mimi.decode(codes.to(torch.long).to(mimi.device)).audio_values
    mimi_wav = mimi_wav.squeeze().float().cpu().numpy()
    sf.write(out / "mimi_recon.wav", mimi_wav, sr)
    print(f"      wrote {out / 'mimi_recon.wav'}  ({len(mimi_wav) / sr:.2f}s @ {sr}Hz)")

    print("[5/6] decode-direction comparison")
    print(f"      qwen  : {len(qwen_wav)} samples, rms={np.sqrt((qwen_wav**2).mean()):.4f}")
    print(f"      mimi  : {len(mimi_wav)} samples, rms={np.sqrt((mimi_wav**2).mean()):.4f}")
    print(f"      log-STFT L1 distance: {spectral_distance(qwen_wav, mimi_wav):.4f}")
    print("      Listen to both wavs -- same words in both means the code spaces agree.")

    check_encode_direction(mimi, model.code2wav, codes, qwen_wav, sr, out)


def code_match_rates(a: torch.Tensor, b: torch.Tensor) -> tuple[float, list[float]]:
    """Per-quantizer exact-match rates between two code tensors, over common frames."""
    frames = min(a.shape[-1], b.shape[-1])
    a, b = a[0, :, :frames].cpu(), b[0, :, :frames].cpu()
    per_group = [(a[q] == b[q]).float().mean().item() for q in range(a.shape[0])]
    return (a == b).float().mean().item(), per_group


def check_encode_direction(mimi, code2wav, talker_codes, qwen_wav, sr, out) -> None:
    """Phase 2 + 3: can Mimi's encoder drive Qwen's decoder, and is the drift intrinsic?

    The audio comparison is the load-bearing one. The code comparison is
    reported *with* its control, because on its own a ~10% match rate reads as a
    broken codec when it is in fact how RVQ behaves.
    """
    num_quantizers = talker_codes.shape[1]
    device = next(mimi.parameters()).device
    waveform = torch.tensor(qwen_wav, dtype=torch.float32, device=device).view(1, 1, -1)

    print("[6/6] encode direction + RVQ control")
    with torch.no_grad():
        reencoded = mimi.encode(waveform, num_quantizers=num_quantizers).audio_codes.long()
        qwen_on_mimi = code2wav.chunked_decode(reencoded.to(code2wav.device), chunk_size=300, left_context_size=25)
        mimi_round_trip = mimi.decode(reencoded).audio_values
        # Control: Mimi's own encode -> decode -> encode. If this also fails to
        # recover its codes, the drift is the codec's nature, not a mismatch.
        control = mimi.encode(mimi_round_trip, num_quantizers=num_quantizers).audio_codes.long()

    # Audio control: the same codes with their time axis shuffled. Establishes
    # what "these two decoders disagree" actually looks like as a number.
    generator = torch.Generator().manual_seed(0)
    shuffled = talker_codes[:, :, torch.randperm(talker_codes.shape[-1], generator=generator)]
    with torch.no_grad():
        shuffled_wav = mimi.decode(shuffled.long().to(device)).audio_values

    qwen_on_mimi_wav = qwen_on_mimi.squeeze().float().cpu().numpy()
    mimi_rt_wav = mimi_round_trip.squeeze().float().cpu().numpy()
    sf.write(out / "qwen_on_mimi_codes.wav", qwen_on_mimi_wav, sr)
    sf.write(out / "mimi_round_trip.wav", mimi_rt_wav, sr)

    print("      audio (log-STFT L1 vs the talker's own audio, lower is better):")
    print(f"        mimi.encode -> mimi.decode      : {spectral_distance(mimi_rt_wav, qwen_wav):.4f}")
    print(f"        mimi.encode -> qwen.code2wav    : {spectral_distance(qwen_on_mimi_wav, qwen_wav):.4f}")
    print(
        f"        shuffled codes -> mimi.decode   : "
        f"{spectral_distance(shuffled_wav.squeeze().float().cpu().numpy(), qwen_wav):.4f}   (CTRL)"
    )

    overall, per_group = code_match_rates(talker_codes.long(), reencoded)
    ctrl_overall, ctrl_per_group = code_match_rates(reencoded, control)
    random_codes = torch.randint(0, mimi.config.codebook_size, reencoded.shape, generator=generator)
    rand_overall, rand_per_group = code_match_rates(talker_codes.long(), random_codes)

    def summarize(label, overall_rate, groups):
        deep = sum(groups[3:]) / max(len(groups[3:]), 1)
        print(f"        {label}: overall {overall_rate:.4f}  q00 {groups[0]:.3f}  q03+ {deep:.3f}")

    print("      exact code match:")
    summarize("qwen.decode -> mimi.encode ", overall, per_group)
    summarize("mimi self round-trip (CTRL)", ctrl_overall, ctrl_per_group)
    summarize("random codes         (CTRL)", rand_overall, rand_per_group)
    print(
        "      Read this as: the audio numbers say the index spaces agree, and the self round-trip\n"
        "      control says the low code-match rate is RVQ residual drift, not a codec mismatch.\n"
        "      Do not validate a trained talker on exact code match beyond q00."
    )


if __name__ == "__main__":
    main()
