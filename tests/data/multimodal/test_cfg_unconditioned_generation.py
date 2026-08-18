from veomni.data.multimodal.multimodal_chat_template import Qwen2VLPretrainTemplate


class _T2ITokenizer:
    pad_token_id = 0

    def __init__(self):
        self._ids = {
            "<|im_start|>": [1],
            "<|im_end|>\n": [2],
            "a cat": [5, 6],
            "<|vision_start|><|image_pad|><|vision_end|>": [7],
        }

    def convert_tokens_to_ids(self, token):
        return {
            "<|image_pad|>": 100,
            "<|video_pad|>": 101,
            "<|vision_start|>": 102,
            "<|vision_end|>": 103,
        }.get(token, -1)

    def encode(self, text, add_special_tokens=False):
        return self._ids[text]


def test_unconditioned_generation_pads_user_turn_when_user_not_last():
    tokenizer = _T2ITokenizer()
    template = Qwen2VLPretrainTemplate(tokenizer, cfg_ratio=1.0)

    # t2i conversation: a user text prompt followed by the assistant image
    # turn. The unconditioned-generation path must pad the *user* turn even
    # though it is not the last message (previously the stale `role` bound to
    # the assistant turn skipped the padding).
    conversations = [
        ["user", ["text", "a cat"]],
        ["assistant", ["image"]],
    ]

    encoded = template.encode_messages(conversations, num_tokens={"image": [1]})

    # bos(1) + pad(0) * 2 (user text replaced) + image(7) + eos(2)
    assert encoded["input_ids"].tolist() == [1, 0, 0, 7, 2]
