import json
import os
from dataclasses import asdict, dataclass, field

import requests
import torch
from PIL import Image

from veomni.data import build_multimodal_chat_template
from veomni.data.multimodal.multimodal_transform import mask_input_ids
from veomni.models import build_foundation_model, build_processor
from veomni.models.seed_omni import SeedOmniModel, SeedOmniProcessor
from veomni.utils import helper
from veomni.utils.arguments import InferArguments, parse_args
from veomni.utils.device import get_device_type


logger = helper.create_logger(__name__)


@dataclass
class MyInferArguments(InferArguments):
    chat_template: str = field(
        default="qwen2vl",
        metadata={"help": "Chat template."},
    )
    data: str = field(
        default="imagenet",
        metadata={"help": "Task type."},
    )


@dataclass
class Arguments:
    infer: "MyInferArguments" = field(default_factory=MyInferArguments)


@dataclass
class TestData:
    prompt: str = field(
        default="",
        metadata={"help": "Prompt."},
    )
    image_path: str = field(
        default=None,
        metadata={"help": "Image path. Single Image Currently."},
    )
    force_image_gen: bool = field(
        default=False,
        metadata={"help": "Force image generation."},
    )

    def get_conversations(self):
        if self.image_path:
            return [
                [
                    "user",
                    ("text", self.prompt),
                    ("image", None),
                ],
                ["assistant"],
            ]
        else:
            return [
                [
                    "user",
                    ("text", self.prompt),
                ],
                ["assistant"],
            ]

    def get_images(self):
        images = []
        if self.image_path:
            if os.path.exists(self.image_path):
                images = [Image.open(self.image_path).convert("RGB")]
            else:
                try:
                    image = Image.open(requests.get(self.image_path, stream=True).raw)
                    images = [image.convert("RGB")]
                except Exception as e:
                    raise e
        return images


GEN = dict(
    prompt=(
        "Generate an image based on the following caption:"
        "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower,"
        "under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue."
    ),
    force_image_gen=True,
)

UND = dict(
    prompt="Describe this image. ",
    image_path="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    force_image_gen=False,
)

EDIT = dict(
    prompt="Make the cat in the image black.",
    image_path="cat.jpg",
    force_image_gen=True,
)


data_mapping = {
    "gen": GEN,
    "und": UND,
    "edit": EDIT,
}


def main() -> None:
    args = parse_args(Arguments)
    logger.info(json.dumps(asdict(args), indent=2))
    helper.set_seed(args.infer.seed)
    helper.enable_third_party_logging()
    # config model and processor
    model: SeedOmniModel = (
        build_foundation_model(args.infer.model_path, args.infer.model_path).eval().to(get_device_type())
    )
    position_id_func = model.get_position_id_func()
    modality_info = model.get_modality()
    processor: SeedOmniProcessor = build_processor(args.infer.model_path)
    chat_template = build_multimodal_chat_template(args.infer.chat_template, processor.tokenizer)

    # config data
    data_config = data_mapping[args.infer.data]
    test_data = TestData(**data_config)
    conversations = test_data.get_conversations()
    images = test_data.get_images()
    force_image_gen = test_data.force_image_gen

    image_inputs = processor(input_images=images, return_tensors="pt")

    image_token_nums = image_inputs.pop("image_input_num_tokens", [])
    image_grid_thw = image_inputs.get("image_input_grid_thw", torch.empty(0, 3, dtype=torch.int32))
    tokenized_example = chat_template.encode_messages(conversations, {"image": image_token_nums})

    inputs = {
        "input_ids": tokenized_example["input_ids"].unsqueeze(0).to(model.device),
        "attention_mask": tokenized_example["attention_mask"].unsqueeze(0).to(model.device),
        **image_inputs.to(model.device),
    }
    position_id_returns = position_id_func(input_ids=inputs["input_ids"], image_grid_thw=image_grid_thw)
    inputs.update(position_id_returns)

    if force_image_gen:
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.tensor([chat_template.image_start_id], device=model.device).unsqueeze(0)],
            dim=-1,
        )

        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.tensor([1], device=model.device).unsqueeze(0)], dim=-1
        )

        pad_position_value = inputs["position_ids"].max() + 1
        pad_shape = list(inputs["position_ids"].shape)
        pad_shape[-1] = 1
        pad = torch.full(
            pad_shape, fill_value=pad_position_value, dtype=inputs["position_ids"].dtype, device=model.device
        )
        inputs["position_ids"] = torch.cat([inputs["position_ids"], pad], dim=-1)

    input_ids, mask_dict = mask_input_ids(modality_info, inputs["input_ids"])
    inputs["input_ids"] = input_ids
    inputs.update(mask_dict)

    gen_kwargs = {
        "do_sample": args.infer.do_sample,
        "temperature": args.infer.temperature,
        "top_p": args.infer.top_p,
        "max_new_tokens": args.infer.max_tokens,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.eos_token_id,
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }
    model.setup_generation_config(
        image_start_token=chat_template.image_start_id,
        image_token_size=processor.output_image_processor.token_size,
        force_image_gen=force_image_gen,
    )
    outputs = model.generate(**inputs, **gen_kwargs)
    x = outputs["sequences"]
    output_sequence = x[0, len(inputs["input_ids"][0]) :]
    target_num_token = processor.output_image_processor.token_num

    image_start_id = chat_template.image_start_id
    index = None

    if force_image_gen:
        index = [0]
    else:
        index = (output_sequence == image_start_id).nonzero()
        if index.shape[0] != 0:
            index = index[:, 0]

    if len(index) != 0:
        image_end_index = 0
        for i in range(len(index)):
            if i != 0 and index[i] - index[i - 1] < target_num_token:
                continue
            if index[i] > image_end_index:
                text = processor.tokenizer.decode(
                    output_sequence[image_end_index : index[i]], skip_special_tokens=False
                )
                print(text)

            hidden_states = []
            for hs in outputs["hidden_states"][index[i] : index[i] + target_num_token]:
                hidden_states.append(hs[-1][:, -1:])
            hidden_states = torch.cat(hidden_states, dim=1)
            if hidden_states.shape[1] != target_num_token:
                print(
                    f"Error: Image not fully generated. Target token_num: {target_num_token}. Generated token_num: {hidden_states.shape[1]}."
                )
                break
            output_image = model.generate_multimodal(hidden_states, modal_type="image")
            output_image = processor.output_image_processor.postprocess(output_image)[0]
            os.makedirs("output", exist_ok=True)
            output_image.save(f"output/generated_image_{i}.png")
            print(f"[IMAGE] saved in output/generated_image_{i}.png")
            image_end_index = index[i] + target_num_token
    else:
        response = processor.tokenizer.decode(output_sequence, skip_special_tokens=False)
        print(response)


if __name__ == "__main__":
    main()
