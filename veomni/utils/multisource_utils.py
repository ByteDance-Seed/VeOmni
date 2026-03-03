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

import yaml


try:
    from hdfs_io import open
except ImportError:
    from .hdfs_io import open


def parse_multisource_config(multisource_config_path: str):
    multisource_config = None
    with open(multisource_config_path) as f:
        multisource_config = yaml.safe_load(f)
    assert multisource_config is not None, "expect multisource_config is not None."
    return _parse_multisource_config(multisource_config)


def _parse_multisource_config(multisource_config: dict):
    if "names_weights" in multisource_config:
        # source_num == len(source_name) will be true in this case
        return multisource_config
    source_num = len(multisource_config["sources"])
    source_name = multisource_config["names"]
    assert len(source_name) == source_num == len(set(source_name)), (
        "names from multisource config is not equal to source_num, or there are sources have same name, "
        + "len(source_name) vs source_num vs len(set(source_name)): "
        + f"{len(source_name)} vs {source_num} vs {len(set(source_name))}."
    )
    schedule = multisource_config["schedule"]
    for value in schedule:
        assert value["schedule_type"] in ["const", "changing"], f"wrong schedule type: {value['schedule_type']}."
        if value["schedule_type"] == "changing":
            assert len(value["init_weights"]) == source_num and len(value["end_weights"]) == source_num, (
                "source_num is not equal to length of init_weights or end_weights, "
                + "source_num vs init_weights vs end_weights: "
            )
            f"{source_num} vs {value['init_weights']} vs {value['end_weights']}."
        else:
            assert len(value["weights"]) == source_num, (
                "source_num is not equal to length of weights, "
                + f"source_num vs weights: {source_num} vs {value['weights']}."
            )
    if "concat_sources" in multisource_config:
        assert len(multisource_config["concat_sources"]) == source_num, (
            "source_num is not equal to length of"
            + f" concat_sources, source_num vs concat_sources: {source_num} vs {multisource_config['concat_sources']}."
        )
    return multisource_config
