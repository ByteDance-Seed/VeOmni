from veomni.distributed.torch_parallelize import (
    _build_fsdp2_backward_prefetch_chains,
    _configure_fsdp2_manual_prefetch,
    _get_fsdp2_prefetch_role,
    _register_fsdp2_prefetch_module,
)


class _FakeFSDPModule:
    def __init__(self, name):
        self.name = name
        self._fsdp_modules = []
        self.forward_prefetch = None
        self.backward_prefetch = None

    def set_modules_to_forward_prefetch(self, modules):
        self.forward_prefetch = modules

    def set_modules_to_backward_prefetch(self, modules):
        self.backward_prefetch = modules


def _make_nested_blocks(num_blocks=3):
    blocks = []
    experts = []
    for index in range(num_blocks):
        block = _FakeFSDPModule(f"decoder.{index}")
        expert = _FakeFSDPModule(f"decoder.{index}.experts")
        block._fsdp_modules = [expert, block]
        blocks.append(block)
        experts.append(expert)
    return blocks, experts


def test_backward_prefetch_configures_every_nested_fsdp_chain():
    blocks, experts = _make_nested_blocks()

    _configure_fsdp2_manual_prefetch(
        blocks,
        {("extra_parallel", "ep", 0): experts, ("target",): blocks},
        enable_forward_prefetch=False,
    )

    assert experts[0].backward_prefetch == [experts[0]]
    assert experts[1].backward_prefetch == [experts[0]]
    assert experts[2].backward_prefetch == [experts[1]]
    assert blocks[0].backward_prefetch == [blocks[0]]
    assert blocks[1].backward_prefetch == [blocks[0]]
    assert blocks[2].backward_prefetch == [blocks[1]]
    assert all(block.forward_prefetch is None for block in blocks)


def test_forward_prefetch_keeps_existing_next_block_order():
    blocks, experts = _make_nested_blocks()

    _configure_fsdp2_manual_prefetch(
        blocks,
        {("extra_parallel", "ep", 0): experts, ("target",): blocks},
        enable_forward_prefetch=True,
    )

    assert blocks[0].forward_prefetch == [blocks[1], experts[1]]
    assert blocks[1].forward_prefetch == [blocks[2], experts[2]]
    assert blocks[2].forward_prefetch is None


def test_singleton_nested_chain_uses_self_target_to_disable_default_prefetch():
    blocks, experts = _make_nested_blocks(num_blocks=1)

    _configure_fsdp2_manual_prefetch(
        blocks,
        {("extra_parallel", "ep", 0): experts, ("target",): blocks},
        enable_forward_prefetch=False,
    )

    assert experts[0].backward_prefetch == [experts[0]]
    assert blocks[0].backward_prefetch == [blocks[0]]


def test_heterogeneous_blocks_keep_independent_role_chains():
    blocks, _ = _make_nested_blocks()
    expert = _FakeFSDPModule("decoder.1.experts")
    blocks[0]._fsdp_modules = [blocks[0]]
    blocks[1]._fsdp_modules = [expert, blocks[1]]
    blocks[2]._fsdp_modules = [blocks[2]]

    _configure_fsdp2_manual_prefetch(
        blocks,
        {("extra_parallel", "ep", 0): [expert], ("target",): blocks},
        enable_forward_prefetch=False,
    )

    assert expert.backward_prefetch == [expert]
    assert blocks[0].backward_prefetch == [blocks[0]]
    assert blocks[1].backward_prefetch == [blocks[0]]
    assert blocks[2].backward_prefetch == [blocks[1]]


def test_prefetch_roles_use_stable_stack_and_relative_module_fqns():
    language_role_0 = _get_fsdp2_prefetch_role(
        "extra_parallel",
        "model.language_model.layers.0",
        "model.language_model.layers.0.mlp.experts_b",
        "ep",
    )
    language_role_1 = _get_fsdp2_prefetch_role(
        "extra_parallel",
        "model.language_model.layers.1",
        "model.language_model.layers.1.mlp.experts_b",
        "ep",
    )
    vision_role = _get_fsdp2_prefetch_role(
        "extra_parallel",
        "model.visual.blocks.0",
        "model.visual.blocks.0.mlp.experts_b",
        "ep",
    )

    assert language_role_0 == language_role_1
    assert language_role_0 != vision_role


def test_prefetch_roles_keep_outer_numeric_stack_namespaces():
    tower_0_block_0 = _get_fsdp2_prefetch_role(
        "target",
        "model.vision_towers.0.blocks.0",
        "model.vision_towers.0.blocks.0",
    )
    tower_0_block_1 = _get_fsdp2_prefetch_role(
        "target",
        "model.vision_towers.0.blocks.1",
        "model.vision_towers.0.blocks.1",
    )
    tower_1_block_0 = _get_fsdp2_prefetch_role(
        "target",
        "model.vision_towers.1.blocks.0",
        "model.vision_towers.1.blocks.0",
    )

    assert tower_0_block_0 == tower_0_block_1
    assert tower_0_block_0 != tower_1_block_0


def test_chain_builder_uses_original_module_order_and_registers_existing_wrappers():
    block_0 = _FakeFSDPModule("model.layers.0")
    block_1 = _FakeFSDPModule("model.layers.1")
    block_10 = _FakeFSDPModule("model.layers.10")
    roles_by_module_id = {}
    target_modules = [
        (block_0.name, block_0),
        (block_1.name, block_1),
        (block_10.name, block_10),
    ]

    # Register in a deliberately different wrapping order. The builder must
    # recover the original model traversal order from target_modules.
    for block in (block_0, block_10, block_1):
        role = _get_fsdp2_prefetch_role("target", block.name, block.name)
        _register_fsdp2_prefetch_module(block, block, role, roles_by_module_id)
        _register_fsdp2_prefetch_module(block, block, role, roles_by_module_id)

    chains = _build_fsdp2_backward_prefetch_chains(target_modules, roles_by_module_id)
    target_role = _get_fsdp2_prefetch_role("target", block_0.name, block_0.name)

    assert chains[target_role] == [block_0, block_1, block_10]
    assert all(block._fsdp_modules == [block] for block in (block_0, block_1, block_10))
