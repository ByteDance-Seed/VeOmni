import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "veomni" / "distributed" / "mesh_topology.py"
SPEC = importlib.util.spec_from_file_location("veomni_mesh_topology_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_extra_parallel_mesh_spec = MODULE.build_extra_parallel_mesh_spec


def _groups_for_dimension(
    shape: tuple[int, ...],
    dim_names: tuple[str, ...],
    selected_dim: str,
) -> tuple[tuple[int, ...], ...]:
    selected_index = dim_names.index(selected_dim)
    strides: list[int] = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    strides.reverse()

    groups: list[tuple[int, ...]] = []
    for rank in range(stride):
        coordinates = [(rank // strides[index]) % shape[index] for index in range(len(shape))]
        if coordinates[selected_index] != 0:
            continue
        members = []
        for selected_coordinate in range(shape[selected_index]):
            member_coordinates = list(coordinates)
            member_coordinates[selected_index] = selected_coordinate
            members.append(sum(member_coordinates[index] * strides[index] for index in range(len(shape))))
        groups.append(tuple(members))
    return tuple(groups)


def test_inner_extra_parallel_keeps_ep_groups_contiguous() -> None:
    spec = build_extra_parallel_mesh_spec(
        dp_replicate_size=1,
        dp_shard_sp_size=32,
        parallel_size=8,
        parallel_name="ep",
        parallel_outside=False,
    )

    assert spec.shape == (4, 8)
    assert spec.dim_names == ("ep_fsdp", "ep")
    assert _groups_for_dimension(spec.shape, spec.dim_names, "ep") == tuple(
        tuple(range(start, start + 8)) for start in range(0, 32, 8)
    )


def test_outer_extra_parallel_keeps_fsdp_groups_contiguous() -> None:
    spec = build_extra_parallel_mesh_spec(
        dp_replicate_size=1,
        dp_shard_sp_size=32,
        parallel_size=8,
        parallel_name="ep",
        parallel_outside=True,
    )

    assert spec.shape == (8, 4)
    assert spec.dim_names == ("ep", "ep_fsdp")
    assert spec.fsdp_dim_names == ("ep_fsdp",)
    assert _groups_for_dimension(spec.shape, spec.dim_names, "ep_fsdp") == tuple(
        tuple(range(start, start + 4)) for start in range(0, 32, 4)
    )


def test_hsdp_replicate_dimension_stays_outermost() -> None:
    spec = build_extra_parallel_mesh_spec(
        dp_replicate_size=2,
        dp_shard_sp_size=32,
        parallel_size=8,
        parallel_name="ep",
        parallel_outside=True,
    )

    assert spec.shape == (2, 8, 4)
    assert spec.dim_names == ("ep_replicate", "ep", "ep_fsdp")
    assert spec.fsdp_dim_names == ("ep_replicate", "ep_fsdp")


def test_extra_parallel_size_must_divide_shard_domain() -> None:
    try:
        build_extra_parallel_mesh_spec(
            dp_replicate_size=1,
            dp_shard_sp_size=30,
            parallel_size=8,
            parallel_name="ep",
            parallel_outside=True,
        )
    except ValueError as error:
        assert "must divide" in str(error)
    else:
        raise AssertionError("expected a non-divisible mesh to be rejected")
