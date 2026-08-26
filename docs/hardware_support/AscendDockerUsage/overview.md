# VeOmni Ascend Docker Images

## Quick Reference

- VeOmni is maintained by [ByteDance Seed](https://github.com/ByteDance-Seed/VeOmni).
- Images are published at [quay.io/ascend/veomni](https://quay.io/repository/ascend/veomni).
- Where to get help:

    - [VeOmni GitHub](https://github.com/ByteDance-Seed/VeOmni)
    - [VeOmni Documentation](https://veomni.readthedocs.io/en/latest/index.html)

---

## VeOmni Ascend Image

The VeOmni Ascend images build on Huawei's [CANN (Compute Architecture for Neural Networks)](https://www.hiascend.com/cann) base images, layering the VeOmni framework and Ascend NPU dependencies on top, for training models on Ascend hardware.

---

## Supported Images and Tag Naming Rules

### Tag Naming Rules

All image tags follow this pattern:

```
<veomni_version>-cann<CANN>-torch_npu<torch_npu>-<chip_series>-<os>-py<python>-veomni
```

| Field | Example Values | Description |
|---|---|---|
| `veomni-version` | `v0.1.11` | VeOmni release tag |
| `cann` | `9.1.0` | CANN version |
| `torch_npu` | `2.10.0.post2` | torch-npu version |
| `chip-series` | `910b`, `a3` | Target Ascend chip series |
| `os` | `ubuntu22.04` | Base operating system |
| `python` | `py3.12` | Python version |
| `veomni` | `-veomni` | Marks the image as built by VeOmni |

> Note: Tags are mutable — rebuilding on the same release tag overwrites the image of the same name.

### Image List

| Chip Series | Architecture | Tag Example | Dockerfile |
|---|---|---|---|
| 910B (A2) | amd64 + arm64 | `v0.1.11-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12-veomni` | [x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_910b.x86) / [arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_910b.arm) |
| A3 | arm64 | `v0.1.11-cann9.1.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.12-veomni` | [a3](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_a3) |

Notes:

- **A2 (910B)** ships two Dockerfiles (x86 and arm64); CI merges them into a single multi-arch image sharing one tag.
- **A3** is arm64 only.
- The historical `8.3.rc2` Dockerfiles (`Dockerfile.ascend_8.3.rc2_a2.x86/.arm/a3`) remain in the repo but are no longer published.

---

## Quick Start

### Prerequisites: Install Driver

An Ascend NPU driver compatible with the container's CANN version must be installed on the host. See the [CANN Compatibility Matrix](https://www.hiascend.com/document) for the driver ↔ CANN version mapping.

### Running a Container

Pull the image from quay.io and run it:

```bash
# 910B (A2), multi-arch
docker pull quay.io/ascend/veomni:v0.1.11-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12-veomni

# A3
docker pull quay.io/ascend/veomni:v0.1.11-cann9.1.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.12-veomni
```

To run the container, mount the Ascend device files and the host driver directories:

```bash
docker run --runtime=runc -it \
  --ulimit nproc=65535 \
  --ulimit nofile=65535 \
  --device=/dev/davinci* \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --shm-size=64G \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools:ro \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro \
  quay.io/ascend/veomni:v0.1.11-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12-veomni \
  /bin/bash
```

For complete, advanced mount and training configuration (checkpoint / dataset mounts, training command examples) per platform, see:

- [build_a2_docker.md](build_a2_docker.md) — A2 (910B) build and usage
- [build_a3_docker.md](build_a3_docker.md) — A3 build and usage

---

## How to Build Locally

Starting from a CANN base image, build locally with the in-repo Dockerfiles. For build commands, proxy configuration, and per-platform details, see:

- [build_a2_docker.md](build_a2_docker.md) (x86 uses uv, arm uses pip — two environments)
- [build_a3_docker.md](build_a3_docker.md)

---

## Development

Use the VeOmni image as the base image and add your own software:

```dockerfile
FROM quay.io/ascend/veomni:v0.1.11-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12-veomni

RUN apt update -y && \
    apt install gcc ...

...
```

---

## License

The CANN and MindSeries software included in these images is subject to their own licenses; see the [Ascend CANN community license information](https://www.hiascend.com/software/cann/community). The VeOmni framework's license is in its [GitHub repository](https://github.com/ByteDance-Seed/VeOmni).

As with all container images, the pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
