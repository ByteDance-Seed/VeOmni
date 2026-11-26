# Ascend Docker Overview

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

## Tag Naming Rules and Supported Images

### Tag Naming Rules

All image tags follow this pattern:

```
<veomni_version>-cann<CANN>-torch_npu<torch_npu>-<chip_series>-<os>-py<python>-veomni
```

| Field | Example Values | Description |
|---|---|---|
| `veomni-version` | `v0.1.12` | VeOmni release tag |
| `cann` | `9.2.1` | CANN version |
| `torch_npu` | `2.10.0.post7` | torch-npu version |
| `chip-series` | `910b`, `a3`, `a5` | Target Ascend chip series |
| `os` | `ubuntu22.04` | Base operating system |
| `python` | `py3.12` | Python version |
| `veomni` | `-veomni` | Marks the image as built by VeOmni |

> Note: Tags are mutable — rebuilding on the same release tag overwrites the image of the same name.

### Image List

| Chip Series | Architecture | Tag Example | Dockerfile |
|---|---|---|---|
| Atlas A2 training products | amd64 + arm64 | `v0.1.12-cann9.2.1-torch_npu2.10.0.post7-910b-ubuntu22.04-py3.12-veomni` | [x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_910b.x86) / [arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_910b.arm) |
| Atlas A3 training products | arm64 | `v0.1.12-cann9.2.1-torch_npu2.10.0.post7-a3-ubuntu22.04-py3.12-veomni` | [Atlas A3 training products](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a3) |
| Ascend 950PR&950DT products | arm64 | `v0.1.12-cann9.2.1-torch_npu2.10.0.post7-a5-ubuntu22.04-py3.12-veomni` | [Ascend 950PR&950DT products](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a5) |

For the complete list of published image tags, see [Ascend Docker Supported Tags](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/hardware_support/AscendDockerUsage/supported_tags.md).

Notes:

- **Atlas A2 training products** ships two Dockerfiles (x86 and arm64); CI merges them into a single multi-arch image sharing one tag.
- **Atlas A3 training products** is arm64 only.
- **Ascend 950PR&950DT products** is arm64 only, using CANN 9.2.1.
- The historical `8.3.rc2` Dockerfiles (`Dockerfile.ascend_8.3.rc2_a2.x86/.arm/a3`) remain in the repo but are no longer published.

---

## Quick Start

### Prerequisites: Install Driver

An Ascend NPU driver compatible with the container's CANN version must be installed on the host. See the [CANN Compatibility Matrix](https://www.hiascend.com/document) for the driver ↔ CANN version mapping.

### Running a Container

Pull the image from quay.io and run it:

```bash
# Atlas A2 training products, multi-arch
docker pull quay.io/ascend/veomni:v0.1.12-cann9.2.1-torch_npu2.10.0.post7-910b-ubuntu22.04-py3.12-veomni

# Atlas A3 training products
docker pull quay.io/ascend/veomni:v0.1.12-cann9.2.1-torch_npu2.10.0.post7-a3-ubuntu22.04-py3.12-veomni

# Ascend 950PR&950DT products (A5)
docker pull quay.io/ascend/veomni:v0.1.12-cann9.2.1-torch_npu2.10.0.post7-a5-ubuntu22.04-py3.12-veomni
```

The following command starts the A2 image with one NPU device. For A3, replace the image tag with the A3 tag you pulled above. For A5, use the launch command in the [A5 build guide](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/hardware_support/AscendDockerUsage/build_a5_docker.md), keeping its device mappings and replacing `ascend-a5-env:v1` with the full image reference from the A5 `docker pull` command above.

To expose more NPUs, add a separate device argument for each one, for example `--device=/dev/davinci1` for a second device.

```bash
docker run --runtime=runc -it \
  --ulimit nproc=65535 \
  --ulimit nofile=65535 \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --shm-size=64G \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools:ro \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro \
  quay.io/ascend/veomni:v0.1.12-cann9.2.1-torch_npu2.10.0.post7-910b-ubuntu22.04-py3.12-veomni \
  /bin/bash
```

For complete, advanced mount and training configuration (checkpoint / dataset mounts, training command examples) per platform, see:

- [build_a2_docker.md](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/hardware_support/AscendDockerUsage/build_a2_docker.md) — Atlas A2 training products build and usage
- [build_a3_docker.md](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/hardware_support/AscendDockerUsage/build_a3_docker.md) — Atlas A3 training products build and usage
- [build_a5_docker.md](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/hardware_support/AscendDockerUsage/build_a5_docker.md) — Ascend 950PR&950DT products build and usage

---

## How to Build Locally

Starting from a CANN base image, build locally with the in-repo Dockerfiles. For build commands, proxy configuration, and per-platform details, see:

- [build_a2_docker.md](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/hardware_support/AscendDockerUsage/build_a2_docker.md) (x86 uses uv, arm uses pip — two environments)
- [build_a3_docker.md](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/hardware_support/AscendDockerUsage/build_a3_docker.md)
- [build_a5_docker.md](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/hardware_support/AscendDockerUsage/build_a5_docker.md)

---

## Development

Use the VeOmni image as the base image and add your own software:

```dockerfile
FROM quay.io/ascend/veomni:v0.1.12-cann9.2.1-torch_npu2.10.0.post7-910b-ubuntu22.04-py3.12-veomni

RUN apt update -y && \
    apt install gcc ...

...
```

---

## Security Risks

When running containers with these images, be aware of the following security risks:

- Running as the `root` user: Containers run as the `root` user by default, which can introduce security risks. In production environments, it is recommended to create a non-privileged user to run the application.

- Lack of CPU and memory resource limits: Not setting resource limits may cause a container to consume excessive system resources and affect host performance. It is recommended to use the `--cpus` and `--memory` parameters to limit resource usage.

- Device use with `rwm` permissions: The NPU device is assigned read, write, and `mknod` permissions. Although these are required for functionality, the scope of permissions should be carefully evaluated in security-sensitive environments.

---

## License

The CANN and MindSeries software included in these images is subject to their own licenses; see the [Ascend CANN community license information](https://www.hiascend.com/software/cann/community). The VeOmni framework's license is in its [GitHub repository](https://github.com/ByteDance-Seed/VeOmni).

As with all container images, the pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
