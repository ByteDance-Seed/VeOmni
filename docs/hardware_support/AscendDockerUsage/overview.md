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
| Atlas A2 training products | amd64 + arm64 | `v0.1.11-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12-veomni` | [x86](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_910b.x86) / [arm](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_910b.arm) |
| Atlas A3 training products | arm64 | `v0.1.11-cann9.1.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.12-veomni` | [a3](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_a3) |

For the complete list of published image tags, see [Ascend Docker Supported Tags](supported_tags.md).

Notes:

- **Atlas A2 training products** ships two Dockerfiles (x86 and arm64); CI merges them into a single multi-arch image sharing one tag.
- **Atlas A3 training products** is arm64 only.
- The historical `8.3.rc2` Dockerfiles (`Dockerfile.ascend_8.3.rc2_a2.x86/.arm/a3`) remain in the repo but are no longer published.

---

## CANN 9.2.1

CANN 9.2.1 Dockerfiles cover Atlas A2 training products x86/ARM64 and Atlas A3 training products ARM64. All three use
system Python and include a uv cache and development dependencies. See the
[Atlas A2 training products guide](build_a2_docker.md), [Atlas A3 training products guide](build_a3_docker.md), and
[publication status](supported_tags.md#cann-921-pending-quay-publication).
The existing Atlas A2 training products/Atlas A3 training products builders select the CANN 9.2.1 Dockerfiles. See the
[build and publication rules](overview.md#build-and-publication-workflows).
NPU test workflows retain their existing environment.

## Quick Start

### Prerequisites: Install Driver

An Ascend NPU driver compatible with the container's CANN version must be installed on the host. See the [CANN Compatibility Matrix](https://www.hiascend.com/document) for the driver ↔ CANN version mapping.

### Running a Container

Pull the image from quay.io and run it:

```bash
# Atlas A2 training products, multi-arch
docker pull quay.io/ascend/veomni:v0.1.11-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12-veomni

# Atlas A3 training products
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

- [build_a2_docker.md](build_a2_docker.md) — Atlas A2 training products build and usage
- [build_a3_docker.md](build_a3_docker.md) — Atlas A3 training products build and usage

---

## How to Build Locally

Starting from a CANN base image, build locally with the in-repo Dockerfiles. For build commands, proxy configuration, and per-platform details, see:

- [build_a2_docker.md](build_a2_docker.md) (CANN 9.2.1 uses system Python on both architectures; older images differ)
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

## Security Risks

When running containers with these images, be aware of the following security risks:

- Running as the `root` user: Containers run as the `root` user by default, which can introduce security risks. In production environments, it is recommended to create a non-privileged user to run the application.

- Lack of CPU and memory resource limits: Not setting resource limits may cause a container to consume excessive system resources and affect host performance. It is recommended to use the `--cpus` and `--memory` parameters to limit resource usage.

- Device use with `rwm` permissions: The NPU device is assigned read, write, and `mknod` permissions. Although these are required for functionality, the scope of permissions should be carefully evaluated in security-sensitive environments.

---

## License

The CANN and MindSeries software included in these images is subject to their own licenses; see the [Ascend CANN community license information](https://www.hiascend.com/software/cann/community). The VeOmni framework's license is in its [GitHub repository](https://github.com/ByteDance-Seed/VeOmni).

As with all container images, the pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.

## Build and publication workflows

The existing Ascend workflows build CANN 9.2.1 images and publish to
`quay.io/ascend/veomni`:

| Workflow | Platforms | Image tag on `main` |
| --- | --- | --- |
| [Atlas A2 training products builder](../../../.github/workflows/docker-build-ascend-a2.yml) | `linux/amd64`, `linux/arm64` | `veomni-9.2.1-910b-ubuntu22.04-py3.12-latest` |
| [Atlas A3 training products builder](../../../.github/workflows/docker-build-ascend-a3.yml) | `linux/arm64` | `veomni-9.2.1-a3-ubuntu22.04-py3.12-latest` |

Both support manual dispatch, published releases and path-filtered pushes to
`main`. Atlas A2 training products watches its workflow and both CANN 9.2.1 Atlas A2 training products Dockerfiles, builds
on native x86 and ARM runners, and merges the digests into a multi-architecture
image. Atlas A3 training products watches its workflow and the CANN 9.2.1 Atlas A3 training products Dockerfile.
Release builds retain the existing release-tag and `.postN` naming scheme.
Upstream publication runs under the `ByteDance-Seed` owner using the existing
Quay credentials. Build jobs are restricted to the `ByteDance-Seed` owner.

## Validation scope

The builders construct and publish images, including the Dockerfiles'
package-version and import checks. They do not run NPU numerical tests. The existing NPU unit/e2e workflows retain their upstream
image and dependency setup; they do not validate these CANN 9.2.1 images.

Device validation remains a local procedure. Keep its scripts, logs and
environment-specific patches outside the committed source tree. Record the
exact image, architecture and package versions when interpreting results;
results from a modified local image do not validate a different Dockerfile.

An image's preinstalled system packages are separate from the project
environment created by `uv sync`. Local validation must explicitly select
which environment it tests. Any wheel overlays applied after synchronization
are additional environment changes, not part of the lockfile alone.

## uv cache included in the image

The CANN 9.2.1 Dockerfiles set `UV_CACHE_DIR=/opt/veomni/uv-cache` and
populate it with `uv sync --frozen` for their hardware extra and the `dev`
group. A disposable virtual environment is removed after warming the cache;
the existing system Python installation remains the runtime environment.
The cache is stored in the image layer, outside `/app`, rather than in a
BuildKit cache mount, so mounting a new source checkout does not hide it.

The warm-up also installs ninja, pybind11 and the target's Triton packages
through uv. Atlas A2 training products x86 additionally caches Triton-Ascend 3.2.1 from the Huawei
index used by the existing upstream workflow. This cache-only installation
does not change the image's runtime Triton-Ascend version.

Reuse requires the same architecture, compatible Python version, dependency
versions and package sources. Changing the lockfile or index, overriding
`UV_CACHE_DIR`, or installing an uncached package can still require network
access. The cache increases image size and build time; it does not guarantee
that every future dependency installation is offline.

The complete locked `dev` group is also exported with `uv export --frozen`
and installed with `uv pip` into the image's default Python. This includes
patchgen, pytest and the lint tools; builds check patchgen imports and its CLI.
The editable patchgen installation uses `/app/patchgen-pkg`. If a different
checkout replaces `/app`, run `uv sync` in that checkout to recreate its project
environment. A fresh CI venv still needs its own `uv sync --group dev`;
the image cache accelerates installation but does not replace synchronization.
All three runtime images pin Triton-Ascend to 3.2.2 and assert its version at
build time. The upstream CI's explicit Triton-Ascend 3.2.1 installation remains
an independent override; using that workflow does not test the image's 3.2.2.
