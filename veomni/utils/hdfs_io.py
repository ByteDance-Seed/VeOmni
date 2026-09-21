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

import os
import shutil
from typing import IO, Any

from .logging import get_logger


logger = get_logger(__name__)


_HDFS_PREFIX = "hdfs://"

_HDFS_BIN_PATH = shutil.which("hdfs")


def exists(path: str, **kwargs) -> bool:
    r"""Works like os.path.exists() but supports hdfs.

    Test whether a path exists. Returns False for broken symbolic links.

    Args:
        path (str): path to test

    Returns:
        bool: True if the path exists, False otherwise
    """
    if _is_non_local(path):
        return _exists(path, **kwargs)
    return os.path.exists(path)


def _exists(file_path: str):
    """hdfs capable to check whether a file_path is exists"""
    if file_path.startswith("hdfs"):
        return _run_cmd(_hdfs_cmd(f"-test -e {file_path}")) == 0
    return os.path.exists(file_path)


def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    r"""Works like os.makedirs() but supports hdfs.

    Super-mkdir; create a leaf directory and all intermediate ones.  Works like
    mkdir, except that any intermediate path segment (not just the rightmost)
    will be created if it does not exist. If the target directory already
    exists, raise an OSError if exist_ok is False. Otherwise no exception is
    raised.  This is recursive.

    Args:
        name (str): directory to create
        mode (int): file mode bits
        exist_ok (bool): if True, do not raise an exception if the directory already exists
        kwargs: keyword arguments for hdfs

    """
    if _is_non_local(name):
        _mkdir(name, **kwargs)
    else:
        os.makedirs(name, mode=mode, exist_ok=exist_ok)


def _mkdir(file_path: str) -> bool:
    """hdfs mkdir"""
    if file_path.startswith("hdfs"):
        _run_cmd(_hdfs_cmd(f"-mkdir -p {file_path}"))
    else:
        os.makedirs(file_path, exist_ok=True)
    return True


def copy(src: str, dst: str, **kwargs) -> bool:
    r"""Works like shutil.copy() for file, and shutil.copytree for dir, and supports hdfs.

    Copy data and mode bits ("cp src dst"). Return the file's destination.
    The destination may be a directory.
    If source and destination are the same file, a SameFileError will be
    raised.

    Arg:
        src (str): source file path
        dst (str): destination file path
        kwargs: keyword arguments for hdfs copy

    Returns:
        str: destination file path

    """
    if _is_non_local(src) or _is_non_local(dst):
        return _copy(src, dst)
    else:
        if os.path.isdir(src):
            return shutil.copytree(src, dst, **kwargs)
        else:
            return shutil.copy(src, dst, **kwargs)


def _isdir(file_path: str) -> bool:
    """hdfs mkdir"""
    if file_path.startswith("hdfs"):
        _run_cmd(_hdfs_cmd(f"-test -d {file_path}"))
    else:
        return os.path.isdir(file_path)
    return True


def isdir(path: str, **kwargs) -> bool:
    r"""Works like os.path.isdir() but supports hdfs.

    Test whether a path is a directory.

    Args:
        path (str): path to test

    Returns:
        bool: True if the path is a directory, False otherwise
    """
    if _is_non_local(path):
        return _isdir(path, **kwargs)
    else:
        return os.path.isdir(path)


def _listdir(path: str, **kwargs) -> list:
    """hdfs listdir"""
    if path.startswith("hdfs"):
        _run_cmd(_hdfs_cmd(f"-ls {path}"))
    else:
        return os.listdir(path)
    return True


def listdir(path: str, **kwargs) -> list:
    r"""Works like os.listdir() but supports hdfs.

    List the directory contents.

    Args:
        path (str): path to list
        kwargs: keyword arguments for hdfs listdir

    Returns:
        list: list of directory contents

    """
    if _is_non_local(path):
        return _listdir(path, **kwargs)
    else:
        return os.listdir(path)


def _copy(from_path: str, to_path: str, timeout: int = None) -> bool:
    if to_path.startswith("hdfs"):
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(_hdfs_cmd(f"-cp -f {from_path} {to_path}"), timeout=timeout)
        else:
            returncode = _run_cmd(_hdfs_cmd(f"-put -f {from_path} {to_path}"), timeout=timeout)
    else:
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(
                _hdfs_cmd(
                    f"-get \
                {from_path} {to_path}"
                ),
                timeout=timeout,
            )
        else:
            try:
                shutil.copy(from_path, to_path)
                returncode = 0
            except shutil.SameFileError:
                returncode = 0
            except Exception as e:
                logger.warning(f"copy {from_path} {to_path} failed: {e}")
                returncode = -1
    return returncode == 0


def open(path: str, mode: str = "r") -> IO[Any]:
    r"""Work like open(). Support hdfs.

    NOTICE: please use in this way to avoid conflicts with Python builtin open:
    >>> from hdfs_io import fs
    >>> f = fs.open(path)

    Args:
        path (str): path to open
        mode (str): mode to open. same as Python builtin open() mode.
            all modes:
            'r' for reading (default)
            'w' for writing (truncating the file first)
            'x' for exclusive creation, failing if the file already exists
            'a' for appending to the end of the file
            'b' for binary mode
            't' for text mode (default)
            '+' for updating (reading and writing)

    Returns:
        IO[Any]: file object

    """
    if _is_non_local(path):
        try:
            from hdfs_io import hopen

            return hopen(path, mode)
        except ImportError:
            logger.warning("hdfs_io not installed, fallback to local open")
            import builtins

            return builtins.open(path, mode)
    elif path.endswith(".gz"):
        import gzip

        return gzip.open(path, mode)
    else:
        import builtins

        return builtins.open(path, mode)


def _run_cmd(cmd: str, timeout=None):
    return os.system(cmd)


def _hdfs_cmd(cmd: str) -> str:
    return f"{_HDFS_BIN_PATH} dfs {cmd}"


def _is_non_local(path: str):
    return path.startswith(_HDFS_PREFIX)
