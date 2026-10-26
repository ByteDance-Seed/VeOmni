# Dev Box Environment

Two things about the ByteDance dev boxes that make an agent shell behave
differently from the user's terminal. Both look like "the network is down" or
"the tool is gone" and are neither.

## External network needs the proxy

The box has egress, but only through the corporate relay. An agent shell does
not inherit it, so `curl https://huggingface.co` and `lark-cli` fail while the
same commands succeed in the user's terminal.

```bash
P='<corporate-http-proxy-url>'
export https_proxy="$P" http_proxy="$P" HTTPS_PROXY="$P" HTTP_PROXY="$P"
export no_proxy='localhost,127.0.0.1' NO_PROXY='localhost,127.0.0.1'
```

DNS resolves without the proxy (NAT64 `64:ff9b::` addresses), so name lookup
succeeding proves nothing — the connect is what times out. `curl -x "$P"`
returning any HTTP status, including 404, means the path works.

Tests that reach the Hub fail without this. `tests/data/test_classification_data_processor.py`
pulls a real Qwen3 tokenizer and reports `[Errno 101] Network is unreachable`.

## Tools installed under nvm are off an agent's PATH

`lark-cli` is the global npm package `@larksuite/cli` at
`~/.nvm/versions/node/<version>/bin/`. The user's interactive shell gets it from
`nvm.sh` sourced in `~/.bashrc`; a non-interactive agent shell skips `~/.bashrc`
entirely, so the binary appears not to exist. Its credentials live separately in
`~/.local/share/lark-cli/` and are unaffected.

```bash
export PATH="$(echo ~/.nvm/versions/node/*/bin | tr ' ' ':'):$PATH"
```

Before concluding a CLI is missing, check `~/.nvm/versions/node/*/bin` — and
scope `find` with `-xdev`, pruning `/mnt`: the HDFS mount makes a bare `find /`
hang long enough to look like a failure.
