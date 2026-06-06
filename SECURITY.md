# Security

persona-gen is a **local, single-user** app. It binds to `127.0.0.1` (localhost only) by
default and has no authentication - it assumes that whoever can reach the port is you.

## Threat model

- **In scope:** a malicious web page in your browser trying to reach the local API
  (CSRF / DNS-rebinding), crafted files in the output directory, malformed requests, and
  resource-exhaustion (DoS) from runaway input.
- **Out of scope:** an attacker who already has a shell on your Mac or write access to your
  files. At that point the app is not your security boundary.

## What protects you

- **Localhost by default.** The server binds `127.0.0.1`. Exposing it on your LAN is an
  explicit opt-in (`--host 0.0.0.0` or `HOST=0.0.0.0 bash scripts/run.sh`) and prints a
  warning. There is no auth, so only do this on a network you trust.
- **DNS-rebinding guard.** Requests are rejected unless the `Host` header is a localhost
  name or a raw IP literal, so a rebound attacker domain cannot drive the API.
- **Cross-origin POST block.** State-changing requests carrying a foreign `Origin` are
  refused (CSRF defense-in-depth on top of the JSON-only API).
- **Path containment.** Every filename from a client is resolved and required to sit
  directly inside the output directory and end in `.png` - no traversal, no escaping the
  gallery folder.
- **Bounded input.** Request bodies, prompt length, the job queue, and the in-memory job
  log are all capped; over-limit requests get `413` / `429` instead of consuming the box.
- **Safe rendering.** The UI escapes untrusted strings and routes gallery actions through
  event delegation (no filename is ever interpolated into an inline handler), so a crafted
  filename cannot become stored XSS.
- **Symlink-safe lock.** The single-render lock is claimed with `O_CREAT|O_EXCL|O_NOFOLLOW`,
  so a pre-planted symlink in the lock directory cannot be used to clobber another file.
- **No code-exec surfaces.** Config is parsed with `yaml.safe_load`; sidecars are JSON.
  There is no `pickle`, `eval`, `exec`, or shell-out of user input.

## Reporting

This is a personal/OSS project with no SLA. If you find an issue, open a GitHub issue with
a clear description and reproduction. Do not include secrets or private prompts.
