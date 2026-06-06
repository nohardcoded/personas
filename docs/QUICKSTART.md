# Quickstart - from a fresh Mac to generating

1. **Open Terminal** (⌘-Space -> "Terminal").
2. **Install Apple's build tools:** `xcode-select --install` (click Install, wait).
3. **Get the code:** `git clone <REPO-URL> persona-gen && cd persona-gen`
4. **Install:** `bash scripts/install.sh` - sets up Python, PyTorch (MPS), diffusers, and
   downloads the Z-Image model (Apache-2.0, no account) and writes a working `config.yaml`. Several GB; be patient. Safe to re-run.
5. **Add models & edit config:** open `config.yaml`, set the paths to your models. Z-Image is set up automatically by the installer. The easiest
   start is **Z-Image-Turbo** (Apache-2.0). See [MODELS.md](MODELS.md).
6. **Run:** `bash scripts/run.sh` -> your browser opens at http://localhost:7860.

Type a prompt in English, pick a format, click **Generate ▶**. The image builds from a
pixelated preview to the final; finished images land in the gallery (click for metadata,
re-run, or variations).

### Tips
- **Seed 0** = random each time; reuse a seed to reproduce an image.
- **Cmd+Enter** generates; **Esc** closes the metadata view.
- One heavy render runs at a time; the rest queue (remove/duplicate per job).
