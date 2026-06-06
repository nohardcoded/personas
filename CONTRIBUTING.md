# Contributing

Thanks for your interest! persona-gen is a local-first image generation studio.

## Dev setup
```bash
bash scripts/install.sh          # uv venv + deps + config.yaml
cp config.example.yaml config.yaml   # if not created; point it at your models
PYTHONPATH=src .venv/bin/python -m persona_gen.server --port 7860
```

## Ground rules
- **No model weights, generated images, secrets, or personal data in commits** (see `.gitignore`).
- Keep model paths in `config.yaml`; never hardcode paths in code.
- Python: 4-space indent, standard library + the pinned deps; keep the server a single process.
- The UI lives in `src/persona_gen/index.html`; escape any user/sidecar value with `esc()`

## Reporting issues
Open an issue with: macOS version, chip, what you ran, the error, and your (sanitized)
`config.yaml` engine block.
