# Troubleshooting

The app is a local server on **http://localhost:7860**, one heavy render at a time guarded
by a lock file (`lock_file` in config, default `/tmp/persona-gen.lock`).

## Nuclear restart (fixes most issues)
```bash
pkill -f persona_gen.server
rm -f /tmp/persona-gen.lock
bash scripts/run.sh
```
Your images are safe in `output_dir` and reappear in the gallery after restart.

## Symptom -> fix
- **UI won't generate / frozen** -> click **⏹ Stop all**, else nuclear restart + hard-refresh (⌘⇧R).
- **Stuck render** -> `pkill -f persona_gen.server` (the render runs in the server process) + remove the lock.
- **"lock held"** -> `cat /tmp/persona-gen.lock`; if that PID is dead, `rm -f /tmp/persona-gen.lock`.
- **Gallery empty** -> images live in `output_dir`; hard-refresh; restart if the server died.
- **Port in use** -> `bash scripts/run.sh 7861` (another port).
- **"No config found"** -> `cp config.example.yaml config.yaml` and edit paths.
- **"MPS not available"** -> you're not on Apple Silicon, or install didn't finish.
- **A refine stage doesn't run** -> its model isn't set in `config.yaml` (face/feet/hands are optional).

## Health check
```bash
pgrep -fl persona_gen.server && curl -s localhost:7860/api/status | python3 -m json.tool | head
```
