# Models

> The installer auto-downloads **Z-Image-Turbo** (Apache-2.0, no token) and writes a working
> `config.yaml`. The table below is only needed for the optional curated SDXL engine.

Weights are **not** bundled - download them yourself and point `config.yaml` at the paths.
Each model has its own license; you accept it at the source.

| Slot | What | Where to get it | Notes |
|---|---|---|---|
| `zimage.model` | Z-Image-Turbo (diffusers dir) | HF `Tongyi-MAI/Z-Image-Turbo` | Apache-2.0, easiest start |
| `sdxl.model` | any SDXL checkpoint (diffusers dir) | HF / convert a `.safetensors` | the base for the curated pipeline |
| `sdxl.refiner` *(opt)* | a 2nd SDXL for the realism img2img | any photoreal SDXL | improves skin; skipped if absent |
| `sdxl.feet_lora` *(opt)* | a feet-fixing SDXL LoRA | - | only used when feet are detected |
| `sdxl.hand_lora` *(opt)* | a hand-fixing SDXL LoRA | - | only used when hands are detected |
| `detectors.yoloface` *(opt)* | yoloface ONNX | open yoloface model | enables face refine |
| `detectors.pose_task` *(opt)* | MediaPipe pose_landmarker | Google MediaPipe (Apache-2.0) | enables feet refine |
| `detectors.hand_task` *(opt)* | MediaPipe hand_landmarker | Google MediaPipe (Apache-2.0) | enables hands refine |

Everything under "optional" can be omitted - the curated SDXL engine simply skips that refine
stage. With only `sdxl.model` set you still get base + (if `refiner` set) the realism pass.

A single-file SDXL `.safetensors` must be converted to a diffusers folder first
(`diffusers` has a conversion script), then point `sdxl.model` at that folder.
