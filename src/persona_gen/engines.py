"""Generation engines for persona-gen - config-driven, self-contained.

Two reference engines:
  - ZImageEngine:      fast single-pass Z-Image (Tongyi-MAI/Z-Image-Turbo), great for SFW.
  - CuratedSDXLEngine: any SDXL checkpoint -> optional refiner img2img -> detector-gated
                       face / feet / hands diffusion refine. A general curated photoreal
                       pipeline; bring your own checkpoints + (optional) refine LoRAs.

Everything (model paths, LoRAs, detector models) comes from config - no hardcoded paths.
The face/feet/hands curation is a generic, widely-documented technique (detect a region,
crop, low-strength img2img, feather-paste). No private content, banks, or personas here.

fp32 on Apple Silicon MPS (fp16 can produce black/NaN VAE output on some setups).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFilter

# ---- generic conditioning (no rating tag - the prompt decides content) -----
PREFIX = "score_9, score_8_up, score_7_up, photorealistic, raw photo, "
NEG = (
    "worst quality, low quality, blurry, deformed, extra limbs, extra arms, extra legs, "
    "mutated hands, fused fingers, bad anatomy, deformed eyes, asymmetric eyes, watermark, "
    "text, signature, cartoon, anime, 3d render, plastic skin, airbrushed, doll"
)
REFINER_PROMPT = (
    PREFIX + "naturalistic detailed skin, film grain, photorealistic, natural skin texture"
)

FACE_PROMPT = (
    "RAW photo, photorealistic detailed face, sharp focus, glassy reflective eyes, "
    "bright catchlight, sharp detailed iris and round pupils, natural skin texture, film grain"
)
FACE_NEG = (
    "blurry, dull flat eyes, deformed eyes, asymmetric eyes, extra pupils, cross-eyed, "
    "plastic skin, airbrushed, waxy smooth skin, doll, cgi, 3d render, cartoon, anime, lowres"
)
FACE_STRENGTH, FACE_STEPS, FACE_CFG, FACE_WORK = 0.40, 28, 5.5, 1024

FEET_PROMPT = (
    "RAW photo, detailed photorealistic human foot, well-formed five toes, separated "
    "natural toes, defined toenails, natural arch, realistic skin texture, sharp focus, film grain"
)
FEET_NEG = (
    "extra toes, missing toes, fused toes, webbed toes, melted foot, deformed ankle, blurry, "
    "doll feet, duplicate foot, mutated, lowres"
)
FEET_STRENGTH, FEET_STEPS, FEET_CFG, FEET_WORK, FEET_LORA_W = 0.45, 28, 5.0, 768, 0.7

HAND_PROMPT = (
    "RAW photo, human hand, natural five-finger anatomy, correct thumb placement, natural "
    "knuckles, fingernails, skin creases, realistic skin texture, sharp focus"
)
HAND_NEG = (
    "extra fingers, six fingers, missing fingers, fused fingers, webbed fingers, duplicate hand, "
    "detached fingers, claw, melted hand, deformed hand, giant finger, elongated finger, blurry"
)
HAND_STRENGTH, HAND_STEPS, HAND_CFG, HAND_WORK, HAND_LORA_W = 0.28, 28, 5.0, 768, 0.45

_POSE_FEET_LM = {"L": (27, 29, 31), "R": (28, 30, 32)}  # mediapipe Pose ankle/heel/foot_index


# ---- detectors (generic CV; gated on the model files being present) --------
def _exists(p) -> bool:
    return bool(p) and Path(p).expanduser().exists()


class _Detectors:
    def __init__(self, cfg: dict):
        self.yoloface_path = cfg.get("yoloface")
        self.pose_path = cfg.get("pose_task")
        self.hand_path = cfg.get("hand_task")
        self._yolo = self._pose = self._hands = None

    def yolo(self):
        if self._yolo is None:
            import onnxruntime as ort

            self._yolo = ort.InferenceSession(
                str(Path(self.yoloface_path).expanduser()), providers=["CPUExecutionProvider"]
            )
        return self._yolo

    def face_kps(self, pil_img):
        if not _exists(self.yoloface_path):
            return []
        import cv2
        import numpy as np

        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        blob = (
            cv2.cvtColor(cv2.resize(bgr, (640, 640)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )
        blob = blob.transpose(2, 0, 1)[None]
        sess = self.yolo()
        out = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0].T
        keep = out[:, 4] > 0.5
        if not keep.any():
            return []
        out = out[keep]
        idx = int(np.argmax(out[:, 4]))
        kps = out[idx][5:20].reshape(5, 3)[:, :2].astype(np.float32).copy()
        kps[:, 0] *= w / 640.0
        kps[:, 1] *= h / 640.0
        return [kps]

    def pose(self):
        if self._pose is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(Path(self.pose_path).expanduser())
                    ),
                    num_poses=1,
                    running_mode=vision.RunningMode.IMAGE,
                )
            )
        return self._pose

    def foot_boxes(self, pil_img, vis=0.4):
        if not _exists(self.pose_path):
            return []
        import mediapipe as mp
        import numpy as np

        rgb = np.array(pil_img.convert("RGB"))
        h, w = rgb.shape[:2]
        res = self.pose().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        out = []
        for person in res.pose_landmarks or []:
            for idxs in _POSE_FEET_LM.values():
                pts = [
                    (person[i].x * w, person[i].y * h) for i in idxs if person[i].visibility >= vis
                ]
                if len(pts) < 2:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                side = int(min(max(max(max(xs) - min(xs), max(ys) - min(ys)) * 2.6, 96), w, h))
                L = int(min(max(cx - side / 2, 0), w - side))
                T = int(min(max(cy - side / 2, 0), h - side))
                out.append((L, T, L + side, T + side))
        return out

    def hands(self):
        if self._hands is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(Path(self.hand_path).expanduser())
                    ),
                    num_hands=4,
                    min_hand_detection_confidence=0.35,
                    min_hand_presence_confidence=0.45,
                    running_mode=vision.RunningMode.IMAGE,
                )
            )
        return self._hands

    def hand_boxes(self, pil_img):
        if not _exists(self.hand_path):
            return []
        import mediapipe as mp
        import numpy as np

        rgb = np.array(pil_img.convert("RGB"))
        h, w = rgb.shape[:2]
        res = self.hands().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        out = []
        for hand in res.hand_landmarks or []:
            xs, ys = [lm.x * w for lm in hand], [lm.y * h for lm in hand]
            if min(xs) < -5 or min(ys) < -5 or max(xs) > w + 5 or max(ys) > h + 5:
                continue
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            side = int(min(max(max(max(xs) - min(xs), max(ys) - min(ys)) * 1.7, 110), w, h))
            L = int(min(max(cx - side / 2, 0), w - side))
            T = int(min(max(cy - side / 2, 0), h - side))
            out.append((L, T, L + side, T + side))
        return out


def _face_box(kps, iw, ih, pad=1.6):
    x0, y0 = float(kps[:, 0].min()), float(kps[:, 1].min())
    x1, y1 = float(kps[:, 0].max()), float(kps[:, 1].max())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - (y1 - y0) * 0.25
    side = int(min(max(int(max(x1 - x0, y1 - y0) * pad * 2), 64), iw, ih))
    L = int(min(max(cx - side / 2, 0), iw - side))
    T = int(min(max(cy - side / 2, 0), ih - side))
    return (L, T, L + side, T + side)


def _feather(cw, ch):
    m = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(m).ellipse([cw * 0.07, ch * 0.05, cw * 0.93, ch * 0.97], fill=255)
    return m.filter(ImageFilter.GaussianBlur(max(cw, ch) * 0.05))


def _load_lora(pipe, path, name):
    try:
        if not _exists(path):
            return False
        pipe.load_lora_weights(str(Path(path).expanduser()), adapter_name=name)
        return True
    except Exception:
        return False


# ---- engines ---------------------------------------------------------------
class ZImageEngine:
    label_default = "Z-Image"

    def __init__(self, cfg: dict):
        self.model = str(Path(cfg["model"]).expanduser())
        self.max_seq = int(cfg.get("max_sequence_length", 768))
        self.pipe = None

    def load(self):
        if self.pipe is None:
            from diffusers import ZImagePipeline

            self.pipe = ZImagePipeline.from_pretrained(
                self.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
            ).to("mps")
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
            self.pipe.set_progress_bar_config(disable=True)

    def render(
        self,
        prompt,
        negative=None,
        seed=0,
        width=832,
        height=1216,
        steps=9,
        cfg=0.0,
        progress=None,
        cancel=None,
    ):
        self.load()
        g = torch.Generator("mps").manual_seed(int(seed))

        def _cb(_p, step, _t, kw):
            if cancel and cancel():
                raise RuntimeError("cancelled by user")
            if progress:
                progress(
                    {
                        "stage": None,
                        "step": step + 1,
                        "total_steps": steps,
                        "latents": kw.get("latents"),
                    }
                )
            return kw

        kw = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            max_sequence_length=self.max_seq,
            generator=g,
        )
        try:
            return self.pipe(
                callback_on_step_end=_cb, callback_on_step_end_tensor_inputs=["latents"], **kw
            ).images[0]
        except TypeError:
            return self.pipe(**kw).images[0]


class CuratedSDXLEngine:
    label_default = "SDXL (curated)"

    def __init__(self, cfg: dict):
        self.base = str(Path(cfg["model"]).expanduser())
        self.refiner = cfg.get("refiner")  # optional SDXL img2img refiner (e.g. a realism model)
        self.feet_lora = cfg.get("feet_lora")  # optional
        self.hand_lora = cfg.get("hand_lora")  # optional
        self.det = _Detectors(cfg.get("detectors", {}))
        self.refine_strength = float(cfg.get("refiner_strength", 0.35))
        self.pony = self.compel = self.i2i = None
        self.has_feet = self.has_hand = False

    def load(self):
        if self.pony is not None:
            return
        from compel import Compel, ReturnedEmbeddingsType
        from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

        self.pony = StableDiffusionXLPipeline.from_pretrained(
            self.base, torch_dtype=torch.float32
        ).to("mps")
        if hasattr(self.pony, "enable_attention_slicing"):
            self.pony.enable_attention_slicing()
        self.pony.set_progress_bar_config(disable=True)
        self.compel = Compel(
            tokenizer=[self.pony.tokenizer, self.pony.tokenizer_2],
            text_encoder=[self.pony.text_encoder, self.pony.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
        # img2img refiner: a separate checkpoint if given, else reuse the base pipe
        if self.refiner and _exists(self.refiner):
            base = StableDiffusionXLPipeline.from_pretrained(
                str(Path(self.refiner).expanduser()), torch_dtype=torch.float32
            ).to("mps")
            self.i2i = StableDiffusionXLImg2ImgPipeline.from_pipe(base)
        else:
            self.i2i = StableDiffusionXLImg2ImgPipeline.from_pipe(self.pony)
        self.i2i.set_progress_bar_config(disable=True)
        self.has_feet = _load_lora(self.i2i, self.feet_lora, "feet")
        self.has_hand = _load_lora(self.i2i, self.hand_lora, "hand")
        if self.has_feet or self.has_hand:
            self.i2i.disable_lora()

    def render(
        self,
        prompt,
        negative=None,
        seed=0,
        width=832,
        height=1216,
        steps=28,
        cfg=6.5,
        progress=None,
        cancel=None,
    ):
        self.load()
        neg = negative or NEG
        if cancel and cancel():
            raise RuntimeError("cancelled by user")

        def _cb(_p, step, _t, kw):
            if cancel and cancel():
                raise RuntimeError("cancelled by user")
            if progress:
                progress(
                    {
                        "stage": "1/5 base",
                        "step": step + 1,
                        "total_steps": steps,
                        "latents": kw.get("latents"),
                    }
                )
            return kw

        # STAGE 1: base SDXL (Compel long prompt)
        cond, pool = self.compel(PREFIX + prompt)
        ncond, npool = self.compel(neg)
        empty = self.compel.build_conditioning_tensor("")[0]
        c, nc = self.compel.pad_conditioning_tensors_to_same_length(
            [cond, ncond], precomputed_padding=empty
        )
        g = torch.Generator("mps").manual_seed(int(seed))
        img = self.pony(
            prompt_embeds=c,
            pooled_prompt_embeds=pool,
            negative_prompt_embeds=nc,
            negative_pooled_prompt_embeds=npool,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            generator=g,
            callback_on_step_end=_cb,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

        # STAGE 2: refiner img2img (skin realism)
        if cancel and cancel():
            raise RuntimeError("cancelled by user")
        if self.has_feet or self.has_hand:
            self.i2i.disable_lora()
        if progress:
            progress({"stage": "2/5 refine", "step": 0, "total_steps": 30})
        gr = torch.Generator("mps").manual_seed(int(seed))
        final = self.i2i(
            prompt=REFINER_PROMPT + ", " + prompt,
            negative_prompt=neg,
            image=img,
            strength=self.refine_strength,
            num_inference_steps=30,
            guidance_scale=5.0,
            generator=gr,
        ).images[0]

        # STAGE 3: face refine
        if cancel and cancel():
            raise RuntimeError("cancelled by user")
        if progress:
            progress({"stage": "3/5 face", "step": 0, "total_steps": FACE_STEPS})
        for kps in self.det.face_kps(final):
            box = _face_box(kps, final.width, final.height)
            crop = final.crop(box)
            cw, ch = crop.size
            up = crop.resize((FACE_WORK, FACE_WORK), Image.LANCZOS)
            gf = torch.Generator("mps").manual_seed(int(seed) + 1)
            ref = self.i2i(
                prompt=FACE_PROMPT,
                negative_prompt=FACE_NEG,
                image=up,
                strength=FACE_STRENGTH,
                num_inference_steps=FACE_STEPS,
                guidance_scale=FACE_CFG,
                generator=gf,
            ).images[0]
            base = final.copy()
            base.paste(ref.resize((cw, ch), Image.LANCZOS), (box[0], box[1]), _feather(cw, ch))
            final = base

        # STAGE 4: feet refine (detector-gated)
        if self.has_feet and not (cancel and cancel()):
            if progress:
                progress({"stage": "4/5 feet", "step": 0, "total_steps": FEET_STEPS})
            boxes = self.det.foot_boxes(final)
            if boxes:
                self.i2i.set_adapters(["feet"], [FEET_LORA_W])
                self.i2i.enable_lora()
                for k, b in enumerate(boxes):
                    fc = final.crop(b)
                    fcw, fch = fc.size
                    up = fc.resize((FEET_WORK, FEET_WORK), Image.LANCZOS)
                    gg = torch.Generator("mps").manual_seed(int(seed) + 100 + k)
                    rf = self.i2i(
                        prompt=FEET_PROMPT,
                        negative_prompt=FEET_NEG,
                        image=up,
                        strength=FEET_STRENGTH,
                        num_inference_steps=FEET_STEPS,
                        guidance_scale=FEET_CFG,
                        generator=gg,
                    ).images[0]
                    final.paste(
                        rf.resize((fcw, fch), Image.LANCZOS), (b[0], b[1]), _feather(fcw, fch)
                    )
                self.i2i.disable_lora()

        # STAGE 5: hands refine (detector-gated)
        if self.has_hand and not (cancel and cancel()):
            if progress:
                progress({"stage": "5/5 hands", "step": 0, "total_steps": HAND_STEPS})
            boxes = self.det.hand_boxes(final)
            if boxes:
                self.i2i.set_adapters(["hand"], [HAND_LORA_W])
                self.i2i.enable_lora()
                for k, b in enumerate(boxes):
                    hc = final.crop(b)
                    hcw, hch = hc.size
                    up = hc.resize((HAND_WORK, HAND_WORK), Image.LANCZOS)
                    gg = torch.Generator("mps").manual_seed(int(seed) + 200 + k)
                    rf = self.i2i(
                        prompt=HAND_PROMPT,
                        negative_prompt=HAND_NEG,
                        image=up,
                        strength=HAND_STRENGTH,
                        num_inference_steps=HAND_STEPS,
                        guidance_scale=HAND_CFG,
                        generator=gg,
                    ).images[0]
                    final.paste(
                        rf.resize((hcw, hch), Image.LANCZOS), (b[0], b[1]), _feather(hcw, hch)
                    )
                self.i2i.disable_lora()
        return final


_TYPES = {"zimage": ZImageEngine, "sdxl": CuratedSDXLEngine}


def build_engines(config: dict) -> dict:
    """Build {key: {engine, label, steps, cfg}} from config['engines']."""
    out = {}
    for key, ec in (config.get("engines") or {}).items():
        cls = _TYPES.get(ec.get("type"))
        if not cls:
            continue
        out[key] = {
            "engine": cls(ec),
            "label": ec.get("label") or cls.label_default,
            "steps": int(ec.get("steps", 28 if ec.get("type") == "sdxl" else 9)),
            "cfg": float(ec.get("cfg", 6.5 if ec.get("type") == "sdxl" else 0.0)),
        }
    return out
