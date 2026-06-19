# ComfyUI-DiT360

Seamless **360° panoramic image generation** in ComfyUI, based on
[DiT360](https://github.com/Insta360-Research-Team/DiT360) (FLUX.1-dev + a LoRA
with circular padding).

**Fully ComfyUI-native, zero extra Python dependencies.** No `diffusers`,
`transformers`, `peft`, or `accelerate` — the nodes import only `torch`, `numpy`
and ComfyUI's own modules. The only things downloaded are model weights.

## Why native?

DiT360's whole inference-time contribution over stock FLUX is **circular
padding**: the packed latent grid is wrap-padded along the longitude (width)
axis and the RoPE position ids of the padded boundary columns are set to those
of the opposite real edge, so the 0°/360° seam is continuous. We reproduce this
on ComfyUI's native FLUX:

- the latent is wrap-padded in the sampler and cropped before VAE decode;
- a `post_input` model patch rewrites `img_ids` so padded columns carry the
  opposite edge's position ids (faithful to how the LoRA was trained).

Because it's native, you inherit ComfyUI's fp8 / GGUF / offload for free — you do
**not** need the ~37 GB unoptimized footprint from the paper.

## Nodes

| Node | Purpose |
|------|---------|
| **(down)Load DiT360 model(s)** | Loads a FLUX.1-dev checkpoint (existing file or auto-download fp8) and applies the DiT360 LoRA. Outputs `MODEL`, `CLIP`, `VAE`. |
| **DiT360 Panorama Sampler** | FLUX sampler with circular padding → seamless equirectangular panorama. |
| _Inpaint / Outpaint_ | _In progress_ — RF-inversion + masked feature sharing (see Status). |

## Models

- **FLUX.1-dev** — use any FLUX.1-dev checkpoint you already have (pick it in the
  loader dropdown), or choose **download** to fetch an fp8 build. The download
  repo/filename are editable inputs; point them at a non-gated mirror you trust.
  An all-in-one fp8 checkpoint (transformer + CLIP + T5 + VAE) is simplest. If
  your base is a bare diffusion model, load CLIP/VAE separately — see Notes.
- **DiT360 LoRA** — auto-downloaded from
  `Insta360-Research/DiT360-Panorama-Image-Generation` (override in the loader).

## Quick start

`workflows/dit360_t2p_api.json` is an **API-format** prompt. Node wiring:

```
(down)Load DiT360 model(s) ─ model ─────────────► DiT360 Panorama Sampler ─► VAEDecode ─► SaveImage
                          ├─ clip ► CLIPTextEncode (positive) ► FluxGuidance(2.8) ─► positive
                          ├─ clip ► CLIPTextEncode (negative "") ──────────────────► negative
                          └─ vae ─────────────────────────────────────────────────► VAEDecode
EmptySD3LatentImage (2048 x 1024) ──────────────► latent_image
```

Defaults from the paper: **2048×1024**, **28 steps**, **guidance 2.8**, sampler
CFG **1.0** (FLUX.1-dev). Prefix prompts with "This is a panorama." as upstream does.

> ⚠️ The model was trained only at 1024×2048; other sizes degrade.

## Status

- ✅ Load node + seamless text-to-panorama sampler — implemented; **pending
  on-GPU validation** (needs FLUX weights).
- 🚧 Inpaint / outpaint — native port of RF-inversion + the per-attention-layer
  masked feature-sharing processor. See `IMPLEMENTATION_NOTES.md`.

## License

MIT (matches upstream DiT360). See `LICENSE`.
