# Implementation notes

How each DiT360 mechanism maps onto ComfyUI-native FLUX. No diffusers/transformers/peft.

## Text-to-panorama (validated logic, pending on-GPU pixels)

Upstream (`src/pipeline.py`) wrap-pads the packed latent + RoPE ids once before
the denoise loop and strips after. Native equivalent (`dit360/padding.py`,
`nodes_t2p.py`):

- `circular_pad_width` wrap-pads the `[B,16,H,W]` latent by 1 token column (2
  latent px) per side; cropped after sampling. Persistent across all steps,
  matching upstream.
- `set_model_post_input_patch(make_wrap_ids_patch())` rewrites `img_ids` so the
  padded boundary columns carry the *opposite* real edge's position ids — the
  faithful wrap-id scheme the LoRA was trained with.
- The noise is generated unpadded then wrap-padded so the pad columns are exact
  copies of the opposite edge (matches `torch.cat([X[...,-p:], X, X[...,:p]])`).

## Inpaint / outpaint (EXPERIMENTAL — needs on-GPU validation)

Two upstream pieces, both ported natively:

### 1. Masked feature sharing — `dit360/attention.py`
Upstream `PersonalizeAnythingAttnProcessor` sets `r_q/r_k/r_v` from
`r_hidden_states`, where the edit branch's masked tokens are replaced by the
source branch's features, gated by `timestep > tau`.

Native: ComfyUI's `attn1_patch` hook (both double & single FLUX blocks) hands us
`q,k,v` *after* modulation+projection and `extra_options["img_slice"]` (text
offset). We copy `q/k/v[1, :, mask, :] = q/k/v[0, :, mask, :]` — identical effect,
no block reimplementation. Verified numerically (inject + tau-gate + no leak).

### 2. RF-inversion + controller — `dit360/flow.py`, `nodes_edit.py`
`apply_model` returns the flow velocity (`v = noise - x0`) = upstream's
`noise_pred`. We reproduce:

- **Invert** (Alg. 1, forward ODE): `Y += [v + γ(v_cond − v)]·dt`, `v_cond = (y₁ − Y)/(1−t)`.
- **Edit** (reverse ODE): 2-item `[source, edit]` batch. Edit branch = plain
  euler `x − v·dt`. Source branch = controller `v̂ = v + η(v_cond − v)`,
  `v_cond = (y₀ − x)/(1−t)`, η active only in `[start, stop)`. Masked attention
  shares source→edit each step.

### Things to validate / likely-to-tune on real weights
1. **Flow sign & sigma scale** — we pass `sigma ∈ [0,1]` straight to `apply_model`
   and assume `x_next = x − v·dt`. If panoramas come out inverted/noisy, this is
   the first suspect.
2. **`(1−t)` blow-up** at `t→1` — clamped to `1e-3`; upstream relies on the
   flux-shifted schedule keeping `t<1`.
3. **Mask convention** — `invert_mask` (False=outpaint/keep-view, True=inpaint/
   regenerate-hole). Flip if regions are swapped.
4. **`model_management`** — we call `apply_model` directly after
   `load_models_gpu([model])`; confirm LoRA weight patches + attn patches are
   both live.
5. **2-batch conditioning** — source/edit T5 contexts are zero-padded to equal
   length in `flow.stack_conds`.
