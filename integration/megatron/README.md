# Megatron-LM integration

This directory holds the *exact* offline-growth scripts we used to grow
our 17B base model into the 35B (depth) and 70B (depth+width) MoE
checkpoints reported in the paper. They are **not** a self-contained
training stack — they expect a Megatron-LM tree that exposes the
`megatron.training.*` and `megatron.core.*` APIs.

For a zero-dependency, reading-friendly version of the *algorithm*, see
[`../../growth/`](../../growth/). For end-to-end training infrastructure,
see Microsoft's public Megatron fork:
**[microsoft/ltp-megatron-lm](https://github.com/microsoft/ltp-megatron-lm)**.

## Files

| File | Notes |
|------|-------|
| [`Offline_checkpoint_growth.py`](Offline_checkpoint_growth.py) | Loads one (or two) base checkpoint(s), expands the layer / expert state dicts in place, and re-saves under the dist-ckpt format. |
| [`Launch_offline_checkpoint_growth.sh`](Launch_offline_checkpoint_growth.sh) | Wrapper that sources the model + training configs and feeds the right flags to `torchrun`. |

## Three-step recipe

1. **Clone the public Megatron fork** and make its source importable:

   ```bash
   git clone https://github.com/microsoft/ltp-megatron-lm.git
   export MEGATRON_PATH="$PWD/ltp-megatron-lm"
   ```

2. **Drop the two files in this directory at the Megatron root** (or
   anywhere on `PYTHONPATH` that can see `pretrain_gpt`):

   ```bash
   cp Offline_checkpoint_growth.py Launch_offline_checkpoint_growth.sh "$MEGATRON_PATH/"
   ```

3. **Provide your own model + training configs** at
   `$MODEL_CONFIG_PATH/{model_config.sh,training_config.sh}` (the launch
   script sources them; they must define the `MODEL_ARGS`, `MOE_ARGS`,
   `TRAINING_ARGS`, `IMPL_ARGS` arrays the script splices into the
   `torchrun` command).

4. **Run the growth pass on a saved base checkpoint** (CPU-only,
   `--use-cpu-initialization` and `--no-load-optim` are already set):

   ```bash
   export PROJECT_PATH=/path/to/your/project
   export MODEL_NAME=my-17b-moe
   export OUTPUT_DIR=/path/to/runs

   # Depth growth (interposition, k=2)
   USE_DEPTH_GROWTH=True \
       bash Launch_offline_checkpoint_growth.sh <iteration>

   # Width growth (E -> 2E, with 1% symmetry-breaking noise)
   USE_MOE_WIDTH_GROWTH=True GROWTH_ADD_EXPERT_NOISE=True \
       GROWTH_EXPERT_NOISE_STD_SCALING_FACTOR=0.01 \
       bash Launch_offline_checkpoint_growth.sh <iteration>
   ```

   The resulting checkpoint is written to
   `${OUTPUT_DIR}/checkpoints/growth_model_<iteration>` and can be
   resumed by the standard Megatron training entrypoint as the new
   `--load` target. From there continue pre-training with your usual
   training launcher.

## Caveats

* **Data and the full pre-training pipeline are not in this repo.** The
  paper's runs used internal corpora and tooling that cannot be released
  under our company's compliance review. Treat this repo as an
  *unofficial reproduction guide*: the growth operator is open-sourced
  here, but you bring your own base checkpoint and data.
* The scripts target the dist-ckpt format (`--ckpt-format torch_dist`
  with `--ckpt-convert-format torch_dist`). Other formats are not
  exercised.
* Only `width_factor=2` and `growth_factor=2` are covered by the paper
  and these scripts.
* `--use-ckpt-merge` (two-checkpoint averaging at growth time) is
  implemented but was used only in exploratory ablations; it is not the
  main result.

## Key flags (`add_growth_args` in `Offline_checkpoint_growth.py`)

| Flag | Meaning |
|------|---------|
| `--do-depth-growth` | Run [`model_depth_growth`](Offline_checkpoint_growth.py#L221) — duplicate layers. |
| `--do-moe-width-growth` | Run [`model_moe_width_growth`](Offline_checkpoint_growth.py#L379) — duplicate experts and double the router. |
| `--growth-stack-method {interleaved,stacked}` | Paper terminology: `interleaved` = "interposition" (Eq. 2), `stacked` = "stack" (Eq. 1). |
| `--growth-ignore-first-num-layers N` / `--growth-ignore-last-num-layers N` | Optional: leave the first / last `N` source layers un-duplicated. Not extensively discussed in the paper for space reasons; the code defaults to `0` and `N=2` is a reasonable starting point if you want to preserve embeddings-/head-adjacent layers. |
| `--growth-add-expert-noise` | Add the small Gaussian perturbation to copied experts. |
| `--growth-expert-noise-std-scaling-factor` | The `α` in `N(0, (α · σ_orig)²)`. Paper default: `0.01`. |
| `--growth-zerofy-output-init` | Zero the output projections of newly copied depth layers (identity-like residual at growth time). |
| `--growth-use-interleaved-moe-cat` | Interleave new experts with originals along the expert dim instead of appending. |
| `--use-ckpt-merge`, `--second-ckpt-step` | Optionally load a second checkpoint and use it as the source for the duplicated experts / layers (model-merging variant). |

## Cite

If you build on this code, please cite the paper — see the [root
README](../../README.md#citation) for the BibTeX entry.
