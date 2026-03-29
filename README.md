# Physics-Informed SYMBA — Pure Transformer

A pure Transformer encoder-decoder for generating squared particle-physics
scattering amplitudes |M|² from unsquared amplitudes M.

## Architecture
```
Interaction + Diagram Text  →  Encoder A  ─┐
                                             ├─►  Decoder (dual cross-attn)  →  |M|²
Amplitude Text              →  Encoder B  ─┘        ↑ Pointer-Generator
```

- **Dual Encoder** — Encoder A (diagram/interaction), Encoder B (amplitude)
- **Dual Cross-Attention** — decoder attends both encoders at every layer
- **Pointer-Generator** — copies tokens directly from amplitude input
- **Physics-Biased Attention** — Mandelstam/mass/coupling token families upweighted
- **Optional MoE** — Top-2 MoE FFN in top decoder layers
- **RoPE + RMSNorm + pre-norm** throughout

## Datasets

| Dataset | Process | Split |
|---------|---------|-------|
| QED | e⁺e⁻ tree-level 2→2 scattering | 80/10/10 |
| QCD | quark-gluon tree-level 2→2 scattering | 80/10/10 |

Data format (one sample per line):
```
INTERACTION : DIAGRAM+AMPLITUDE : SQUARED_AMPLITUDE
```

## Setup
```bash
pip install torch einops editdistance tqdm sympy
```

For Colab — Cell 1 of the notebook handles all installs automatically.

## Usage

1. Set data paths in **Cell 5** (`DATA_FILES` or `PKL_PATH`)
2. Configure model in **Cell 3** (`CFG` dict)
3. Run all cells sequentially
4. Training starts at **Cell 21**, evaluation at **Cell 22**

## Key Results (30 epochs)

| Dataset | Token Acc | Seq Acc | SymPy Valid |
|---------|-----------|---------|-------------|
| QED     | 99.5%     | 63.9%   | 97.2%       |
| QCD     | 97.0%     | 25.0%   | 83.3%       |

## Ablation Switches (Cell 3 — CFG)

| Flag | Default | Controls |
|------|---------|---------|
| `use_phys_bias` | `True` | Physics-biased attention |
| `use_pointer` | `True` | Pointer-generator copy mechanism |
| `use_moe` | `False` | MoE FFN in top decoder layers |
| `use_loss_coupling` | `True` | Coupling-power auxiliary loss |
| `use_grammar_mask` | `False` | Grammar-constrained decoding |

## Project Structure
```
notebook/
  Physics_Informed_SYMBA_PureTransformer.ipynb  # main notebook (28 cells)
checkpoints/          # saved model weights (git-ignored)
results/              # evaluation outputs (git-ignored)
data/                 # raw .txt and .pkl data files (git-ignored)
README.md
.gitignore
```

## Citation

If you use this work, please cite the SYMBA dataset and this repository.
