# Key Results Table

| Result Element | Value / Observation | Evidence Link | Related Element |
|---|---|---|---|
| Baseline generated outputs | 22 frames recovered | [sample baseline frame](../04_evidence/sample_frames/baseline_generated_0000.jpg) | [artifact count graph](../02_graphs/artifact_counts.svg) |
| ControlNet outputs | 20 frames listed in metadata | [sample controlnet frame](../04_evidence/sample_frames/controlnet_generated_0000.jpg) | [generation profile](../02_graphs/generation_profile.svg) |
| ControlNet-fixed outputs | 22 frames listed in metadata | [sample fixed frame](../04_evidence/sample_frames/controlnet_fixed_generated_0000.jpg) | [artifact count graph](../02_graphs/artifact_counts.svg) |
| Structural conditioning evidence | Depth and refined mask snapshots available | [depth sample](../04_evidence/sample_frames/depth_0000.png), [mask sample](../04_evidence/sample_frames/mask_refined_0000.png) | [diagram explanations](../01_diagrams/diagram_explanations.md) |
| Generation setup consistency | `strength=0.6`, `num_steps=12`, `device=cpu` in recovered metadata | [generation profile graph](../02_graphs/generation_profile.svg) | [supporting tracking data](../05_supporting_docs/IEEE_PAPER_DATA_PROGRESS.md) |

## Interpretation

This table links qualitative outputs, structural maps, and configuration metadata so that each finding is inspectable across visual, tabular, and textual evidence.
