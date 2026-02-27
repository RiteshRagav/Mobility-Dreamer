# Evidence Crosswalk

This section maps representative sample images to the corresponding pipeline stage and interpretation context.

## Representative Samples

- Baseline generated frame: [sample_frames/baseline_generated_0000.jpg](sample_frames/baseline_generated_0000.jpg)
- ControlNet generated frame: [sample_frames/controlnet_generated_0000.jpg](sample_frames/controlnet_generated_0000.jpg)
- ControlNet fixed frame: [sample_frames/controlnet_fixed_generated_0000.jpg](sample_frames/controlnet_fixed_generated_0000.jpg)
- Depth map sample: [sample_frames/depth_0000.png](sample_frames/depth_0000.png)
- Refined mask sample: [sample_frames/mask_refined_0000.png](sample_frames/mask_refined_0000.png)

## How these complement each other

1. **Generated frames** show output appearance under different generation paths.
2. **Depth map** provides structural cue evidence for scene geometry handling.
3. **Refined mask** indicates segmentation/policy-region support information.
4. The combination is quantified in [../02_graphs/artifact_counts.svg](../02_graphs/artifact_counts.svg) and summarized in [../03_tables/key_results_table.md](../03_tables/key_results_table.md).

## Full snapshot folders

- [../results_snapshot/generated_frames](../results_snapshot/generated_frames)
- [../results_snapshot/generated_frames_controlnet](../results_snapshot/generated_frames_controlnet)
- [../results_snapshot/generated_frames_controlnet_fixed](../results_snapshot/generated_frames_controlnet_fixed)
- [../results_snapshot/depth_maps](../results_snapshot/depth_maps)
- [../results_snapshot/masks_refined](../results_snapshot/masks_refined)
