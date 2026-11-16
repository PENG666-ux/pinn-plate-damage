# Manuscript figure sources

These assets are extracted from the user-supplied 24-page manuscript, *U-PINN: Damage inference in thin-plate structures from sparse UGW sensor data using a two-stage physics-informed neural network*. Page numbers below refer to physical PDF pages.

They show the manuscript's method and reported experiments. They are not outputs from a new repository run. The original signals, simulation data, optimization histories, and trained weights remain unavailable; extracting an image does not establish an independent reproduction of its results.

| Repository asset | Manuscript source | Original dimensions | Content |
|---|---|---|---|
| [figure-02-stage-i.jpg](figures/figure-02-stage-i.jpg) | Figure 2, PDF p. 6 | 1975 × 1382 px | Stage-I simulation pretraining and healthy experimental transfer |
| [figure-03-inversion.jpg](figures/figure-03-inversion.jpg) | Figure 3, PDF p. 8 | 2013 × 933 px | Stage-II damage-inference architecture and trainable-parameter flow |
| [damage.png](../damage.png) | Figure 7, PDF p. 12 | 1931 × 1259 px | Cases I–III: damage evolution, three snapshots per case, and trajectories |
| [result_1.png](../result_1.png) | Figure 8, PDF p. 13 | 2020 × 1230 px | Cases I–III: position, radius, and gate optimization histories |

## Extraction and provenance

The workflow JPGs retain the exact embedded JPEG bytes. The two root PNG files decode the corresponding embedded JPEG pixels into lossless PNG while preserving the original dimensions. No plots, curve values, labels, legends, or colors have been redrawn or retouched. The manuscript's raster figures are retained as raster assets.

The [manifest](figures/manifest.json) records the source PDF SHA-256, source figure number, physical PDF page, PDF image object, image dimensions, original-image hash, and committed-asset hash. The PDF itself and local filesystem paths are not stored in this repository.

To rebuild these assets from your local copy of the same manuscript, install the optional documentation dependency and run from the repository root:

```bash
python -m pip install pymupdf
python scripts/extract_paper_figures.py --pdf /path/to/PAPER.pdf
```

The script checks the expected figure captions and page layout before extraction. For a revised manuscript, first verify the page mapping and update the extraction specification as needed. PyMuPDF is not required for model training or viewing the committed figures.

## Replacement map

- The previous `damage.png` was a five-panel perspective illustration, rather than the manuscript's three-case Figure 7. It has been replaced by the original Figure 7.
- The previous `result_1.png` was a “true damage / simulation / PINN prediction” montage that does not occur in the supplied manuscript. Its filename is retained, but its content and README caption now explicitly refer to Figure 8's parameter convergence. It is not presented as a replacement experiment or a three-panel prediction result.
- `src/plot.py` continues to create diagnostics from an actual run beneath `--output`. Those generated diagnostics are separate from the manuscript assets shown in the README.

Interpret the figures together with the [paper alignment notes](paper_alignment.md) and [method description](method.md), including the equivalent areal-density model and the limits of the attached-mass experiments.
