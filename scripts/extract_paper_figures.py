"""Extract selected original manuscript figures without redrawing their content.

PyMuPDF is an optional documentation dependency, separate from training.
The source PDF is supplied by the user and is not bundled with the repository.
"""

import argparse
import hashlib
import json
from pathlib import Path


FIGURES = (
    (2, 6, "docs/figures/figure-02-stage-i.jpg",
     "Architecture and training workflow of the Stage-I PINN-TL model."),
    (3, 8, "docs/figures/figure-03-inversion.jpg",
     "Architecture and trainable-parameter flow of the U-PINN model for damage inference."),
    (7, 12, "damage.png",
     "Damage evolution during the training process, with three snapshots "
     "shown for each case at different training iterations."),
    (8, 13, "result_1.png",
     "Optimization trajectories and convergence of the identified damage "
     "parameters under different damage cases."),
)


def sha256(content: bytes) -> str:
    """Return a deterministic digest for provenance checks."""
    return hashlib.sha256(content).hexdigest()


def extract(pdf_path: Path, output_root: Path) -> dict:
    """Extract original embedded pixels; preserve JPEG bytes where possible."""
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit("Install the optional dependency: pip install pymupdf") from exc

    pdf_bytes = pdf_path.read_bytes()
    records = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        # Validate every page before writing any assets.
        for number, page_number, _, _ in FIGURES:
            if page_number > len(document):
                raise ValueError("The supplied PDF is not the expected manuscript version.")
            page = document[page_number - 1]
            if f"Figure {number}." not in page.get_text():
                raise ValueError(f"Figure {number} is not on PDF page {page_number}.")
            if len(page.get_images(full=True)) != 1:
                raise ValueError(f"Expected one complete embedded figure on page {page_number}.")

        for number, page_number, relative_path, caption in FIGURES:
            image = document[page_number - 1].get_images(full=True)[0]
            xref, softmask = image[:2]
            original = document.extract_image(xref)
            pixmap = fitz.Pixmap(document, xref)
            target = output_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix == ".jpg":
                if original["ext"] != "jpeg" or softmask:
                    raise ValueError("Expected the manuscript's original JPEG image.")
                target.write_bytes(original["image"])
                conversion = "Original embedded JPEG bytes, unchanged."
            else:
                if softmask:
                    pixmap = fitz.Pixmap(pixmap, fitz.Pixmap(document, softmask))
                if pixmap.colorspace and pixmap.colorspace.n not in (1, 3):
                    pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                target.write_bytes(pixmap.tobytes("png"))
                conversion = "Embedded image decoded to lossless PNG; no resizing or redrawing."
            records.append({
                "figure": number, "pdf_page": page_number, "path": relative_path,
                "caption": caption, "xref": xref,
                "width_px": pixmap.width, "height_px": pixmap.height,
                "source_image_sha256": sha256(original["image"]),
                "asset_sha256": sha256(target.read_bytes()),
                "conversion": conversion,
                "evidence": "Paper-reported figure; not an output of a repository run.",
            })
    manifest = {
        "source": "PAPER.pdf supplied for this project",
        "manuscript_title": "U-PINN: Damage inference in thin-plate structures from "
                            "sparse UGW sensor data using a two-stage "
                            "physics-informed neural network",
        "source_pdf_sha256": sha256(pdf_bytes),
        "figures": records,
    }
    destination = output_root / "docs/figures/manifest.json"
    destination.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest


def main() -> None:
    """Regenerate assets from a local copy of the supplied manuscript."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    manifest = extract(args.pdf, args.output_root)
    for figure in manifest["figures"]:
        print(f"Figure {figure['figure']} (PDF p. {figure['pdf_page']}): "
              f"{figure['path']} [{figure['width_px']} x {figure['height_px']}]")


if __name__ == "__main__":
    main()
