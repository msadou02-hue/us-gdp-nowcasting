"""Crop the "good / bad / ugly" nowcast charts from page 18 of IMF WP/22/52.

Download the paper from https://www.imf.org/en/Publications/WP/Issues/2022/03/11/Nowcasting-GDP-A-Scalable-Approach-Using-DFM-Machine-Learning-and-Novel-Data-Applied-to-513957
and run:

    python scripts/extract_imf_figures.py path/to/wpiea2022052-print-pdf.pdf

Requires `pdf2image` and the Poppler utilities.
"""

import sys
from pathlib import Path

from pdf2image import convert_from_path


def main(pdf: str, out_dir: str = "docs/imf_figures", page: int = 18) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    image = convert_from_path(pdf, dpi=300, first_page=page, last_page=page)[0]
    w, h = image.size
    # Crop boxes found by visual inspection of the page layout.
    panels = {"the_good": (0.05, 0.33), "the_bad": (0.34, 0.63), "the_ugly": (0.64, 0.93)}
    for name, (top, bottom) in panels.items():
        image.crop((0, top * h, w, bottom * h)).save(out / f"gdp_predictions_{name}.png")
    print(f"Saved {len(panels)} figures to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
