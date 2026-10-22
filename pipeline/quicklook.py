"""Quick-look QC: mid slices and max-intensity projections of each stack. Usage: python -I quicklook.py ROOT"""
import sys, glob
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = Path(sys.argv[1])
for d in sorted((root / "data").iterdir()):
    fs = sorted(glob.glob(str(d / "stack" / "*.bmp")))
    v = np.stack([np.asarray(Image.open(f)) for f in fs])  # raw indices: some BMP palettes are inverted
    print(d.name, v.shape, v.dtype, np.percentile(v, [1, 50, 99, 99.9]))
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(v[len(v)//2], cmap="gray"); ax[0].set_title("mid z")
    ax[1].imshow(v.max(0), cmap="gray"); ax[1].set_title("MIP z")
    ax[2].imshow(v.max(1), cmap="gray", aspect="auto"); ax[2].set_title("MIP y")
    ax[3].hist(v.ravel()[::17], bins=256, log=True); ax[3].set_title("hist")
    fig.savefig(root / "outputs" / "qc" / f"{d.name}.png", dpi=70); plt.close(fig)
