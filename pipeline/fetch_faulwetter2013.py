"""Fetch micro-CT reconstructed slice stacks of polychaetes from Faulwetter et al. (2013).

Source
------
Faulwetter S, Vasileiadou A, Kouratoras M, Dailianis T, Arvanitidis C (2013)
"Micro-computed tomography: Introducing new dimensions to taxonomy."
ZooKeys 263: 1-45. https://doi.org/10.3897/zookeys.263.4261
Data: Dryad https://doi.org/10.5061/dryad.84m54 , mirrored on Zenodo record 4959912
(https://zenodo.org/records/4959912).

Licence: CC0 1.0 (as declared on the Zenodo/Dryad record; checked via the Zenodo API
field metadata.license.id at download time and stored in data/<name>/PROVENANCE.json).

Default specimens (small, whole animals, published size in ZooKeys Table 1):
  Eunice_sp_I2E.zip  (~46 MB)  juvenile Eunice sp., iodine-stained, 2.162 um/px, 1.8 mm
  Eunice_sp_PTA.zip  (~63 MB)  juvenile Eunice sp., PTA-stained,    2.298 um/px, 2.2 mm
  Eunice_roussaei.zip (~51 MB) dissected parapodium, HMDS dried,    2.878 um/px, 2.9 mm

Usage:  python -I pipeline/fetch_faulwetter2013.py ROOT [name ...]
Files land in ROOT/data/<name>/ (zip extracted, zip deleted). MD5 checked against Zenodo.
"""
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RECORD = "4959912"
API = f"https://zenodo.org/api/records/{RECORD}"
DEFAULT = ["Eunice_sp_I2E", "Eunice_sp_PTA", "Eunice_roussaei"]


def get(url, dest=None, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lifewatch-microct-test/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if dest is None:
                    return r.read()
                with open(dest, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                return dest
        except Exception as e:  # network retry
            print(f"  retry {i + 1}/{tries} after error: {e}", file=sys.stderr)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    names = sys.argv[2:] or DEFAULT
    rec = json.loads(get(API))
    lic = rec["metadata"].get("license", {}).get("id")
    files = {f["key"]: f for f in rec["files"]}
    for n in names:
        out = root / "data" / n
        if (out / "PROVENANCE.json").exists():
            print(f"{n}: already present")
            continue
        out.mkdir(parents=True, exist_ok=True)
        f = files[f"{n}.zip"]
        z = out / f"{n}.zip"
        print(f"{n}: downloading {f['size'] / 1e6:.0f} MB")
        get(f["links"]["self"], z)
        assert md5(z) == f["checksum"].split(":")[1], "MD5 mismatch"
        readme = files.get(f"README_for_{n}.txt")
        if readme:
            get(readme["links"]["self"], out / "README.txt")
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():  # guard against path traversal
                target = (out / "stack" / m.filename).resolve()
                assert str(target).startswith(str(out.resolve()))
            zf.extractall(out / "stack")
        z.unlink()
        (out / "PROVENANCE.json").write_text(json.dumps({
            "record": f"https://zenodo.org/records/{RECORD}", "doi": rec["metadata"].get("doi"),
            "file": f["key"], "size_bytes": f["size"], "checksum": f["checksum"], "license": lic,
            "paper": "https://doi.org/10.3897/zookeys.263.4261",
            "retrieved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
        print(f"{n}: done")


if __name__ == "__main__":
    main()
