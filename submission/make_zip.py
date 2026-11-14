"""Builds ClaudeLaggye_code.zip: the files of this folder at the top level of the zip, plus the lc package beside them.
Model weights (models/), the environment (.venv/), caches and this script stay out.

python submission/make_zip.py [--out ClaudeLaggye_code.zip]
"""
import argparse
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = {"models", ".venv", ".python", "__pycache__", "make_zip.py"}


def entries():
    for src, prefix in ((HERE, ""), (os.path.join(os.path.dirname(HERE), "lc"), "lc/")):
        for d, dirs, names in os.walk(src):
            dirs[:] = sorted(x for x in dirs if x not in SKIP)
            for name in sorted(names):
                if name not in SKIP and not name.endswith((".pyc", ".zip", ".tmp", ".csv", ".jsonl", ".log")):
                    path = os.path.join(d, name)
                    yield path, prefix + os.path.relpath(path, src).replace(os.sep, "/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="ClaudeLaggye_code.zip")
    a = ap.parse_args()
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for path, arc in entries():
            info = zipfile.ZipInfo.from_file(path, arc)
            info.create_system = 3  # Unix: unzip applies the mode bits only then (on Windows the entry says MS-DOS)
            info.external_attr = (0o100755 if arc in ("setup.sh", "predict.py") else 0o100644) << 16  # Windows has no x bit
            with open(path, "rb") as f:
                z.writestr(info, f.read(), zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(a.out) as z:
        names = z.namelist()
    assert {"predict.py", "judge.py", "setup.sh", "requirements.txt", "team_info.json", "lc/combine.py"} <= set(names), names
    print(f"{a.out}: {len(names)} files, {os.path.getsize(a.out) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
