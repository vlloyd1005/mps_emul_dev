"""
fix_keras_compat.py

Workaround for loading .keras models saved with a newer Keras
(which adds fields like `quantization_config` to every layer's
serialized config) on an older Keras that doesn't recognize them.

Usage:
    python fix_keras_compat.py /path/to/model.keras

Produces /path/to/model_fixed.keras with the offending keys stripped
from config.json, recursively, at every nesting level.

You will still need to pass `custom_objects` (e.g. CustomActivationLayer)
when calling keras.models.load_model on the fixed file, exactly as you
would have for the original.
"""

import json
import os
import shutil
import sys
import tempfile
import zipfile

# Keys that newer Keras adds to layer configs that older Keras's
# layer __init__ signatures don't accept. Add more here if you hit
# similar "Unrecognized keyword arguments" errors for other keys.
KEYS_TO_STRIP = {"quantization_config"}


def strip_keys(obj, keys):
    """Recursively remove `keys` from every dict found in a nested
    JSON-like structure (dicts/lists), in place."""
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k in keys:
                del obj[k]
            else:
                strip_keys(obj[k], keys)
    elif isinstance(obj, list):
        for item in obj:
            strip_keys(item, keys)


def fix_keras_file(path, keys_to_strip=KEYS_TO_STRIP, out_path=None):
    if out_path is None:
        base, ext = os.path.splitext(path)
        out_path = f"{base}_fixed{ext}"

    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(path, "r") as zin:
            zin.extractall(tmpdir)

        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        strip_keys(config, keys_to_strip)

        with open(config_path, "w") as f:
            json.dump(config, f)

        # Repack as a .keras (zip) archive
        if os.path.exists(out_path):
            os.remove(out_path)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for root, _, files in os.walk(tmpdir):
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, tmpdir)
                    zout.write(full, rel)
    finally:
        shutil.rmtree(tmpdir)

    print(f"Wrote compat-fixed model to: {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_keras_compat.py /path/to/model.keras")
        sys.exit(1)
    fix_keras_file(sys.argv[1])