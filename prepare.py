"""Prepare an oversized ZIP for the 32 MiB per-file transfer limit.

Run beside RWB_train_for_Codex.zip:
    python prepare_codex_upload.py
Or pass the archive path as the first argument.
Only reads the original archive. Writes new ZIP parts plus Markdown/index files.
"""
from pathlib import Path
import hashlib
import json
import re
import sys
import time
import zipfile

CHUNK = 30 * 1024 * 1024
archive = Path(sys.argv[1] if len(sys.argv) > 1 else "RWB_train_for_Codex.zip")
if not archive.is_file():
    raise SystemExit(f"Файл не найден: {archive.resolve()}")

folder = archive.parent / "Codex_upload_parts"
if folder.exists():
    folder = archive.parent / ("Codex_upload_parts_" + time.strftime("%Y%m%d_%H%M%S"))
folder.mkdir(parents=True)
index = []
markdown = []
with zipfile.ZipFile(archive) as z:
    for i, entry in enumerate(z.infolist(), 1):
        index.append({"name": entry.filename, "bytes": entry.file_size,
                      "compressed_bytes": entry.compress_size})
        if not entry.is_dir() and entry.filename.lower().endswith(".md"):
            if entry.file_size > 5 * 1024 * 1024:
                continue
            basename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", entry.filename)
            target = folder / f"markdown_{i:04d}_{basename}"
            target.write_bytes(z.read(entry))
            markdown.append(target.name)

(folder / "archive_index.json").write_text(
    json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
parts = []
digest = hashlib.sha256()
with archive.open("rb") as source:
    number = 0
    while True:
        payload = source.read(CHUNK)
        if not payload:
            break
        number += 1
        digest.update(payload)
        filename = f"RWB_train.part{number:04d}.zip"
        target = folder / filename
        member = f"archive.part{number:04d}.bin"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr(member, payload)
        parts.append({"file": filename, "member": member, "bytes": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest()})
        print(f"Готово: {filename} ({target.stat().st_size / 1024**2:.1f} МиБ)", flush=True)

manifest = {"original_name": archive.name, "original_bytes": archive.stat().st_size,
            "original_sha256": digest.hexdigest(), "parts": parts,
            "markdown_files": markdown,
            "reassembly": "Read each listed ZIP member in manifest order and concatenate payloads."}
(folder / "parts_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nРезультат: {folder.resolve()}")
print("Прикрепи markdown-файлы, archive_index.json, parts_manifest.json и RWB_train.part*.zip.")
