#!/usr/bin/env python3
"""Validate and index a merged Mesa archive before linking a Switch build."""

import argparse
import pathlib
import shutil
import subprocess


def validate(path):
    data = path.read_bytes()
    if data[:8] != b"!<arch>\n":
        raise ValueError("expected a regular merged archive, not a thin archive")
    offset = 8
    objects = 0
    while offset < len(data):
        header = data[offset:offset + 60]
        if len(header) != 60 or header[58:60] != b"`\n":
            raise ValueError("invalid archive member header")
        size = int(header[48:58])
        member = data[offset + 60:offset + 60 + size]
        if len(member) != size:
            raise ValueError("truncated archive member")
        name = header[:16].strip()
        if name not in (b"/", b"//", b"/SYM64/"):
            if member[:6] != b"\x7fELF\x02\x01" or len(member) < 20:
                raise ValueError(f"{name!r}: expected a 64-bit little-endian ELF object")
            if int.from_bytes(member[18:20], "little") != 183:
                raise ValueError(f"{name!r}: object was not compiled for AArch64")
            objects += 1
        offset += 60 + size + (size & 1)
    if not objects:
        raise ValueError("archive has no objects")
    return objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("destination", type=pathlib.Path)
    parser.add_argument("--ranlib", required=True,
                        help="devkitA64 aarch64-none-elf-ranlib executable")
    args = parser.parse_args()
    count = validate(args.source)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    if args.source.resolve() != args.destination.resolve():
        shutil.copyfile(args.source, args.destination)
    # The host GNU ar can omit Rust runtime symbols from its index when its
    # LLVM plugin cannot read the newer embedded bitcode. The target ranlib
    # indexes the native ELF symbols that the Switch linker actually needs.
    subprocess.run([args.ranlib, str(args.destination)], check=True)
    print(f"Staged {count} AArch64 objects: {args.destination}")


if __name__ == "__main__":
    main()
