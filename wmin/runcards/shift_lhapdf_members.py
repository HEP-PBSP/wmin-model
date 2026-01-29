#!/usr/bin/env python3
"""
Script to shift the numeric suffix of files by -1 and remove any existing _0000.dat files.
Example:
  pod_basis_0001.dat -> pod_basis_0000.dat
  pod_basis_0002.dat -> pod_basis_0001.dat
  ...
If a file with index 0000 exists, it will be deleted before renaming to avoid collisions.
Also updates the NumMembers field in any .info files.
"""

import os
import re
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Shift file indices by -1 and remove index _0000.dat files."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Target directory (defaults to current directory)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Print the intended operations without executing them",
    )
    args = parser.parse_args()

    # Regex to match prefix and 4-digit index
    pattern = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{4})\.dat$")

    # List all files in the target directory
    try:
        all_files = os.listdir(args.directory)
    except OSError as e:
        print(f"Error reading directory '{args.directory}': {e}")
        return

    # Identify files to remove and to shift
    to_remove = []
    to_shift = []  # (filename, prefix, original_index)

    for fname in all_files:
        m = pattern.match(fname)
        if not m:
            continue
        idx = int(m.group("index"))
        prefix = m.group("prefix")

        if idx == 0:
            to_remove.append(fname)
        else:
            to_shift.append((fname, prefix, idx))

    # Remove any existing _0000.dat files
    for fname in to_remove:
        full_path = os.path.join(args.directory, fname)
        print(f"Removing: {full_path}")
        if not args.dry_run:
            os.remove(full_path)

    # First pass: rename originals to temporary names to avoid collisions
    temp_suffix = ".tmp_renaming"
    for fname, prefix, idx in to_shift:
        src = os.path.join(args.directory, fname)
        tmp = src + temp_suffix
        print(f"Renaming to temp: {src} -> {tmp}")
        if not args.dry_run:
            os.rename(src, tmp)

    # Second pass: rename temps to final shifted names
    for fname, prefix, idx in to_shift:
        tmp = os.path.join(args.directory, fname + temp_suffix)
        new_idx = idx - 1
        new_name = f"{prefix}_{new_idx:04d}.dat"
        dst = os.path.join(args.directory, new_name)
        print(f"Final rename: {tmp} -> {dst}")
        if not args.dry_run:
            # overwrite if exists
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(tmp, dst)

    # Update NumMembers in .info files
    info_files = [f for f in all_files if f.endswith(".info")]
    num_members = len(to_shift)  # number of shifted files = number of final members

    for info_file in info_files:
        info_path = os.path.join(args.directory, info_file)
        print(f"Updating NumMembers to {num_members} in: {info_path}")
        if not args.dry_run:
            with open(info_path, "r") as f:
                content = f.read()
            # Replace REPLACE_NREP or any existing NumMembers value
            content = re.sub(r"NumMembers:.*", f"NumMembers: {num_members}", content)
            with open(info_path, "w") as f:
                f.write(content)

    print("Done.")


if __name__ == "__main__":
    main()
