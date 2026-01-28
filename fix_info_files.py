#!/usr/bin/env python
"""
Fix malformed COCO .info files.

The issue: When batch folders were merged, data lines for some dimensions
were appended without their required header lines.

This script reads each .info file and adds missing headers for orphaned data lines.
"""

import re
from pathlib import Path


def fix_info_file(info_path: Path) -> bool:
    """
    Fix a single .info file by adding missing headers.

    Returns True if file was modified.
    """
    with open(info_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        return False

    # Extract function ID from filename
    match = re.search(r'bbobexp_f(\d+)\.info', info_path.name)
    if not match:
        print(f"  Skipping {info_path.name} - cannot parse function ID")
        return False
    func_id = int(match.group(1))

    # Parse the file and find orphaned data lines
    fixed_lines = []
    last_header = None
    last_dim = None
    modified = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Check if this is a header line
        if line.startswith("suite = 'bbob'"):
            last_header = line
            # Extract dimension from header
            dim_match = re.search(r'DIM = (\d+)', line)
            if dim_match:
                last_dim = int(dim_match.group(1))
            fixed_lines.append(line + '\n')
            i += 1
            continue

        # Check if this is a comment line
        if line.startswith('%'):
            fixed_lines.append(line + '\n')
            i += 1
            continue

        # Check if this is a data line
        data_match = re.match(r'(data_f\d+/bbobexp_f\d+_DIM(\d+)\.dat),', line)
        if data_match:
            data_dim = int(data_match.group(2))

            # Check if this data line has a matching header
            if last_dim != data_dim:
                # Missing header! Add one
                print(f"  Adding missing header for DIM={data_dim}")

                # Create header based on template from file or default
                if last_header:
                    new_header = re.sub(r'DIM = \d+', f'DIM = {data_dim}', last_header)
                else:
                    new_header = f"suite = 'bbob', funcId = {func_id}, DIM = {data_dim}, Precision = 1.000e-08, algId = 'NEVO', coco_version = '2.8.2', logger = 'bbob', data_format = 'bbob-new2', settings = ''"

                fixed_lines.append(new_header + '\n')
                fixed_lines.append('% NEVO neuromorphic optimiser\n')
                last_dim = data_dim
                modified = True

            fixed_lines.append(line + '\n')
            i += 1
            continue

        # Empty or other line
        if line.strip():
            fixed_lines.append(line + '\n')
        i += 1

    if modified:
        # Write fixed file
        with open(info_path, 'w') as f:
            f.writelines(fixed_lines)

    return modified


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix malformed COCO .info files")
    parser.add_argument(
        "exdata_dir",
        type=str,
        nargs="?",
        default="exdata/NEVO",
        help="Path to exdata directory containing .info files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without modifying files"
    )

    args = parser.parse_args()

    exdata_path = Path(args.exdata_dir)
    if not exdata_path.exists():
        print(f"Error: Directory '{exdata_path}' does not exist")
        return 1

    info_files = sorted(exdata_path.glob("bbobexp_f*.info"))
    if not info_files:
        print(f"No .info files found in {exdata_path}")
        return 1

    print(f"Checking {len(info_files)} .info files in {exdata_path}")
    print("=" * 60)

    fixed_count = 0
    for info_file in info_files:
        print(f"\nProcessing {info_file.name}...")

        if args.dry_run:
            # Just check without modifying
            with open(info_file, 'r') as f:
                content = f.read()

            # Count data lines and headers
            data_lines = re.findall(r'data_f\d+/bbobexp_f\d+_DIM(\d+)\.dat', content)
            header_dims = re.findall(r"suite = 'bbob'.*?DIM = (\d+)", content)

            data_dims = set(data_lines)
            header_dims_set = set(header_dims)

            missing = data_dims - header_dims_set
            if missing:
                print(f"  Would add headers for dimensions: {sorted(int(d) for d in missing)}")
                fixed_count += 1
            else:
                print(f"  OK - all {len(data_dims)} dimensions have headers")
        else:
            if fix_info_file(info_file):
                fixed_count += 1
                print(f"  Fixed!")
            else:
                print(f"  OK - no changes needed")

    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"Would fix {fixed_count} files (dry run)")
    else:
        print(f"Fixed {fixed_count} files")

    return 0


if __name__ == "__main__":
    exit(main())
