"""
This is a utility script that converts the old-style published LD matrices (magenpy 0.0.X) to the new
format deployed since magenpy>=0.1. The old LD matrix format used ragged Zarr arrays, while the new format
uses flattened Zarr arrays that are more efficient and easier to work with. The script takes the path to the
old LD matrices and converts them to the new format with the desired precision (e.g. float32).

The user may also specify the compressor name and compression level for the new LD matrices.
The script will validate the conversion by checking the integrity of the new LD matrices.

Usage:

    python convert_old_ld_matrices.py --old-matrix-path /path/to/old/ld_matrices/chr_* \
                                      --new-path /path/to/new/ld_matrices/ \
                                      --dtype float32

"""

import magenpy as mgp
from magenpy.utils.system_utils import makedir
import zarr
import os.path as osp
import glob
import argparse


parser = argparse.ArgumentParser(description="""
    Convert old-style LD matrices (magenpy 0.0.X) to the new format (magenpy >=0.1).
""")

parser.add_argument('--old-matrix-path', dest='old_path', type=str, required=True,
                    help='The path to the old LD matrix. Can be a wild card of the form "path/to/chr_*"')
parser.add_argument('--new-path', dest='new_path', type=str, required=True,
                    help='The path where to store the new LD matrix.')
parser.add_argument('--dtype', dest='dtype', type=str, default='int16',
                    choices={'int8', 'int16', 'float32', 'float64'},
                    help='The desired data type for the entries of the new LD matrix.')
parser.add_argument('--compressor', dest='compressor', type=str, default='zstd',
                    help='The compressor name for the new LD matrix.')
parser.add_argument('--compression-level', dest='compression_level', type=int, default=9,
                    help='The compression level for the new LD matrix.')

args = parser.parse_args()

for f in glob.glob(args.old_path):

    try:
        z_arr = zarr.open(f, 'r')
        chrom = z_arr.attrs['Chromosome']
    except Exception as e:
        print(f"Error: {e}")
        continue

    print(f"> Converting LD matrix for chromosome: {chrom}")

    new_path_suffix = f'chr_{chrom}'
    if new_path_suffix not in args.new_path:
        new_path = osp.join(args.new_path, new_path_suffix)
    else:
        new_path = args.new_path

    makedir(new_path)

    ld_mat = mgp.LDMatrix.from_ragged_zarr_matrix(f,
                                                  new_path,
                                                  overwrite=True,
                                                  dtype=args.dtype,
                                                  compressor_name=args.compressor,
                                                  compression_level=args.compression_level)
    print("Valid conversion:", ld_mat.validate_ld_matrix())
