"""Render voxel tensors given a directory of npy files.
"""
from lib.preprocess import load_voxel
from lib.visualization.voxel_rendering import batch_render_voxels

import argparse
import glob
import os


os.environ['QT_API'] = 'pyqt'


def batch_render(voxel_dir, voxel_processor=None):
    """Render a bunch of voxel tensors stored as npy files in the voxel_dir.

    Args:
        voxel_dir: A directory containing voxel tensors stored in npy files.
        voxel_processor: Function that processes the voxels before rendering
    """
    npy_files = glob.glob(os.path.join(voxel_dir, '*.npy'))
    batch_render_voxels(npy_files, write_txt=True, voxel_processor=voxel_processor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('voxel_dir', help='Directory of saved npy files (and directory to save to)')

    args = parser.parse_args()
    postprocessor = None
    batch_render(args.voxel_dir, postprocessor)


if __name__ == '__main__':
    main()
