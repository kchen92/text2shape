"""Python wrapper for rendering voxels using stk.
"""

import os
import subprocess

from lib.config import cfg
from lib.data_io import get_voxel_file
import lib.nrrd_rw as nrrd_rw


def render_model_id(model_ids, out_dir, check=True):
    """Render models based on their model IDs.

    Args:
        model_ids: List of model ID strings.
        nrrd_dir: Directory to write the NRRD files to.
        out_dir: Directory to write the PNG files to.
        check: Check if the output directory already exists and provide a warning if so.
    """
    if cfg.CONST.DATASET == 'primitives':
        categories = [model_id.split('_')[0] for model_id in model_ids]
    elif cfg.CONST.DATASET == 'shapenet':
        categories = [None] * len(model_ids)
    else:
        raise ValueError('Please use a valid dataset.')

    if check is True:
        if os.path.isdir(out_dir):
            print('Output directory:', out_dir)
            input('Output render directory exists! Continue?')
        else:
            os.makedirs(out_dir)
    else:
        os.makedirs(out_dir, exist_ok=True)

    nrrd_files = []
    for category, model_id in zip(categories, model_ids):
        nrrd_files.append(get_voxel_file(category, model_id))

    txt_filepath = os.path.join(out_dir, 'nrrd_filenames.txt')
    with open(txt_filepath, 'w') as f:
        for outfile in nrrd_files:
            f.write(outfile + '\n')
    print('Filenames written to {}'.format(txt_filepath))

    render_nrrd(txt_filepath, out_dir, check=False)
    # else:
    #     voxel_tensors = []
    #     for model_id in model_ids:
    #         voxel_tensors.append(load_voxel(None, model_id))
    #     render_voxels(voxel_tensors, nrrd_dir, out_dir)


def write_nrrd(voxel_tensors, nrrd_dir):
    """Writes a list of voxel tensors to NRRDs in the directory specified by nrrd_dir.

    Args:
        voxel_tensors: A list of voxel tensors.
        nrrd_dir: Directory to write the NRRD files to.
    """
    if os.path.isdir(nrrd_dir):
        input('NRRD directory exists! Continue?')
    else:
        os.makedirs(nrrd_dir)
    num_renderings = len(voxel_tensors)
    outfiles = []
    for voxel_idx, voxel_tensor in enumerate(voxel_tensors):
        if (voxel_idx + 1) % 20 == 0:
            print('Rendering %d/%d.' % (voxel_idx + 1, num_renderings))

        outfile = 'voxel_tensor_%04d.nrrd' % voxel_idx
        nrrd_rw.write_nrrd(voxel_tensor, outfile)
        outfiles.append(outfile)

    txt_dir = nrrd_dir
    txt_filename = 'nrrd_filenames.txt'
    txt_filepath = os.path.join(txt_dir, txt_filename)
    with open(txt_filepath, 'w') as f:
        for outfile in outfiles:
            f.write(outfile + '\n')
    print('Filenames written to {}'.format(txt_filepath))
    return txt_filepath


def render_nrrd(nrrd,
                out_dir,
                turntable=False,
                turntable_step=10,
                compress_png=False,
                check=True):
    """Render NRRD files.

    Args:
        nrrd: An NRRD filename or txt file containing the NRRD filenames.
        out_dir: Output directory for the NRRD files.
        turntable: Whether or not to render a turntable.
        turntable_step: Number of degrees between each turntable step.
        compress_png: Whether to compress the png.
        check: Check if the output directory already exists and provide a warning if so.
    """
    if (check is True) and os.path.isdir(out_dir):
        input('Output render directory exists! Continue?')
    else:
        os.makedirs(out_dir, exist_ok=True)

    if turntable is True:
        turntable_str = '--render_turntable --turntable_step {}'.format(turntable_step)
    else:
        turntable_str = ''

    if compress_png is True:
        compress_png_str = '--compress_png'
    else:
        compress_png_str = ''

    render_command = [
        # 'node',
        # '--max-old-space-size=24000',
        '{}/ssc/render-voxels.js'.format(cfg.DIR.TOOLKIT_PATH),
        '--input',
        nrrd,
        '--output_dir',
        out_dir,
        turntable_str,
        compress_png_str,
    ]

    # subprocess.run is only supported on Python 3.5+
    # Otherwise, using subprocess.call
    subprocess.run(render_command, stdout=subprocess.PIPE)


def render_voxels(voxel_tensors, nrrd_dir, out_dir):
    """Render voxels by converting them to NRRD files first and then using stk to render them.

    Args:
        voxel_tensors: List of voxel tensors.

    Returns:
        outfiles: List of written files.
    """
    txt_filepath = write_nrrd(voxel_tensors, nrrd_dir)
    render_nrrd(txt_filepath, out_dir)
