import os
import numpy as np
import lib.nrrd_rw as nrrd_rw


def batch_render_voxels(voxel_filenames, write_txt=False, voxel_processor=None):
    """Render a list of voxel tensors from npy files.

    Example: voxel_filenames = ['/tmp/voxel.npy']

    Args:
        model_list: List of model filenames (such as in .npy format).
        write_txt: Boolean for whether to write the generated NRRD filenames to a txt file. The file
                is written to the directory containing the FIRST file in voxel_filenames.
        voxel_processor: Function that postprocesses the loaded voxels

    Returns:
        outfiles: List of written files.
    """
    if len(voxel_filenames) > 9999:
        raise NotImplementedError('Cannot render %d images' % len(voxel_filenames))

    filename_ext = '.nrrd'
    voxels = voxel_filenames
    OVERWRITE = True
    num_renderings = len(voxels)
    outfiles = []
    for voxel_idx, voxel_f in enumerate(voxels):
        if (voxel_idx + 1) % 10 == 0:
            print('Rendering %d/%d.' % (voxel_idx + 1, num_renderings))

        outfile = os.path.splitext(voxel_f)[0] + filename_ext
        if not OVERWRITE and os.path.isfile(outfile):
            continue
        voxel_tensor = np.load(voxel_f)
        if voxel_processor is not None:
            voxel_tensor = voxel_processor(voxel_tensor)

        nrrd_rw.write_nrrd(voxel_tensor, outfile)
        outfiles.append(outfile)

    if write_txt is True:
        txt_dir = os.path.dirname(voxel_filenames[0])
        txt_filename = 'nrrd_filenames.txt'
        txt_filepath = os.path.join(txt_dir, txt_filename)
        with open(txt_filepath, 'w') as f:
            for outfile in outfiles:
                f.write(outfile + '\n')
        print('Filenames written to {}'.format(txt_filepath))
    return outfiles
