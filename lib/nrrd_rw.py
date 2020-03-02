"""Convert between voxel tensors and nrrd format.
"""
import nrrd
import numpy as np


def write_nrrd(voxel_tensor, filename):
    """Converts binvox tensor to NRRD (RGBA) format and writes it if a filename is provided.

    Example usage:
        voxel_tensor = np.load('/capri15/kchen92/Dev/text2voxel/output/tmp/ckpt-10500/0000_voxel_tensor_output.npy')
        _ = nrrd_rw.write_nrrd(voxel_tensor, filename='/tmp/test_nrrd.nrrd')

    Args:
        voxel_tensor: A tensor representing the binary voxels. Values can range from 0 to 1, and
            they will be properly scaled. The format is [height, width, depth, channels].
        filename: Filename that the NRRD will be written to.

    Writes:
        nrrd_tensor: An RGBA tensor where the channel dimension (RGBA) comes first
            (channels, height, width, depth).
    """
    if voxel_tensor.ndim == 3:  # Add a channel if there is no channel dimension
        voxel_tensor = voxel_tensor[np.newaxis, :]
    elif voxel_tensor.ndim == 4:  # Roll axes so order is (channel, height, width, depth) (not sure if (h, w, d))
        voxel_tensor = np.rollaxis(voxel_tensor, 3)
    else:
        raise ValueError('Voxel tensor must have 3 or 4 dimensions.')

    # Convert voxel_tensor to uint8
    voxel_tensor = (voxel_tensor * 255).astype(np.uint8)

    if voxel_tensor.shape[0] == 1:  # Add channels if voxel_tensor is a binvox tensor
        nrrd_tensor_slice = voxel_tensor
        nrrd_tensor = np.vstack([nrrd_tensor_slice] * 4)
        nrrd_tensor[:3, :, :, :] = 128  # Make voxels gray
        nrrd_tensor = nrrd_tensor.astype(np.uint8)
    elif voxel_tensor.shape[0] == 4:
        nrrd_tensor = voxel_tensor
    elif voxel_tensor.shape[0] != 4:
        raise ValueError('Voxel tensor must be single-channel or 4-channel')

    nrrd.write(filename, nrrd_tensor)


def read_nrrd(nrrd_filename):
    """Reads an NRRD file and returns an RGBA tensor.

    Args:
        nrrd_filename: Filename of the NRRD file.

    Returns:
        voxel_tensor: 4-dimensional voxel tensor with 4 channels (RGBA) where the alpha channel
                is the last channel (aka vx[:, :, :, 3]).
    """
    nrrd_tensor, options = nrrd.read(nrrd_filename)
    assert nrrd_tensor.ndim == 4

    # Convert to float [0, 1]
    voxel_tensor = nrrd_tensor.astype(np.float32) / 255.

    # Move channel dimension to last dimension
    voxel_tensor = np.rollaxis(voxel_tensor, 0, 4)

    # Make the model stand up straight by swapping axes (see README for more information)
    voxel_tensor = np.swapaxes(voxel_tensor, 0, 1)
    voxel_tensor = np.swapaxes(voxel_tensor, 0, 2)

    return voxel_tensor


def test_nrrd_rw():
    """Function for testing whether the read/write functions work correctly.
    """
    from lib.config import cfg
    from lib.data_io import get_voxel_file

    # Read an NRRD file, then write it and manually make sure it looks the same
    model_id = 'fcfb7012968416679c0b027ae5b223d6'
    nrrd_filepath = get_voxel_file(None, model_id)
    voxel_tensor = read_nrrd(nrrd_filepath)
    write_nrrd(voxel_tensor, '/tmp/test_nrrd_rw.nrrd')

    # Make a voxel tensor with red, green, blue, and black voxels (1 each) and manually verify it
    sparse_voxel_tensor = np.zeros((32, 32, 32, 4))
    sparse_voxel_tensor[0, 0, 0, :] = np.array([1, 0, 0, 1])  # Red
    sparse_voxel_tensor[31, 0, 0, :] = np.array([0, 1, 0, 1])  # Green
    sparse_voxel_tensor[0, 31, 0, :] = np.array([0, 0, 1, 1])  # Blue
    sparse_voxel_tensor[0, 0, 31, :] = np.array([0, 0, 0, 1])  # Black
    write_nrrd(sparse_voxel_tensor, '/tmp/test_nrrd_rw_2.nrrd')


if __name__ == '__main__':
    test_nrrd_rw()
