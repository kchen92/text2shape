from lib.config import cfg


def get_voxel_file(category, model_id):
    """Get the voxel absolute filepath for the model specified by category
    and model_id.

    Args:
        category: Category of the model as a string (eg. '03001627')
        model_id: Model ID of the model as a string
            (eg. '587ee5822bb56bd07b11ae648ea92233')

    Returns:
        voxel_filepath: Filepath of the binvox file corresonding to category and
            model_id.
    """
    if cfg.CONST.DATASET == 'shapenet':  # ShapeNet dataset
        return cfg.DIR.RGB_VOXEL_PATH % (model_id, model_id)
    elif cfg.CONST.DATASET == 'primitives':  # Primitives dataset
        return cfg.DIR.PRIMITIVES_RGB_VOXEL_PATH % (category, model_id)
    else:
        raise ValueError('Please choose a valid dataset (shapenet, primitives).')
