import argparse
import numpy as np
import os
import pprint
import yaml

# HACK: Get logger to print to stdout
import sys
sys.ps1 = '>>> '  # Make it "interactive"

import tensorflow as tf
from multiprocessing import Queue

from lib.config import cfg_from_file, cfg_from_list, cfg
from lib.data_process import make_data_processes, kill_processes
from lib.solver import Solver
from lib.solver_encoder import TextEncoderSolver, TextEncoderCosDistSolver, LBASolver
from lib.solver_gan import End2EndGANDebugSolver
from lib.solver_classifier import ClassifierSolver
from lib.cwgan import CWGAN
from lib.lba import LBA
from lib.classifier import Classifier
import lib.utils as utils
import models


del sys.ps1  # HACK: Get logger to print to stdout


def parse_args():
    """Parse the arguments.
    """
    parser = argparse.ArgumentParser(
        description='Main text2voxel train/test file.')
    parser.add_argument('--cfg',
                        dest='cfg_files',
                        action='append',
                        help='optional config file',
                        default=None,
                        type=str)
    parser.add_argument('--dont_save_voxels', dest='dont_save_voxels', action='store_true')
    parser.add_argument('--lba_only', dest='lba_only', action='store_true')
    parser.add_argument('--metric_learning_only', dest='metric_learning_only', action='store_true')
    parser.add_argument('--non_inverted_loss', dest='non_inverted_loss', action='store_true')
    parser.add_argument('--synth_embedding', dest='synth_embedding', action='store_true')
    parser.add_argument('--all_tuples', dest='all_tuples', action='store_true')
    parser.add_argument('--reed_classifier', dest='reed_classifier', action='store_true')
    parser.add_argument('--val_split',
                        dest='split',
                        help='data split for validation/testing (train, val, test)',
                        default=None,
                        type=str)
    parser.add_argument('--queue_capacity',
                        dest='queue_capacity',
                        help='size of queue',
                        default=None,
                        type=int)
    parser.add_argument('--n_minibatch_test',
                        dest='n_minibatch_test',
                        help='number of minibatches to use for test phase',
                        default=None,
                        type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset',
                        default=None,
                        type=str)
    parser.add_argument('--improved_wgan', dest='improved_wgan', action='store_true')
    parser.add_argument('--debug', dest='is_debug', action='store_true')
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--tiny_dataset', dest='tiny_dataset',
                        help='use a tiny dataset (~5 examples)',
                        action='store_true')
    parser.add_argument('--model',
                        dest='model',
                        help='name of the network model',
                        default=None,
                        type=str)
    parser.add_argument('--text_encoder', dest='text_encoder',
                        help='train/test on text encoder',
                        action='store_true')
    parser.add_argument('--classifier', dest='classifier',
                        help='train/test on classifier',
                        action='store_true')
    parser.add_argument('--end2end', dest='end2end',
                        help='train/test using end2end model such as End2EndLBACWGAN',
                        action='store_true')
    parser.add_argument('--shapenet_ct_classifier', dest='shapenet_ct_classifier',
                        help='chair/table classifier (sets up for classification)',
                        action='store_true')
    parser.add_argument('--noise_size',
                        dest='noise_size',
                        help='dimension of the noise',
                        default=None,
                        type=int)
    parser.add_argument('--noise_dist', dest='noise_dist',
                        help='noise distribution (uniform, gaussian)',
                        default=None,
                        type=str)
    parser.add_argument('--validation', dest='validation',
                        help='run validation while training',
                        action='store_true')
    parser.add_argument('--test', dest='test',
                        help='test mode',
                        action='store_true')
    parser.add_argument('--test_npy', dest='test_npy',
                        help='test mode using npy files',
                        action='store_true')
    parser.add_argument('--save_outputs', dest='save_outputs',
                        help='save the outputs to a file',
                        action='store_true')
    parser.add_argument('--summary_freq',
                        dest='summary_freq',
                        help='summary frequency',
                        default=None,
                        type=int)
    parser.add_argument('--optimizer',
                        dest='optimizer',
                        help='name of the optimizer',
                        default=None,
                        type=str)
    parser.add_argument('--critic_optimizer',
                        dest='critic_optimizer',
                        help='name of the critic optimizer',
                        default=None,
                        type=str)
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        help='batch size',
                        default=None,
                        type=int)
    parser.add_argument('--lba_mode',
                        dest='lba_mode',
                        help='LBA mode type (TST, STS, MM)',
                        default=None,
                        type=str)
    parser.add_argument('--lba_test_mode',
                        dest='lba_test_mode',
                        help='LBA test mode (shape, text) - what to input during forward pass',
                        default=None,
                        type=str)
    parser.add_argument('--visit_weight',
                        dest='visit_weight',
                        help='visit weight for lba models',
                        default=None,
                        type=float)
    parser.add_argument('--lba_unnormalize', dest='lba_unnormalize', action='store_true')
    parser.add_argument('--num_critic_steps',
                        dest='num_critic_steps',
                        help='number of critic steps per train step',
                        default=None,
                        type=int)
    parser.add_argument('--intense_training_freq',
                        dest='intense_training_freq',
                        help='frequency of intense critic training',
                        default=None,
                        type=int)
    parser.add_argument('--uniform_max',
                        dest='uniform_max',
                        help='absolute max for uniform distribution',
                        default=None,
                        type=float)
    parser.add_argument('--match_loss_coeff',
                        dest='match_loss_coeff',
                        help='coefficient for real match loss',
                        default=None,
                        type=float)
    parser.add_argument('--fake_match_loss_coeff',
                        dest='fake_match_loss_coeff',
                        help='coefficient for fake match loss',
                        default=None,
                        type=float)
    parser.add_argument('--fake_mismatch_loss_coeff',
                        dest='fake_mismatch_loss_coeff',
                        help='coefficient for fake mismatch loss',
                        default=None,
                        type=float)
    parser.add_argument('--gp_weight',
                        dest='gp_weight',
                        help='coefficient for gradient penalty',
                        default=None,
                        type=float)
    parser.add_argument('--text2text_weight',
                        dest='text2text_weight',
                        help='coefficient for text2text loss',
                        default=None,
                        type=float)
    parser.add_argument('--shape2shape_weight',
                        dest='shape2shape_weight',
                        help='coefficient for shape2shape loss',
                        default=None,
                        type=float)
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        help='learning rate',
                        default=None,
                        type=float)
    parser.add_argument('--critic_lr_multiplier',
                        dest='critic_lr_multiplier',
                        help='critic learning rate multiplier',
                        default=None,
                        type=float)
    parser.add_argument('--decay_steps',
                        dest='decay_steps',
                        help='decay steps',
                        default=None,
                        type=int)
    parser.add_argument('--num_epochs',
                        dest='num_epochs',
                        help='number of epochs',
                        default=None,
                        type=int)
    parser.add_argument('--augment_max',
                        dest='augment_max',
                        help='maximum augmentation perturbation out of 255',
                        default=None,
                        type=int)
    parser.add_argument('--set',
                        dest='set_cfgs',
                        help='set config keys',
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--ckpt_path', dest='ckpt_path',
                        help='Initialize network from checkpoint',
                        default=None)
    parser.add_argument('--lba_ckpt_path', dest='lba_ckpt_path',
                        help='Initialize LBA component of end2endlbawgan network from checkpoint',
                        default=None)
    parser.add_argument('--val_ckpt_path', dest='val_ckpt_path',
                        help='Initialize validation network from checkpoint',
                        default=None)
    parser.add_argument('--log_path', dest='log_path', help='set log path',
                        default=None)

    args = parser.parse_args()
    return args


def modify_args(args):
    """Modify the default config based on the command line arguments.
    """
    # modify default config if requested
    if args.cfg_files is not None:
        for cfg_file in args.cfg_files:
            cfg_from_file(cfg_file)
    randomize = args.randomize
    if args.test:  # Always randomize in test phase
        randomize = True
    if not randomize:
        np.random.seed(cfg.CONST.RNG_SEED)

    # NOTE: Unfortunately order matters here
    if args.lba_only is True:
        cfg_from_list(['LBA.COSINE_DIST', False])
    if args.metric_learning_only is True:
        cfg_from_list(['LBA.NO_LBA', True])
    if args.non_inverted_loss is True:
        cfg_from_list(['LBA.INVERTED_LOSS', False])
    if args.dataset is not None:
        cfg_from_list(['CONST.DATASET', args.dataset])
    if args.lba_mode is not None:
        cfg_from_list(['LBA.MODEL_TYPE', args.lba_mode])
    if args.lba_test_mode is not None:
        cfg_from_list(['LBA.TEST_MODE', args.lba_test_mode])
        # cfg_from_list(['LBA.N_CAPTIONS_PER_MODEL', 1])  # NOTE: Important!
    if args.shapenet_ct_classifier is True:
        cfg_from_list(['CONST.SHAPENET_CT_CLASSIFIER', args.shapenet_ct_classifier])
    if args.visit_weight is not None:
        cfg_from_list(['LBA.VISIT_WEIGHT', args.visit_weight])
    if args.lba_unnormalize is True:
        cfg_from_list(['LBA.NORMALIZE', False])
    if args.improved_wgan is True:
        cfg_from_list(['CONST.IMPROVED_WGAN', args.improved_wgan])
    if args.synth_embedding is True:
        cfg_from_list(['CONST.SYNTH_EMBEDDING', args.synth_embedding])
    if args.all_tuples is True:
        cfg_from_list(['CONST.TEST_ALL_TUPLES', args.all_tuples])
    if args.reed_classifier is True:
        cfg_from_list(['CONST.REED_CLASSIFIER', args.reed_classifier])
    if args.noise_dist is not None:
        cfg_from_list(['GAN.NOISE_DIST', args.noise_dist])
    if args.uniform_max is not None:
        cfg_from_list(['GAN.NOISE_UNIF_ABS_MAX', args.uniform_max])
    if args.num_critic_steps is not None:
        cfg_from_list(['WGAN.NUM_CRITIC_STEPS', args.num_critic_steps])
    if args.intense_training_freq is not None:
        cfg_from_list(['WGAN.INTENSE_TRAINING_FREQ', args.intense_training_freq])
    if args.match_loss_coeff is not None:
        cfg_from_list(['WGAN.MATCH_LOSS_COEFF', args.match_loss_coeff])
    if args.fake_match_loss_coeff is not None:
        cfg_from_list(['WGAN.FAKE_MATCH_LOSS_COEFF', args.fake_match_loss_coeff])
    if args.fake_mismatch_loss_coeff is not None:
        cfg_from_list(['WGAN.FAKE_MISMATCH_LOSS_COEFF', args.fake_mismatch_loss_coeff])
    if args.gp_weight is not None:
        cfg_from_list(['WGAN.GP_COEFF', args.gp_weight])
    if args.text2text_weight is not None:
        cfg_from_list(['WGAN.TEXT2TEXT_WEIGHT', args.text2text_weight])
    if args.shape2shape_weight is not None:
        cfg_from_list(['WGAN.SHAPE2SHAPE_WEIGHT', args.shape2shape_weight])
    if args.learning_rate is not None:
        cfg_from_list(['TRAIN.LEARNING_RATE', args.learning_rate])
    if args.critic_lr_multiplier is not None:
        cfg_from_list(['GAN.D_LEARNING_RATE_MULTIPLIER', args.critic_lr_multiplier])
    if args.decay_steps is not None:
        cfg_from_list(['TRAIN.DECAY_STEPS', args.decay_steps])
    if args.queue_capacity is not None:
        cfg_from_list(['CONST.QUEUE_CAPACITY', args.queue_capacity])
    if args.n_minibatch_test is not None:
        cfg_from_list(['CONST.N_MINIBATCH_TEST', args.n_minibatch_test])
    if args.noise_size is not None:
        cfg_from_list(['GAN.NOISE_SIZE', args.noise_size])
    if args.batch_size is not None:
        cfg_from_list(['CONST.BATCH_SIZE', args.batch_size])
    if args.summary_freq is not None:
        cfg_from_list(['TRAIN.SUMMARY_FREQ', args.summary_freq])
    if args.num_epochs is not None:
        cfg_from_list(['TRAIN.NUM_EPOCHS', args.num_epochs])
    if args.model is not None:
        cfg_from_list(['NETWORK', args.model])
    if args.optimizer is not None:
        cfg_from_list(['TRAIN.OPTIMIZER', args.optimizer])
    if args.critic_optimizer is not None:
        cfg_from_list(['GAN.D_OPTIMIZER', args.critic_optimizer])
    if args.ckpt_path is not None:
        cfg_from_list(['DIR.CKPT_PATH', args.ckpt_path])
    if args.lba_ckpt_path is not None:
        cfg_from_list(['END2END.LBA_CKPT_PATH', args.lba_ckpt_path])
    if args.val_ckpt_path is not None:
        cfg_from_list(['DIR.VAL_CKPT_PATH', args.val_ckpt_path])
    if args.log_path is not None:
        cfg_from_list(['DIR.LOG_PATH', args.log_path])
    if args.augment_max is not None:
        cfg_from_list(['TRAIN.AUGMENT_MAX', args.augment_max])
    if args.test:
        cfg_from_list(['TRAIN.AUGMENT_MAX', 0])
        cfg_from_list(['CONST.BATCH_SIZE', 1])
        cfg_from_list(['LBA.N_CAPTIONS_PER_MODEL', 1])  # NOTE: Important!
        cfg_from_list(['LBA.N_PRIMITIVE_SHAPES_PER_CATEGORY', 1])  # NOTE: Important!
    if args.test_npy:
        cfg_from_list(['CONST.BATCH_SIZE', 1])

    # To overwrite default variables, put the set_cfgs after all argument initializations
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)


def get_inputs_dict(args):
    """Gets the input dict for the current model and dataset.
    """
    if cfg.CONST.DATASET == 'shapenet':
        if (args.text_encoder is True) or (args.end2end is True) or (args.classifier is True):
            inputs_dict = utils.open_pickle(cfg.DIR.TRAIN_DATA_PATH)
            val_inputs_dict = utils.open_pickle(cfg.DIR.VAL_DATA_PATH)
            test_inputs_dict = utils.open_pickle(cfg.DIR.TEST_DATA_PATH)
        else:  # Learned embeddings
            inputs_dict = utils.open_pickle(cfg.DIR.SHAPENET_METRIC_EMBEDDINGS_TRAIN)
            val_inputs_dict = utils.open_pickle(cfg.DIR.SHAPENET_METRIC_EMBEDDINGS_VAL)
            test_inputs_dict = utils.open_pickle(cfg.DIR.SHAPENET_METRIC_EMBEDDINGS_TEST)
    elif cfg.CONST.DATASET == 'primitives':
        if ((cfg.CONST.SYNTH_EMBEDDING is True) or (args.text_encoder is True) or
                (args.classifier is True)):
            if args.classifier and not cfg.CONST.REED_CLASSIFIER:  # Train on all splits for classifier
                tf.compat.v1.logging.info('Using all (train/val/test) splits for training')
                inputs_dict = utils.open_pickle(cfg.DIR.PRIMITIVES_ALL_SPLITS_DATA_PATH)
            else:
                tf.compat.v1.logging.info('Using train split only for training')
                inputs_dict = utils.open_pickle(cfg.DIR.PRIMITIVES_TRAIN_DATA_PATH)
            val_inputs_dict = utils.open_pickle(cfg.DIR.PRIMITIVES_VAL_DATA_PATH)
            test_inputs_dict = utils.open_pickle(cfg.DIR.PRIMITIVES_TEST_DATA_PATH)
        else:  # Learned embeddings
            inputs_dict = utils.open_pickle(cfg.DIR.PRIMITIVES_METRIC_EMBEDDINGS_TRAIN)
            val_inputs_dict = utils.open_pickle(cfg.DIR.PRIMITIVES_METRIC_EMBEDDINGS_VAL)
            test_inputs_dict = utils.open_pickle(cfg.DIR.PRIMITIVES_METRIC_EMBEDDINGS_TEST)
    else:
        raise ValueError('Please use a valid dataset (shapenet, primitives).')

    if args.tiny_dataset is True:
        if ((cfg.CONST.DATASET == 'primitives' and cfg.CONST.SYNTH_EMBEDDING is True)
                or (args.text_encoder is True)):
            raise NotImplementedError('Tiny dataset not supported for synthetic embeddings.')

        ds = 5  # New dataset size
        if cfg.CONST.BATCH_SIZE > ds:
            raise ValueError('Please use a smaller batch size than {}.'.format(ds))
        inputs_dict = utils.change_dataset_size(inputs_dict, new_dataset_size=ds)
        val_inputs_dict = utils.change_dataset_size(val_inputs_dict, new_dataset_size=ds)
        test_inputs_dict = utils.change_dataset_size(test_inputs_dict, new_dataset_size=ds)

    # Select the validation/test split
    if args.split == 'train':
        split_str = 'train'
        val_inputs_dict = inputs_dict
    elif (args.split == 'val') or (args.split is None):
        split_str = 'val'
        val_inputs_dict = val_inputs_dict
    elif args.split == 'test':
        split_str = 'test'
        val_inputs_dict = test_inputs_dict
    else:
        raise ValueError('Please select a valid split (train, val, test).')
    print('Validation/testing on {} split.'.format(split_str))

    if (cfg.CONST.DATASET == 'shapenet') and (cfg.CONST.SHAPENET_CT_CLASSIFIER is True):
        category_model_list, class_labels = Classifier.set_up_classification(inputs_dict)
        val_category_model_list, val_class_labels = Classifier.set_up_classification(val_inputs_dict)
        assert class_labels == val_class_labels

        # Update inputs dicts
        inputs_dict['category_model_list'] = category_model_list
        inputs_dict['class_labels'] = class_labels
        val_inputs_dict['category_model_list'] = val_category_model_list
        val_inputs_dict['class_labels'] = val_class_labels

    return inputs_dict, val_inputs_dict


def get_solver(g, net, args, is_training):
    if isinstance(net, LBA):
        solver = LBASolver(net, g, is_training)
    elif args.text_encoder:
        solver = TextEncoderSolver(net, g, is_training)
    elif isinstance(net, Classifier):
        solver = ClassifierSolver(net, g, is_training)
    elif isinstance(net, CWGAN):
        solver = End2EndGANDebugSolver(net, g, is_training)
    else:
        raise ValueError('Invalid network.')
    return solver


def main():
    """Main text2voxel function.
    """
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.save_outputs is True and args.test is False:
        raise ValueError('Can only save outputs when testing, not training.')
    if args.validation:
        assert not args.test
    if args.test:
        assert args.ckpt_path is not None

    modify_args(args)

    print('----------------- CONFIG -------------------')
    pprint.pprint(cfg)

    # Save yaml
    os.makedirs(cfg.DIR.LOG_PATH, exist_ok=True)
    with open(os.path.join(cfg.DIR.LOG_PATH, 'run_cfg.yaml'), 'w') as out_yaml:
        yaml.dump(cfg, out_yaml, default_flow_style=False)

    # set up logger
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    try:
        with tf.Graph().as_default() as g:  # create graph
            # Load data
            inputs_dict, val_inputs_dict = get_inputs_dict(args)

            # Build network
            is_training = not args.test
            print('------------ BUILDING NETWORK -------------')
            network_class = models.load_model(cfg.NETWORK)
            net = network_class(inputs_dict, is_training)

            # Prefetching data processes
            #
            # Create worker and data queue for data processing. For training data, use
            # multiple processes to speed up the loading. For validation data, use 1
            # since the queue will be popped every TRAIN.NUM_VALIDATION_ITERATIONS.
            # set up data queue and start enqueue
            np.random.seed(123)
            data_process_class = models.get_data_process_pairs(cfg.NETWORK, is_training)
            val_data_process_class = models.get_data_process_pairs(cfg.NETWORK, is_training=False)
            if is_training:
                global train_queue, train_processes
                train_queue = Queue(cfg.CONST.QUEUE_CAPACITY)
                train_processes = make_data_processes(data_process_class, train_queue, inputs_dict,
                                                      cfg.CONST.NUM_WORKERS, repeat=True)
                if args.validation:
                    global val_queue, val_processes
                    val_queue = Queue(cfg.CONST.QUEUE_CAPACITY)
                    val_processes = make_data_processes(val_data_process_class, val_queue,
                                                        val_inputs_dict, 1, repeat=True)
            else:
                global test_queue, test_processes
                test_inputs_dict = val_inputs_dict
                test_queue = Queue(cfg.CONST.QUEUE_CAPACITY)
                test_processes = make_data_processes(val_data_process_class, test_queue,
                                                     test_inputs_dict, 1, repeat=False)

            # Create solver
            solver = get_solver(g, net, args, is_training)

            # Run solver
            if is_training:
                if args.validation:
                    if cfg.DIR.VAL_CKPT_PATH is not None:
                        assert train_processes[0].iters_per_epoch != 0
                        assert val_processes[0].iters_per_epoch != 0
                        solver.train(train_processes[0].iters_per_epoch, train_queue,
                                     val_processes[0].iters_per_epoch, val_queue=val_queue,
                                     val_inputs_dict=val_inputs_dict)
                    else:
                        if isinstance(net, LBA):
                            assert cfg.LBA.TEST_MODE is not None
                            assert cfg.LBA.TEST_MODE == 'shape'
                            assert train_processes[0].iters_per_epoch != 0
                            assert val_processes[0].iters_per_epoch != 0
                            solver.train(train_processes[0].iters_per_epoch, train_queue,
                                         val_processes[0].iters_per_epoch, val_queue=val_queue,
                                         val_inputs_dict=val_inputs_dict)
                        else:
                            assert train_processes[0].iters_per_epoch != 0
                            assert val_processes[0].iters_per_epoch != 0
                            solver.train(train_processes[0].iters_per_epoch, train_queue,
                                         val_processes[0].iters_per_epoch, val_queue=val_queue)
                else:
                    solver.train(train_processes[0].iters_per_epoch, train_queue)
            else:
                solver.test(test_processes[0], test_queue,
                            num_minibatches=cfg.CONST.N_MINIBATCH_TEST,
                            save_outputs=args.save_outputs)
    finally:
        # Clean up the processes and queues
        if is_training:
            kill_processes(train_queue, train_processes)
            if args.validation:
                kill_processes(val_queue, val_processes)
        else:
            kill_processes(test_queue, test_processes)


if __name__ == '__main__':
    main()
