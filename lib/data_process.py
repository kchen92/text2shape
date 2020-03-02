"""Parallel data loading functions.

Modified from: https://github.com/chrischoy/3D-R2N2/blob/master/lib/data_process.py
"""
import sys
import time
import numpy as np
import traceback
from six.moves import queue
from multiprocessing import Process, Event

from lib.config import cfg


def print_error(func):
    """Flush out error messages. Mainly used for debugging separate processes.
    """

    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            traceback.print_exception(*sys.exc_info())
            sys.stdout.flush()

    return func_wrapper


class DataProcess(Process):

    def __init__(self, data_queue, data_paths, batch_size=None, repeat=True):
        """Initialize a DataProcess.

        Args:
            data_queue: Multiprocessing queue.
            data_paths: List of data and label pairs used to load data.
            batch_size: Batch size.
            repeat: If set True, return data until exit is set.
        """
        super(DataProcess, self).__init__()
        if batch_size is None:
            batch_size = cfg.CONST.BATCH_SIZE

        # Queue to transfer the loaded mini batches
        self.data_queue = data_queue
        self.data_paths = data_paths
        self.num_data = len(data_paths)
        self.repeat = repeat

        # Tuple of data shape
        self.batch_size = batch_size
        self.exit = Event()
        self.shuffle_db_inds()

        # Only for external use and is only approximate - don't rely on this when testing
        # When testing, set repeat to False and rely on the data process to quit!
        self.iters_per_epoch = self.num_data // self.batch_size

    def shuffle_db_inds(self):
        # Randomly permute the training roidb
        if self.repeat:
            self.perm = np.random.permutation(np.arange(self.num_data))
        else:
            self.perm = np.random.permutation(np.arange(self.num_data))
            # self.perm = np.arange(self.num_data)
        self.cur = 0

    def get_next_minibatch(self):
        if (self.cur + self.batch_size) >= self.num_data and self.repeat:
            self.shuffle_db_inds()

        db_inds = self.perm[self.cur:min(self.cur + self.batch_size, self.num_data)]
        self.cur += self.batch_size
        return db_inds

    def shutdown(self):
        self.exit.set()

    @print_error
    def run(self):
        # Run the loop until exit flag is set
        while not self.exit.is_set() and self.cur < self.num_data:
            # Ensure that the network sees (almost) all data per epoch
            db_inds = self.get_next_minibatch()

            data_list = []
            label_list = []
            for db_ind in db_inds:
                datum = self.load_datum(self.data_paths[db_ind])
                label = self.load_label(self.data_paths[db_ind])

                data_list.append(datum)
                label_list.append(label)

            batch_data = np.array(data_list).astype(np.float32)
            batch_label = np.array(label_list).astype(np.float32)

            # The following will wait until the queue frees
            self.data_queue.put((batch_data, batch_label), block=True)

    def load_datum(self, path):
        pass

    def load_label(self, path):
        pass


def kill_processes(queue, processes):
    print('Signal processes to shut down.')
    for p in processes:
        p.shutdown()

    print('Emptying queue.')
    while not queue.empty():
        time.sleep(0.5)
        queue.get(False)

    print('Killing processes.')
    for p in processes:
        p.terminate()


def make_data_processes(data_process_class, queue, data_paths, num_workers, repeat):
    """Make a set of data processes for parallel data loading.
    """
    processes = []
    for i in range(num_workers):
        process = data_process_class(queue, data_paths, repeat=repeat)
        process.start()
        processes.append(process)
    return processes


def get_while_running(data_process, data_queue, sleep_time=0):
    while True:
        time.sleep(sleep_time)
        try:
            batch_data = data_queue.get_nowait()
        except queue.Empty:
            if not data_process.is_alive():
                break
            else:
                continue
        yield batch_data


if __name__ == '__main__':
    pass
