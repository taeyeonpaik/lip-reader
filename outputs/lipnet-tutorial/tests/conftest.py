import tensorflow as tf


def pytest_sessionstart(session):
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
