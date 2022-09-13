# For some reason the JVM causes a segfault with OpenBLAS (numpy's linalg sovler). Need to halt multithreading before starting JVM:
import os, logging, sys

os.environ['OMP_NUM_THREADS'] = '1'

# formatter = logging.Formatter('[%(levelname)s: %(asctime)s]: %(message)s')

# ch = logging.StreamHandler()
# ch.setFormatter(formatter)
# ch.setLevel(logging.DEBUG)

# logger = logging.getLogger()
# logger.addHandler(ch)
# logger.setLevel(logging.INFO)

# logging.captureWarnings(True)
# logging.basicConfig(stream=, level=logging.INFO)