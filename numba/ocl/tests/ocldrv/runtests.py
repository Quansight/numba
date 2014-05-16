from __future__ import print_function, division, absolute_import
from numba.testing import discover_tests, run_tests


def test():
    suite = discover_tests("numba.ocl.tests.ocldrv")
    return run_tests(suite).wasSuccessful()


if __name__ == '__main__':
    test()