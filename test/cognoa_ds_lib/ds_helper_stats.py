'''
Unit tests for simple statistical methods found in cognoa_ds_lib.ds_helper_stats

WARNING: EXPECTED VALUES ARE SELECTED BASED ON RESULTS OF FUNCTIONS AS WRITTEN
         NOT A REAL TEST!
'''

import unittest

from cognoa_ds_lib.ds_helper_stats import *


PRECISION = 10


class TestAll(unittest.TestCase):


    X = ['a', 'a', 'b', 'b', 'b', 'b']
    Y = ['0', '0', '1', '1', '1','0' ]


    def test_entropy(self):
        expected = round(1.0, PRECISION)
        res = round(entropy(self.Y), PRECISION)

        self.assertEqual(res, expected)


    def test_cond_entropy(self):
        expected = round(0.540852082973, PRECISION)
        res = round(conditionalEntropy(self.Y, self.X), PRECISION)

        self.assertEqual(res, expected)


    def test_info_gain(self):
        expected = round(0.459147917027, PRECISION)
        res = round(informationGain(self.Y, self.X), PRECISION)

        self.assertEqual(res, expected)


if __name__ == '__main__':
    unittest.main()
