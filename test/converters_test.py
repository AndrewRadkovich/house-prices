import unittest
import numpy as np

from lib.converters import SalePriceConverter


class SalePriceConverterTest(unittest.TestCase):
    def test_convert(self):
        sale_price_converter = SalePriceConverter()
        array = np.array([1234., 2345., 3456.]).reshape(-1, 1)
        scaled_array = sale_price_converter.scale(array)
        inv_scaled = sale_price_converter.inv_scale(scaled_array)
        self.assertTrue((inv_scaled == array).all())


if __name__ == '__main__':
    unittest.main()
