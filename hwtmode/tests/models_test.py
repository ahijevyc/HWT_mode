from hwtmode.models import BaseConvNet
import unittest
import numpy as np
import xarray as xr


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        self.x_shape = [32, 32, 3]
        self.train_size = 128
        self.x = np.random.normal(size=[self.train_size] + self.x_shape).astype(np.float32)
        self.x_ds = xr.DataArray(self.x, dims=("p", "row", "col", "var_name"),
                                 coords={"p": np.arange(self.train_size),
                                         "row": np.arange(self.x_shape[0]),
                                         "col": np.arange(self.x_shape[1]),
                                         "var_name": ["a", "b", "c"]})
        self.y = np.random.randint(0, 2, size=self.train_size)

    def test_network_build(self):
        bcn = BaseConvNet(min_filters=4, filter_growth_rate=1.5, min_data_width=8,
                          dense_neurons=4, output_type="sigmoid")
        bcn.build_network(self.x_shape, 1)
        assert bcn.model_.layers[1].output.shape[-1] == bcn.min_filters
        assert bcn.model_.layers[-6].output.shape[1] == bcn.min_data_width
        return

    def test_saliency(self):
        bcn = BaseConvNet(min_filters=4, filter_growth_rate=1.5, min_data_width=8,
                          dense_neurons=8, output_type="sigmoid")
        bcn.build_network(self.x_shape, 1)
        sal = bcn.saliency(self.x_ds)
        assert sal.max() > 0
        self.assertListEqual(list(sal.shape[1:]), list(self.x.shape))
