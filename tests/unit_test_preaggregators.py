import sys
import os

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import byzfl

class TestNNMAggregation:

    def setup_method(self):
        self.agg = byzfl.NNM(1)
        self.expected_np = np.array([[2.5, 3.5, 4.5],
                                     [2.5, 3.5, 4.5],
                                     [5.5, 6.5, 7.5]])
        self.expected_torch = torch.tensor([[2.5, 3.5, 4.5],
                                            [2.5, 3.5, 4.5],
                                            [5.5, 6.5, 7.5]])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_allclose(result, self.expected_np, rtol=1e-5)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-4)

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_allclose(result, self.expected_np, rtol=1e-5)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.]),
             torch.tensor([4., 5., 6.]),
             torch.tensor([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-4)

class TestClippingAggregation:

    def setup_method(self):
        self.agg = byzfl.Clipping(2.0)
        self.expected_np = np.array([
            [0.53452248, 1.06904497, 1.60356745],
            [0.91168461, 1.13960576, 1.36752692],
            [1.00514142, 1.14873305, 1.29232469]
        ])
        self.expected_torch = torch.tensor([
            [0.5345, 1.0690, 1.6036],
            [0.9117, 1.1396, 1.3675],
            [1.0051, 1.1487, 1.2923]
        ])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_allclose(result, self.expected_np, rtol=1e-5, atol=1e-5)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-3, atol=1e-3)

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_allclose(result, self.expected_np, rtol=1e-5, atol=1e-5)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.]),
             torch.tensor([4., 5., 6.]),
             torch.tensor([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-3, atol=1e-3)


class TestARCAggregation:

    def setup_method(self):
        self.agg = byzfl.ARC(1)
        self.expected_np = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [4.41004009, 5.04004582, 5.67005155]
        ])
        self.expected_torch = torch.tensor([
            [1.0000, 2.0000, 3.0000],
            [4.0000, 5.0000, 6.0000],
            [4.4100, 5.0400, 5.6701]
        ])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_allclose(result, self.expected_np, rtol=1e-5, atol=1e-5)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-3, atol=1e-3)

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_allclose(result, self.expected_np, rtol=1e-5, atol=1e-5)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.]),
             torch.tensor([4., 5., 6.]),
             torch.tensor([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-3, atol=1e-3)
