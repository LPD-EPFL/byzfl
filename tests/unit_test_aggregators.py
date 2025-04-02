import sys
import os

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import byzfl

class TestAverageAggregation:

    def setup_method(self):
        self.agg = byzfl.Average()
        self.expected = np.array([4., 5., 6.])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_array_equal(result, self.expected)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_array_equal(result.numpy(), self.expected)

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_array_equal(result, self.expected)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.]),
             torch.tensor([4., 5., 6.]),
             torch.tensor([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_array_equal(result.numpy(), self.expected)


class TestMedianAggregation:

    def setup_method(self):
        self.agg = byzfl.Median()
        self.expected = np.array([4., 5., 6.])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_array_equal(result, self.expected)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_array_equal(result.numpy(), self.expected)

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_array_equal(result, self.expected)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.]),
             torch.tensor([4., 5., 6.]),
             torch.tensor([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_array_equal(result.numpy(), self.expected)


class TestTrMeanAggregation:

    def setup_method(self):
        self.agg = byzfl.TrMean(1)
        self.expected = np.array([4., 5., 6.])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_array_equal(result, self.expected)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_array_equal(result.numpy(), self.expected)

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_array_equal(result, self.expected)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.]),
             torch.tensor([4., 5., 6.]),
             torch.tensor([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_array_equal(result.numpy(), self.expected)


class TestGeometricMedianAggregation:

    def setup_method(self):
        self.agg = byzfl.GeometricMedian()
        self.expected_np = np.array([3.78788764, 4.78788764, 5.78788764])
        self.expected_torch = torch.tensor([3.7879, 4.7879, 5.7879])  # for display match

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
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-5)

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
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-5)


class TestKrumAggregation:

    def setup_method(self):
        self.agg = byzfl.Krum(1)
        self.expected_np = np.array([1., 2., 3.])
        self.expected_torch = torch.tensor([1., 2., 3.])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_array_equal(result, self.expected_np)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_array_equal(result.numpy(), self.expected_torch.numpy())

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_array_equal(result, self.expected_np)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.]),
             torch.tensor([4., 5., 6.]),
             torch.tensor([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_array_equal(result.numpy(), self.expected_torch.numpy())


class TestMultiKrumAggregation:

    def setup_method(self):
        self.agg = byzfl.MultiKrum(1)
        self.expected_np = np.array([2.5, 3.5, 4.5])
        self.expected_torch = torch.tensor([2.5, 3.5, 4.5])

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


class TestCenteredClippingAggregation:

    def setup_method(self):
        self.agg = byzfl.CenteredClipping()
        self.expected_np = np.array([4., 5., 6.])
        self.expected_torch = torch.tensor([4., 5., 6.])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_array_equal(result, self.expected_np)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
        result = self.agg(x)
        np.testing.assert_array_equal(result.numpy(), self.expected_torch.numpy())

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_array_equal(result, self.expected_np)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.]),
             torch.tensor([4., 5., 6.]),
             torch.tensor([7., 8., 9.])]
        result = self.agg(x)
        np.testing.assert_array_equal(result.numpy(), self.expected_torch.numpy())

class TestMDAAggregation:

    def setup_method(self):
        self.agg = byzfl.MDA(1)
        self.expected_np = np.array([2.5, 3.5, 4.5])
        self.expected_torch = torch.tensor([2.5, 3.5, 4.5])

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
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-5)

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
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-5)

class TestMoNNAAggregation:

    def setup_method(self):
        self.agg = byzfl.MoNNA(1, 1)
        self.expected_np = np.array([2.5, 3.5, 4.5])
        self.expected_torch = torch.tensor([2.5, 3.5, 4.5])

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
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-5)

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
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-5)


class TestMeamedAggregation:

    def setup_method(self):
        self.agg = byzfl.Meamed(1)
        self.expected_np = np.array([2.5, 3.5, 4.5])
        self.expected_torch = torch.tensor([2.5, 3.5, 4.5])

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
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-5)

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
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-5)
