import sys
import os

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import byzfl

class TestSignFlippingAttack:

    def setup_method(self):
        self.attack = byzfl.SignFlipping()
        self.expected_np = np.array([-4., -5., -6.])
        self.expected_torch = torch.tensor([-4., -5., -6.])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.attack(x)
        np.testing.assert_array_equal(result, self.expected_np)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
        result = self.attack(x)
        np.testing.assert_array_equal(result.numpy(), self.expected_torch.numpy())

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.attack(x)
        np.testing.assert_array_equal(result, self.expected_np)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.]),
             torch.tensor([4., 5., 6.]),
             torch.tensor([7., 8., 9.])]
        result = self.attack(x)
        np.testing.assert_array_equal(result.numpy(), self.expected_torch.numpy())


class TestInnerProductManipulationAttack:

    def setup_method(self):
        self.attack = byzfl.InnerProductManipulation(2.0)
        self.expected_np = np.array([-8., -10., -12.])
        self.expected_torch = torch.tensor([-8., -10., -12.])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.attack(x)
        np.testing.assert_array_equal(result, self.expected_np)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
        result = self.attack(x)
        np.testing.assert_array_equal(result.numpy(), self.expected_torch.numpy())

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.attack(x)
        np.testing.assert_array_equal(result, self.expected_np)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.]),
             torch.tensor([4., 5., 6.]),
             torch.tensor([7., 8., 9.])]
        result = self.attack(x)
        np.testing.assert_array_equal(result.numpy(), self.expected_torch.numpy())


class TestOptimalInnerProductManipulationAttack:

    def setup_method(self):
        agg = byzfl.TrMean(f=1)
        pre_agg_list = [byzfl.NNM(f=1), byzfl.Clipping()]
        self.attack = byzfl.Optimal_InnerProductManipulation(agg, pre_agg_list=pre_agg_list, f=1)

        self.expected_np = np.array([-2.74877907, -3.43597384, -4.1231686 ])
        self.expected_torch = torch.tensor([-2.748779 , -3.435974 , -4.1231685])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.attack(x)
        np.testing.assert_allclose(result, self.expected_np, rtol=1e-5, atol=1e-5)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
        result = self.attack(x)
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-4, atol=1e-4)

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.attack(x)
        np.testing.assert_allclose(result, self.expected_np, rtol=1e-5, atol=1e-5)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.]),
             torch.tensor([4., 5., 6.]),
             torch.tensor([7., 8., 9.])]
        result = self.attack(x)
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-4, atol=1e-4)


class TestALittleIsEnoughAttack:

    def setup_method(self):
        self.attack = byzfl.ALittleIsEnough(1.5)
        self.expected_np = np.array([8.5, 9.5, 10.5])
        self.expected_torch = torch.tensor([8.5, 9.5, 10.5])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.attack(x)
        np.testing.assert_allclose(result, self.expected_np, rtol=1e-5)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]], dtype=torch.float32)
        result = self.attack(x)
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-4)

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.attack(x)
        np.testing.assert_allclose(result, self.expected_np, rtol=1e-5)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.], dtype=torch.float32),
             torch.tensor([4., 5., 6.], dtype=torch.float32),
             torch.tensor([7., 8., 9.], dtype=torch.float32)]
        result = self.attack(x)
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-4)


class TestOptimalALittleIsEnoughAttack:

    def setup_method(self):
        agg = byzfl.TrMean(f=1)
        pre_agg_list = [byzfl.NNM(f=1), byzfl.Clipping()]
        self.attack = byzfl.Optimal_ALittleIsEnough(agg, pre_agg_list=pre_agg_list, f=1)

        self.expected_np = np.array([12.91985116, 13.91985116, 14.91985116])
        self.expected_torch = torch.tensor([12.919851, 13.919851, 14.919851])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.attack(x)
        np.testing.assert_allclose(result, self.expected_np, rtol=1e-5, atol=1e-5)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]], dtype=torch.float32)
        result = self.attack(x)
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-4, atol=1e-4)

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.attack(x)
        np.testing.assert_allclose(result, self.expected_np, rtol=1e-5, atol=1e-5)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.], dtype=torch.float32),
             torch.tensor([4., 5., 6.], dtype=torch.float32),
             torch.tensor([7., 8., 9.], dtype=torch.float32)]
        result = self.attack(x)
        np.testing.assert_allclose(result.numpy(), self.expected_torch.numpy(), rtol=1e-4, atol=1e-4)


class TestMimicAttack:

    def setup_method(self):
        self.attack = byzfl.Mimic(0)
        self.expected_np = np.array([1., 2., 3.])
        self.expected_torch = torch.tensor([1., 2., 3.])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.attack(x)
        np.testing.assert_array_equal(result, self.expected_np)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]])
        result = self.attack(x)
        np.testing.assert_array_equal(result.numpy(), self.expected_torch.numpy())

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.attack(x)
        np.testing.assert_array_equal(result, self.expected_np)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.]),
             torch.tensor([4., 5., 6.]),
             torch.tensor([7., 8., 9.])]
        result = self.attack(x)
        np.testing.assert_array_equal(result.numpy(), self.expected_torch.numpy())


class TestInfAttack:

    def setup_method(self):
        self.attack = byzfl.Inf()
        self.expected_np = np.array([np.inf, np.inf, np.inf])
        self.expected_torch = torch.tensor([float('inf'), float('inf'), float('inf')])

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.attack(x)
        np.testing.assert_array_equal(result, self.expected_np)

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]], dtype=torch.float32)
        result = self.attack(x)
        np.testing.assert_array_equal(result.numpy(), self.expected_torch.numpy())

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.attack(x)
        np.testing.assert_array_equal(result, self.expected_np)

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.], dtype=torch.float32),
             torch.tensor([4., 5., 6.], dtype=torch.float32),
             torch.tensor([7., 8., 9.], dtype=torch.float32)]
        result = self.attack(x)
        np.testing.assert_array_equal(result.numpy(), self.expected_torch.numpy())


class TestGaussianAttack:

    def setup_method(self):
        self.attack = byzfl.Gaussian(mu=0.0, sigma=1.0)
        self.shape = (3,)

    def test_numpy_array(self):
        x = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
        result = self.attack(x)
        assert result.shape == self.shape
        assert np.isfinite(result).all()

    def test_torch_tensor(self):
        x = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]], dtype=torch.float32)
        result = self.attack(x)
        assert result.shape == self.shape
        assert torch.isfinite(result).all()

    def test_list_of_numpy_arrays(self):
        x = [np.array([1., 2., 3.]),
             np.array([4., 5., 6.]),
             np.array([7., 8., 9.])]
        result = self.attack(x)
        assert result.shape == self.shape
        assert np.isfinite(result).all()

    def test_list_of_torch_tensors(self):
        x = [torch.tensor([1., 2., 3.], dtype=torch.float32),
             torch.tensor([4., 5., 6.], dtype=torch.float32),
             torch.tensor([7., 8., 9.], dtype=torch.float32)]
        result = self.attack(x)
        assert result.shape == self.shape
        assert torch.isfinite(result).all()