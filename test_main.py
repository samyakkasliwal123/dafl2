import unittest
import torch
from models import ResCNNWithAuxiliaries

class TestResCNN(unittest.TestCase):
    def test_forward_pass(self):
        model = ResCNNWithAuxiliaries()
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

if __name__ == '__main__':
    unittest.main()