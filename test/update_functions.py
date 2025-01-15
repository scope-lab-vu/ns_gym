
import unittest
from ns_gym.base import UpdateDistributionFn,UpdateFn
"""Test that update functions work as expected"""


class TestUpdateFn(unittest.TestCase):

    def setUp(self):
        return super().setUp()

    def test_update(self):
        raise NotImplementedError
    

def create_test_case(cls):
    class TestCase(TestUpdateFn):
        def setUp(self):
            super().setUp
            self.class_under_test = cls()
    return TestCase

for subclss in UpdateFn.__subclasses__():
    globals()[f"Test{subclss.__name__}"] = create_test_case(subclss)

if __name__ == "__main__":
    unittest.main()


    
