import unittest

from tests.dummy_classes.dummy import DummyA
from util.init_from_file import create_object, create_object_with_parameters

CODE = '/Users/nils/Studium/master-thesis/mmlib/tests/dummy_classes/dummy.py'


class TestInitFromFIle(unittest.TestCase):

    def test_init_no_args(self):
        obj = create_object(CODE, 'DummyC')

        self.assertEqual(42, obj.state)
        self.assertEqual(DummyA(42).state, obj.state2.state)

    def test_init_no_ref_args(self):
        init_args = {'state': 42}
        obj = create_object_with_parameters(CODE, 'DummyA', init_args=init_args)

        self.assertEqual(42, obj.state)

    def test_init_ref_args(self):
        init_args = {'arg1': 42, 'arg2': 43}
        ref_args = {'arg3': DummyA(42)}
        obj = create_object_with_parameters(CODE, 'DummyB', init_args=init_args, init_ref_type_args=ref_args)

        self.assertEqual(42, obj.state1)
        self.assertEqual(43, obj.state2)
        self.assertEqual(42, obj.state3.state)
