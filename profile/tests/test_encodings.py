"""Objective-C type-encoding helpers used to decode query results."""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAVE_UNICORN = importlib.util.find_spec('unicorn') is not None


@unittest.skipUnless(HAVE_UNICORN, 'the target module imports the emulator (unicorn)')
class Encodings(unittest.TestCase):
    def setUp(self):
        from webmetal_profile import target
        self.t = target

    def test_return_type(self):
        rt = self.t.return_type
        self.assertEqual(rt('B16@0:8'), 'B')
        self.assertEqual(rt('{?=QQQ}16@0:8'), '{?=QQQ}')
        self.assertEqual(rt('{IndirectArgumentBufferCapabilities=b1b1b1b29}16@0:8'),
                         '{IndirectArgumentBufferCapabilities=b1b1b1b29}')
        self.assertEqual(rt('r^{?="a"I}16@0:8'), '^{?="a"I}')

    def test_argument_types(self):
        at = self.t.argument_types
        self.assertEqual(at('B16@0:8'), [])
        self.assertEqual(at('B24@0:8Q16'), ['Q'])
        self.assertEqual(at('{?=QQ}32@0:8Q16Q24'), ['Q', 'Q'])
        self.assertEqual(at('v32@0:8^{?=ff}16Q24'), ['^{?=ff}', 'Q'])

    def test_struct_size(self):
        size = self.t.struct_size
        self.assertEqual(size('{?=QQQ}'), 24)
        self.assertEqual(size('{?=II}'), 8)
        self.assertEqual(size('{X=b1b1b1b29}'), 4)

    def test_struct_fields(self):
        self.assertEqual(self.t.struct_fields('{?="a"I"b"f"c"Q}'), [('a', 'I'), ('b', 'f'), ('c', 'Q')])


if __name__ == '__main__':
    unittest.main()
