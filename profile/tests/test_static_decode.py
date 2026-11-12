"""The B2 evaluator on instruction text in the form llvm-mc prints it."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webmetal_profile.static_decode import DecodeError, evaluate  # noqa: E402


class Evaluate(unittest.TestCase):
    def test_immediate(self):
        self.assertEqual(evaluate([('mov', 'w0, #0x4000'), ('ret', '')]), {'x0': 0x4000})

    def test_movk(self):
        out = evaluate([('mov', 'w2, #19'), ('movk', 'w2, #0x100, lsl #16'), ('mov', 'x0, x2'), ('ret', '')])
        self.assertEqual(out['x0'], 0x1000013)

    def test_negative_immediate_is_64_bit(self):
        self.assertEqual(evaluate([('mov', 'x0, #-0x1'), ('ret', '')])['x0'], (1 << 64) - 1)

    def test_zero_register(self):
        self.assertEqual(evaluate([('mov', 'x0, xzr'), ('ret', '')])['x0'], 0)

    def test_argument_passthrough(self):
        out = evaluate([('mov', 'x0, x2'), ('mov', 'w1, #0x100'), ('ret', '')])
        self.assertEqual(out, {'x0': ('arg', 0), 'x1': 0x100})

    def test_vector_struct_return(self):
        out = evaluate([('mov', 'w9, #0x200'), ('dup', 'v0.2d, x9'), ('str', 'q0, [x8]'),
                        ('str', 'x9, [x8, #16]'), ('ret', '')])
        self.assertEqual(out['x8_stores'], {0: (8, 0x200), 8: (8, 0x200), 16: (8, 0x200)})

    def test_tail_branch_carries_state(self):
        out = evaluate([('mov', 'w9, #3'), ('b', '#64')])
        self.assertEqual(out['branch'], 64)
        regs, _, _ = out['state']
        self.assertEqual(regs['x9'], 3)

    def test_sample_count_predicate(self):
        # sub x8, x2, #1; tst x2, x8; ccmp x2, #0, #4, eq; ccmp x2, #5, #2, ne; cset w0, lo
        body = [('sub', 'x8, x2, #1'), ('tst', 'x2, x8'), ('ccmp', 'x2, #0, #4, eq'),
                ('ccmp', 'x2, #5, #2, ne'), ('cset', 'w0, lo'), ('ret', '')]
        accepted = [n for n in range(0, 17) if evaluate(list(body), arguments=(n,))['x0']]
        self.assertEqual(accepted, [1, 2, 4])

    def test_csinc(self):
        body = [('cmp', 'x2, #4'), ('csinc', 'w0, wzr, w2, hs'), ('ret', '')]
        self.assertEqual(evaluate(list(body), arguments=(3,))['x0'], 4)
        self.assertEqual(evaluate(list(body), arguments=(9,))['x0'], 0)

    def test_signed_conditions(self):
        body = [('cmp', 'w2, #0'), ('cset', 'w0, lt'), ('ret', '')]
        self.assertEqual(evaluate(list(body), arguments=(0xffffffff,))['x0'], 1)
        self.assertEqual(evaluate(list(body), arguments=(1,))['x0'], 0)

    def test_compare_needs_a_value(self):
        with self.assertRaises(DecodeError):
            evaluate([('cmp', 'x2, #7'), ('cset', 'w0, lo'), ('ret', '')])

    def test_rejects_loads(self):
        with self.assertRaises(DecodeError):
            evaluate([('ldr', 'x0, [x0, #8]'), ('ret', '')])

    def test_requires_ret(self):
        with self.assertRaises(DecodeError):
            evaluate([('mov', 'w0, #1')])


if __name__ == '__main__':
    unittest.main()
