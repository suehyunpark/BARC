import unittest
from typing import List
from gen_inputs_induction_correction import (
    get_correct_indices,
    find_maximal_groups,
    is_strictly_better
)

class TestInductionOrdering(unittest.TestCase):
    def test_get_correct_indices(self):
        verdicts = [True, False, True, None]
        self.assertEqual(get_correct_indices(verdicts), {0, 2})
        
        verdicts = [False, False, False]
        self.assertEqual(get_correct_indices(verdicts), set())
        
        verdicts = [True, True, True]
        self.assertEqual(get_correct_indices(verdicts), {0, 1, 2})

    def test_is_strictly_better(self):
        # Test cases from your example
        h0 = [None, True, None]
        h1 = [False, True, False]   # {1}
        h2 = [False, True, True]    # {1, 2}
        h3 = [True, True, False]    # {0, 1}
        h4 = [True, True, True]     # {0, 1, 2}
        h5 = [True, True, False]    # {0, 1}
        h6 = [None, False, True]    # {2}
        
        # Direct comparisons
        self.assertTrue(is_strictly_better(h0, h1))
        self.assertTrue(is_strictly_better(h1, h2))   # {1} < {1, 2}
        self.assertTrue(is_strictly_better(h1, h3))   # {1} < {0, 1}
        self.assertTrue(is_strictly_better(h1, h4))   # {1} < {0, 1, 2}
        self.assertTrue(is_strictly_better(h2, h4))   # {1, 2} < {0, 1, 2}
        self.assertTrue(is_strictly_better(h3, h4))   # {0, 1} < {0, 1, 2}
        self.assertTrue(is_strictly_better(h6, h1))   # {2} < {1}
        
        # Non-comparable pairs
        self.assertFalse(is_strictly_better(h2, h3))  # {1, 2} ≮ {0, 1}
        self.assertFalse(is_strictly_better(h3, h2))  # {0, 1} ≮ {1, 2}
        self.assertFalse(is_strictly_better(h2, h5))  # {1, 2} ≮ {0, 1}
        
        # Reflexivity
        self.assertFalse(is_strictly_better(h1, h1))  # {1} ≮ {1}
        
        # Different lengths
        self.assertFalse(is_strictly_better([True, False], [True, False, True]))

    def test_find_maximal_groups(self):
        # Test case from your example
        verdicts = [
            [False, True, False],    # h1: {1}
            [False, True, True],     # h2: {1, 2}
            [True, True, False],     # h3: {0, 1}
            [True, True, True]       # h4: {0, 1, 2}
        ]
        
        groups = find_maximal_groups(verdicts)
        
        # Expected groups (using indices):
        # {0, 1, 3} representing {h1, h2, h4}
        # {0, 2, 3} representing {h1, h3, h4}
        
        expected_groups = [[0, 1, 3], [0, 2, 3]]
        
        # Sort both lists for comparison
        groups = sorted([sorted(g) for g in groups])
        expected_groups = sorted([sorted(g) for g in expected_groups])
        
        self.assertEqual(groups, expected_groups)
        
    def test_edge_cases(self):
        # Empty case
        self.assertEqual(find_maximal_groups([]), [])
        
        # Single hypothesis
        self.assertEqual(find_maximal_groups([[True]]), [])
        
        # All identical
        verdicts = [[True, False], [True, False], [True, False]]
        self.assertEqual(find_maximal_groups(verdicts), [])
        
        # No valid groups (no subset relationships)
        verdicts = [
            [True, False, False],
            [False, True, False],
            [False, False, True]
        ]
        self.assertEqual(find_maximal_groups(verdicts), [])

    def test_complex_case(self):
        verdicts = [
            [True, False, False, False],  # h1: {0}
            [True, True, False, False],   # h2: {0, 1}
            [True, False, True, False],   # h3: {0, 2}
            [True, True, True, False],    # h4: {0, 1, 2}
            [True, True, True, True]      # h5: {0, 1, 2, 3}
        ]
        
        groups = find_maximal_groups(verdicts)
        expected_groups = [
            [0, 1, 3, 4],  # {h1, h2, h4, h5}
            [0, 2, 3, 4],  # {h1, h3, h4, h5}
        ]
        
        groups = sorted([sorted(g) for g in groups])
        expected_groups = sorted([sorted(g) for g in expected_groups])
        
        self.assertEqual(groups, expected_groups)

    def test_is_strictly_better_with_none(self):
        # Basic None vs False comparisons
        h1 = [None, None, None]
        h2 = [False, None, None]
        h3 = [False, False, None]
        h4 = [False, False, False]
        self.assertTrue(is_strictly_better(h1, h2))   # More False is better than None
        self.assertTrue(is_strictly_better(h2, h3))   # More False is better
        self.assertTrue(is_strictly_better(h3, h4))   # More False is better
        self.assertTrue(is_strictly_better(h1, h4))   # Transitive
        
        # None vs False vs True progression
        h5 = [None, None, True]
        h6 = [False, None, True]
        h7 = [False, False, True]
        h8 = [True, False, True]
        h9 = [True, True, True]
        self.assertTrue(is_strictly_better(h5, h6))   # None -> False improvement
        self.assertTrue(is_strictly_better(h6, h7))   # None -> False improvement
        self.assertTrue(is_strictly_better(h7, h8))   # False -> True improvement
        self.assertTrue(is_strictly_better(h8, h9))   # False -> True improvement
        
        # Mixed comparisons
        h10 = [None, True, None]
        h11 = [False, True, False]
        h12 = [True, True, None]
        self.assertTrue(is_strictly_better(h10, h11))  # None -> False with preserved True
        self.assertTrue(is_strictly_better(h11, h12))  # False -> True with None
        
        # Non-comparable cases
        h13 = [None, True, False]
        h14 = [False, None, True]
        self.assertFalse(is_strictly_better(h13, h14))  # Different patterns
        self.assertFalse(is_strictly_better(h14, h13))  # Different patterns
        
        # Equal cases
        h15 = [None, True, False]
        h16 = [None, True, False]
        self.assertFalse(is_strictly_better(h15, h16))  # Same pattern
        
        # Complex trade-offs
        h17 = [None, None, True, True]
        h18 = [False, False, True, None]
        h19 = [False, False, True, True]
        self.assertFalse(is_strictly_better(h17, h18))   # More False is better despite one None
        self.assertTrue(is_strictly_better(h18, h19))   # None -> True improvement
        
        # Edge cases with all types
        h20 = [None, False, True]
        h21 = [False, False, True]
        h22 = [True, False, True]
        self.assertTrue(is_strictly_better(h20, h21))   # None -> False improvement
        self.assertTrue(is_strictly_better(h21, h22))   # False -> True improvement
        
        # Regressions
        h23 = [True, None, False]
        h24 = [True, False, None]
        self.assertFalse(is_strictly_better(h23, h24))  # Just different arrangements
        self.assertFalse(is_strictly_better(h24, h23))  # Just different arrangements

    def test_find_maximal_groups_with_none(self):
        verdicts = [
            [None, None, None],     # h0: {}
            [False, None, None],    # h1: {F}
            [False, False, None],   # h2: {F,F}
            [True, False, None],    # h3: {T,F}
            [True, True, None],     # h4: {T,T}
            [True, True, True]      # h5: {T,T,T}
        ]
        
        groups = find_maximal_groups(verdicts)
        expected_groups = [
            [0, 1, 2, 3, 4, 5]    # Single complete progression: None->False->True
        ]
        
        groups = sorted([sorted(g) for g in groups])
        expected_groups = sorted([sorted(g) for g in expected_groups])
        
        self.assertEqual(groups, expected_groups)
        
        # Test with mixed None patterns
        verdicts = [
            [None, True, None],    # h0
            [False, True, False],  # h1
            [True, True, False],   # h2
            [True, True, True]     # h3
        ]
        
        groups = find_maximal_groups(verdicts)
        expected_groups = [
            [0, 1, 2, 3]          # Single progression path
        ]
        
        groups = sorted([sorted(g) for g in groups])
        expected_groups = sorted([sorted(g) for g in expected_groups])
        
        self.assertEqual(groups, expected_groups)

if __name__ == '__main__':
    unittest.main()