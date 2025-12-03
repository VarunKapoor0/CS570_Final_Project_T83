class BasicSequenceAligner:
    """
    Encapsulates O(mn) dynamic programming logic for sequence alignment,
    including penalty initialization, DP table construction, and backtracking.
    """

    def __init__(self):
        # Initialize fixed penalties on instance creation
        self.delta, self.alpha = self._init_penalties()

    def _init_penalties(self):
        """Private: Initialize fixed gap and mismatch penalties (doc-specified)."""
        delta = 30  # Gap penalty
        alpha = {
            ('A', 'A'): 0, ('A', 'C'): 110, ('A', 'G'): 48, ('A', 'T'): 94,
            ('C', 'A'): 110, ('C', 'C'): 0, ('C', 'G'): 118, ('C', 'T'): 48,
            ('G', 'A'): 48, ('G', 'C'): 118, ('G', 'G'): 0, ('G', 'T'): 110,
            ('T', 'A'): 94, ('T', 'C'): 48, ('T', 'G'): 110, ('T', 'T'): 0
        }
        return delta, alpha

    def _build_dp_table(self, X, Y):
        """
        Private: Construct O(mn) DP table to compute minimum alignment cost.
        :param X: 1st DNA string
        :param Y: 2nd DNA string
        :return: DP table; [[int]]
        """
        m, n = len(X), len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize 0th row (X empty: all gaps for X)
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + self.delta
        # Initialize 0th column (Y empty: all gaps for Y)
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + self.delta

        # Bottom up
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match_cost = dp[i - 1][j - 1] + self.alpha[(X[i - 1], Y[j - 1])]
                gap_x_cost = dp[i][j - 1] + self.delta
                gap_y_cost = dp[i - 1][j] + self.delta
                dp[i][j] = min(match_cost, gap_x_cost, gap_y_cost)

        return dp

    def _backtrack_alignment(self, dp, X, Y):
        """
        Private: Backtrack DP table to reconstruct optimal alignment.
        :param dp: DP table constructed by _build_dp_table method
        :param X: 1st DNA string
        :param Y: 2nd DNA string
        :return: tuple (align_X, align_Y)
            align_X: Aligned X string (with '_' for gaps)
            align_Y: Aligned Y string (with '_' for gaps)
        """
        align_X, align_Y = [], []
        i, j = len(X), len(Y)

        while i > 0 or j > 0:
            # Case 1: Cost from match/mismatch (prioritize for tie handling)
            if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + self.alpha[(X[i - 1], Y[j - 1])]:
                align_X.append(X[i - 1])
                align_Y.append(Y[j - 1])
                i -= 1
                j -= 1
            # Case 2: Cost from gap in X
            elif j > 0 and dp[i][j] == dp[i][j - 1] + self.delta:
                align_X.append('_')
                align_Y.append(Y[j - 1])
                j -= 1
            # Case 3: Cost from gap in Y
            else:
                align_X.append(X[i - 1])
                align_Y.append('_')
                i -= 1

        # Reverse to correct backtracking order
        return ''.join(reversed(align_X)), ''.join(reversed(align_Y))

    def align(self, X, Y):
        """
        Public API: Compute optimal alignment for two input strings.
        X (str): First input string (only A/C/G/T)
        Y (str): Second input string (only A/C/G/T)
        Returns:
            tuple: (min_cost: int, align_X: str, align_Y: str)
                min_cost: Minimum alignment cost
                align_X: Aligned X (with '_' for gaps)
                align_Y: Aligned Y (with '_' for gaps)
        """
        # Validate input characters (optional but defensive)
        valid_chars = {'A', 'C', 'G', 'T'}
        if not (all(c in valid_chars for c in X) and all(c in valid_chars for c in Y)):
            raise ValueError("Input strings can only contain A/C/G/T")

        dp_table = self._build_dp_table(X, Y)
        min_cost = dp_table[len(X)][len(Y)]
        align_X, align_Y = self._backtrack_alignment(dp_table, X, Y)

        return min_cost, align_X, align_Y


def test_basic_aligner():

    aligner = BasicSequenceAligner()
    test_cases = [
        # Case 1: (A vs C, expected cost 60)
        {
            "name": "DocSample-A_vs_C",
            "X": "A",
            "Y": "C",
            "expected_cost": 60
        },
        # Case 2: Exact match (expected cost 0)
        {
            "name": "ExactMatch-ACTG_vs_ACTG",
            "X": "ACTG",
            "Y": "ACTG",
            "expected_cost": 0
        },
        # Case 3: Edge case - X is empty
        {
            "name": "EdgeCase-X_Empty",
            "X": "",
            "Y": "T",
            "expected_cost": 30
        },
        # Case 4: Large length difference (X=A vs Y=ACG)
        {
            "name": "LenDiff-A_vs_ACG",
            "X": "A",
            "Y": "ACG",
            "expected_cost": 60
        },
        # Case 5: Complete mismatch (G vs T)
        {
            "name": "CompleteMismatch-G_vs_T",
            "X": "G",
            "Y": "T",
            "expected_cost": 60
        }
    ]

    passed = 0
    total = len(test_cases)
    valid_chars = {'A', 'C', 'G', 'T', '_'}

    for case in test_cases:
        try:
            cost, ax, ay = aligner.align(case["X"], case["Y"])

            # Validate core metrics
            assert cost == case["expected_cost"], f"Cost mismatch: exp {case['expected_cost']}, act {cost}"
            assert len(ax) == len(ay), f"Length mismatch: ax={len(ax)}, ay={len(ay)}"
            assert all(c in valid_chars for c in ax) and all(
                c in valid_chars for c in ay), "Invalid characters in alignment"

            passed += 1
            print(f" Passed: {case['name']}")
            print(f" Cost: {cost} | Align X: {ax} | Align Y: {ay}\n")

        except (AssertionError, ValueError) as e:
            print(f"Failed: {case['name']} - {str(e)}\n")
        except Exception as e:
            print(f"Error: {case['name']} - {str(e)}\n")

    print(f"Test Summary: {passed}/{total} cases passed")
    if passed != total:
        raise Exception("Some tests failed. Check aligner logic.")


if __name__ == "__main__":
    import sys
    import time
    import psutil

    if len(sys.argv) == 3:
        in_file = sys.argv[1]
        out_file = sys.argv[2]

        from memory_efficient import parse_input_file, generate_string  

        s0, s_indices, t0, t_indices = parse_input_file(in_file)
        X = generate_string(s0, s_indices)
        Y = generate_string(t0, t_indices)

        aligner = BasicSequenceAligner()

        # --- time & memory before ---
        t_before = time.time() * 1000
        mem_before = psutil.Process().memory_info().rss / 1024

        cost, ax, ay = aligner.align(X, Y)

        # --- time & memory after ---
        t_after = time.time() * 1000
        mem_after = psutil.Process().memory_info().rss / 1024

        # --- output  ---
        with open(out_file, "w") as f:
            f.write(str(cost) + "\n")
            f.write(ax + "\n")
            f.write(ay + "\n")
            f.write(str(t_after - t_before) + "\n")
            f.write(str(mem_after) + "\n") 

    else:
        test_basic_aligner()

