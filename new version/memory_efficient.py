import sys
import time
import psutil


# String generation

def parse_input_file(input_path):

    with open(input_path, 'r') as f:
        # Strip whitespace and ignore completely empty lines
        lines = [line.strip() for line in f if line.strip() != ""]

    if len(lines) < 2:
        raise ValueError("Input file does not contain enough lines.")

    # First base string s0
    s0 = lines[0]

    # Find boundary where numeric indices for s0 stop and t0 begins
    idx = 1
    n_lines = len(lines)
    while idx < n_lines and lines[idx].isdigit():
        idx += 1

    if idx >= n_lines:
        raise ValueError("Could not find second base string t0 in input.")

    # s_indices are all integer lines between s0 and t0
    s_indices = [int(x) for x in lines[1:idx]]

    # t0 is the first non-integer line after indices for s
    t0 = lines[idx]

    # Remaining lines (if any) are indices for t0
    t_indices = [int(x) for x in lines[idx + 1:]]

    return s0, s_indices, t0, t_indices


def generate_string(base, indices):
 #Generate the final string from a base string and a list of insertion indices.


    s = base
    for i in indices:
        # Insert s after position i (0-indexed)
        insert_pos = i + 1  # "after index i" => after character at i
        s = s[:insert_pos] + s + s[insert_pos:]
    return s


#Memory-efficient alignment logic

class MemoryEfficientSequenceAligner:
    """
    Memory-efficient sequence aligner.

    - Time:  O(m * n)
    - Space: O(m + n)
    """

    def __init__(self):
        self.delta, self.alpha = self._init_penalties()

    def _init_penalties(self):
        #Initialize fixed gap and mismatch penalties.
        delta = 30  # Gap penalty
        alpha = {
            ('A', 'A'): 0,   ('A', 'C'): 110, ('A', 'G'): 48,  ('A', 'T'): 94,
            ('C', 'A'): 110, ('C', 'C'): 0,   ('C', 'G'): 118, ('C', 'T'): 48,
            ('G', 'A'): 48,  ('G', 'C'): 118, ('G', 'G'): 0,   ('G', 'T'): 110,
            ('T', 'A'): 94,  ('T', 'C'): 48,  ('T', 'G'): 110, ('T', 'T'): 0
        }
        return delta, alpha

    #Small-DP fallback

    def _align_small(self, X, Y):
        """
        Basic O(mn) DP + backtracking for small subproblems.

        this is only called when min(len(X), len(Y)) <= 1,
        so space stays O(m + n) overall.
        """
        m, n = len(X), len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize 0th row and 0th column
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + self.delta
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + self.delta

        # Bottom-up DP
        for i in range(1, m + 1):
            xi = X[i - 1]
            for j in range(1, n + 1):
                yj = Y[j - 1]
                match_cost = dp[i - 1][j - 1] + self.alpha[(xi, yj)]
                gap_x_cost = dp[i][j - 1] + self.delta  # gap in X
                gap_y_cost = dp[i - 1][j] + self.delta  # gap in Y
                dp[i][j] = min(match_cost, gap_x_cost, gap_y_cost)

        # Backtrack to reconstruct alignment
        align_X, align_Y = [], []
        i, j = m, n
        while i > 0 or j > 0:
            # Case 1: match/mismatch
            if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + self.alpha[(X[i - 1], Y[j - 1])]:
                align_X.append(X[i - 1])
                align_Y.append(Y[j - 1])
                i -= 1
                j -= 1
            # Case 2: gap in X
            elif j > 0 and dp[i][j] == dp[i][j - 1] + self.delta:
                align_X.append('_')
                align_Y.append(Y[j - 1])
                j -= 1
            # Case 3: gap in Y
            else:
                align_X.append(X[i - 1])
                align_Y.append('_')
                i -= 1

        return ''.join(reversed(align_X)), ''.join(reversed(align_Y))

    #Low-memory DP

    def _last_row_costs(self, X, Y):
        """
        Compute DP costs for aligning X with all prefixes of Y,
        keeping only the last row (length len(Y)+1).

        Standard DNA matching:
          dp[i][j] = min(
              dp[i-1][j-1] + alpha(X[i-1], Y[j-1]),  # match/mismatch
              dp[i][j-1]   + delta,                  # gap in X
              dp[i-1][j]   + delta                   # gap in Y
          )
        This implementation runs in O(|X||Y|) time and O(|Y|) space.
        """
        m, n = len(X), len(Y)

        # dp for i = 0 (X empty): cost is j * delta
        prev = [j * self.delta for j in range(n + 1)]

        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            curr[0] = i * self.delta  # cost for j = 0 (Y empty)
            xi = X[i - 1]

            for j in range(1, n + 1):
                yj = Y[j - 1]
                match_cost = prev[j - 1] + self.alpha[(xi, yj)]
                gap_x_cost = curr[j - 1] + self.delta  # dp[i][j-1] + delta
                gap_y_cost = prev[j] + self.delta      # dp[i-1][j] + delta
                curr[j] = min(match_cost, gap_x_cost, gap_y_cost)

            prev = curr

        return prev  # costs for aligning X with Y[0..j] for each j

    #divide-and-conquer

    def _divideconquer(self, X, Y):
        """
        Recursive divide-and-conquer alignment.
        Returns:
            (align_X, align_Y)
        """
        m, n = len(X), len(Y)

        # Base cases
        if m == 0:
            return '_' * n, Y
        if n == 0:
            return X, '_' * m
        if m == 1 or n == 1:
            # For tiny subproblems, use standard DP + backtracking
            return self._align_small(X, Y)

        # Divide: split X into two halves
        mid = m // 2
        X_left, X_right = X[:mid], X[mid:]

        # Forward DP: X_left vs all prefixes of Y
        left_costs = self._last_row_costs(X_left, Y)

        # Backward DP: reverse(X_right) vs prefixes of reverse(Y)
        right_costs_rev = self._last_row_costs(X_right[::-1], Y[::-1])

        # Find split point k of Y that minimizes:
        #   left_costs[k] + cost(X_right, Y[k..n])
        # where cost(X_right, Y[k..n]) = right_costs_rev[n-k]
        best_total = None
        best_k = 0
        for k in range(n + 1):
            total = left_costs[k] + right_costs_rev[n - k]
            if best_total is None or total < best_total:
                best_total = total
                best_k = k

        # Conquer: recursively align the two halves
        Y_left, Y_right = Y[:best_k], Y[best_k:]
        ax_left, ay_left = self._divideconquer(X_left, Y_left)
        ax_right, ay_right = self._divideconquer(X_right, Y_right)

        return ax_left + ax_right, ay_left + ay_right

    #Called by main

    def align(self, X, Y):
        """
        Compute optimal alignment for two input strings using
        the memory-efficient DP + divide-and-conquer algorithm.
        X (str): First input string (only A/C/G/T)
        Y (str): Second input string (only A/C/G/T)
        Returns:
            tuple: (min_cost: int, align_X: str, align_Y: str)
        """

        valid_chars = {'A', 'C', 'G', 'T'}
        if not (all(c in valid_chars for c in X) and all(c in valid_chars for c in Y)):
            raise ValueError("Input strings can only contain A/C/G/T")

        align_X, align_Y = self._divideconquer(X, Y)
        if len(align_X) != len(align_Y):
            raise RuntimeError("Alignment strings have different lengths")

        # Compute total cost based on the final alignment
        cost = 0
        for cx, cy in zip(align_X, align_Y):
            if cx == '_' or cy == '_':
                cost += self.delta
            else:
                cost += self.alpha[(cx, cy)]

        return cost, align_X, align_Y


# Time & memory utils



def process_memory():
    # Return total memory (RSS) in kilobytes using psutil.

    process = psutil.Process()
    mem_bytes = process.memory_info().rss
    return mem_bytes / 1024


# Main

def main(input_path, output_path):
    # 1. Parse input and generate full strings
    s0, s_indices, t0, t_indices = parse_input_file(input_path)
    X = generate_string(s0, s_indices)
    Y = generate_string(t0, t_indices)

    aligner = MemoryEfficientSequenceAligner()

    # 2. Measure memory and time around the alignment algorithm only
    before_mem = process_memory()
    start_time = time.time()

    cost, align_X, align_Y = aligner.align(X, Y)

    end_time = time.time()
    after_mem = process_memory()

    time_ms = (end_time - start_time) * 1000.0
    mem_kb = process_memory()

    # 3. Write output to file: 5 lines in required format
    #    1. Cost of the alignment (Integer)
    #    2. First string alignment
    #    3. Second string alignment
    #    4. Time in Milliseconds (Float)
    #    5. Memory in Kilobytes (Float)
    with open(output_path, 'w') as out:
        out.write("Cost: " + str(cost) + '\n')
        out.write("First alignment: " + align_X + '\n')
        out.write("Second Alignment: " + align_Y + '\n')
        out.write("Time: " + str(time_ms) + '\n')
        out.write("Memory: " + str(mem_kb) + '\n')


if __name__ == "__main__":
    # Expect exactly two arguments: input file path and output file path
    if len(sys.argv) < 3:
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
