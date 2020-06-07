# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code includes an implementation of the algorithm described in Ikeda,
# K., Nakamura, Y. & Humble, T.S. Application of Quantum Annealing to Nurse
# Scheduling Problem. Sci Rep 9, 12837 (2019).
# https://doi.org/10.1038/s41598-019-49172-3, © The Author(s) 2019, use of
# which is licensed under a Creative Commons Attribution 4.0 International
# License (To view a copy of this license, visit
# http://creativecommons.org/licenses/by/4.0/).

from __future__ import print_function

import dwavebinarycsp
import re
def get_maze_bqm(n_rows, n_cols, start, end, walls, penalty_per_tile=0.5):
    """Returns a BQM that corresponds to a valid path through a maze. This maze is described by the parameters.
    Specifically, it uses the parameters to build a maze constraint satisfaction problem (CSP). This maze CSP is then
    converted into the returned BQM.
    Note: If penalty_per_tile is too large, the path will be too heavily penalized and the optimal solution might
    produce no path at all.
    Args:
        n_rows: Integer. The number of rows in the maze.
        n_cols: Integer. The number of cols in the maze.
        start: String. The location of the starting point of the maze. String follows the format of get_label(..).
        end: String. The location of the end point of the maze. String follows the format of get_label(..).
        walls: List of Strings. The list of inner wall locations. Locations follow the format of get_label(..).
        penalty_per_tile: A number. Penalty for each tile that is included in the path; encourages shorter paths.
    Returns:
        A dimod.BinaryQuadraticModel
    """
    maze = Maze(n_rows, n_cols, start, end, walls)
    return maze.get_bqm(penalty_per_tile)
def get_label(row, col, direction):
    """Provides a string that follows a standard format for naming constraint variables in Maze.
        Namely, "<row_index>,<column_index><north_or_west_direction>".
        Args:
        row: Integer. Index of the row.
        col: Integer. Index of the column.
        direction: String in the set {'n', 'w'}. 'n' indicates north and 'w' indicates west.
    """
    return "{row},{col}{direction}".format(**locals())


def assert_label_format_valid(label):
    """Checks that label conforms with the standard format for naming constraint variables in Maze.
    Namely, "<row_index>,<column_index><north_or_west_direction>".
    Args:
        label: String.
    """
    is_valid = bool(re.match(r'^(\d+),(\d+)[nw]$', label))
    assert is_valid, ("{label} is in the incorrect format. Format is <row_index>,<column_index><north_or_west>. "
                      "Example: '4,3w'").format(**locals())


def sum_to_two_or_zero(*args):
    """Checks to see if the args sum to either 0 or 2.
    """
    sum_value = sum(args)
    return sum_value in [0, 2]
class Maze:
    """An object that stores all the attributes necessary to represent a maze as a constraint satisfaction problem.
        Args:
        n_rows: Integer. The number of rows in the maze.
        n_cols: Integer. The number of cols in the maze.
        start: String. The location of the starting point of the maze. String follows the format of get_label(..).
        end: String. The location of the end point of the maze. String follows the format of get_label(..).
        walls: List of Strings. The list of inner wall locations. Locations follow the format of get_label(..).
    """
    def __init__(self, n_rows, n_cols, start, end, walls):
        assert isinstance(n_rows, int) and n_rows > 0, "'n_rows' is not a positive integer".format(n_rows)
        assert isinstance(n_cols, int) and n_cols > 0, "'n_cols' is not a positive integer".format(n_cols)
        assert start != end, "'start' cannot be the same as 'end'"

        # Check label format
        assert_label_format_valid(start)
        assert_label_format_valid(end)

        for wall in walls:
            assert_label_format_valid(wall)

        # Instantiate
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = start
        self.end = end
        self.walls = walls
        self.csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)

    def _apply_valid_move_constraint(self):
        """Applies a sum to either 0 or 2 constraint on each tile of the maze.
        Note: This constraint ensures that a tile is either not entered at all (0), or is entered and exited (2).
        """
        # Grab the four directions of each maze tile and apply two-or-zero constraint
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                directions = {get_label(i, j, 'n'), get_label(i, j, 'w'), get_label(i+1, j, 'n'),
                              get_label(i, j+1, 'w')}
                self.csp.add_constraint(sum_to_two_or_zero, directions)

    def _set_start_and_end(self):
        """Sets the values of the start and end locations of the maze.
        """
        self.csp.fix_variable(self.start, 1)  # start location
        self.csp.fix_variable(self.end, 1)  # end location

    def _set_borders(self):
        """Sets the values of the outer border of the maze; prevents a path from forming over the border.
        """
        for j in range(self.n_cols):
            top_border = get_label(0, j, 'n')
            bottom_border = get_label(self.n_rows, j, 'n')

            try:
                self.csp.fix_variable(top_border, 0)
            except ValueError:
                if not top_border in [self.start, self.end]:
                    raise ValueError

            try:
                self.csp.fix_variable(bottom_border, 0)
            except ValueError:
                if not bottom_border in [self.start, self.end]:
                    raise ValueError

        for i in range(self.n_rows):
            left_border = get_label(i, 0, 'w')
            right_border = get_label(i, self.n_cols, 'w')

            try:
                self.csp.fix_variable(left_border, 0)
            except ValueError:
                if not left_border in [self.start, self.end]:
                    raise ValueError

            try:
                self.csp.fix_variable(right_border, 0)
            except ValueError:
                if not right_border in [self.start, self.end]:
                    raise ValueError

    def _set_inner_walls(self):
        """Sets the values of the inner walls of the maze; prevents a path from forming over an inner wall.
        """
        for wall in self.walls:
            self.csp.fix_variable(wall, 0)

    def get_bqm(self, penalty_per_tile=0.5):
        """Applies the constraints necessary to form a maze and returns a BQM that would correspond to a valid path
        through said maze.
        Note: If penalty_per_tile is too large, the path will be too heavily penalized and the optimal solution might
          no path at all.
        Args:
            penalty_per_tile: A number. Penalty for each tile that is included in the path; encourages shorter paths.
        Returns:
            A dimod.BinaryQuadraticModel
        """
        # Apply constraints onto self.csp
        self._apply_valid_move_constraint()
        self._set_start_and_end()
        self._set_borders()
        self._set_inner_walls()

        # Grab bqm constrained for valid solutions
        bqm = dwavebinarycsp.stitch(self.csp)

        # Edit bqm to favour optimal solutions
        for v in bqm.variables:
            # Ignore auxiliary variables
            if isinstance(v, str) and re.match(r'^aux\d+$', v):
                continue

            # Add a penalty to every tile of the path
            bqm.add_variable(v, penalty_per_tile)

        return bqm
    def visualize(self, solution=None):
        def get_visual_coords(coords):
            coord_pattern = "^(\d+),(\d+)([nw])$"
            row, col, dir = re.findall(coord_pattern, coords)[0]
            new_row, new_col = map(lambda x: int(x) * 2 + 1, [row, col])
            new_row, new_col = (new_row-1, new_col) if dir == "n" else (new_row, new_col-1)

            return new_row, new_col, dir

        # Constants for maze symbols
        WALL = "#"      # maze wall
        NS = "|"        # path going in north-south direction
        EW = "_"        # path going in east-west direction
        POS = "."       # coordinate position
        EMPTY = " "     # whitespace; indicates no path drawn

        # Check parameters
        if solution is None:
            solution = []

        # Construct empty maze visual
        # Note: the maze visual is (2 * original-maze-dimension + 1) because
        #   each position has an associated north-edge and an associated
        #   west-edge. This requires two rows and two columns to draw,
        #   respectively. Thus, the "2 * original-maze-dimension" is needed.
        #      |      <-- north edge
        #     _.      <-- west edge and position
        #   To get a south-edge or an east-edge, the north-edge from the row
        #   below or the west-edge from the column on the right can be used
        #   respectively. This trick, however, cannot be used for the last row
        #   nor for the rightmost column, hence the "+ 1" in the equation.
        width = 2*self.n_cols + 1       # maze visual's width
        height = 2*self.n_rows + 1      # maze visual's height

        empty_row = [EMPTY] * (width-2)
        empty_row = [WALL] + empty_row + [WALL]   # add left and right borders

        visual = [list(empty_row) for _ in range(height)]
        visual[0] = [WALL] * width      # top border
        visual[-1] = [WALL] * width     # bottom border

        # Add coordinate positions in maze visual
        # Note: the symbol POS appears at every other position because there
        #   could potentially be a path segment sitting between the two
        #   positions.
        for position_row in visual[1::2]:
            position_row[1::2] = [POS] * self.n_cols

        # Add maze start and end to visual
        start_row, start_col, start_dir = get_visual_coords(self.start)
        end_row, end_col, end_dir = get_visual_coords(self.end)
        visual[start_row][start_col] = NS if start_dir=="n" else EW
        visual[end_row][end_col] = NS if end_dir=="n" else EW

        # Add interior walls to visual
        for w in self.walls:
            row, col, _ = get_visual_coords(w)
            visual[row][col] = WALL

        # Add solution path to visual
        for s in solution:
            row, col, dir = get_visual_coords(s)
            visual[row][col] = NS if dir=="n" else EW

        # Print solution
        for s in visual:
            print("".join(s))
    
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import re

# from maze import get_maze_bqm, Maze

# Create maze
n_rows = 3
n_cols = 4
start = '0,0n'              # maze entrance location
end = '2,4w'                # maze exit location
walls = ['1,1n', '2,2w']    # maze interior wall locations

# Construct BQM
m = Maze(n_rows, n_cols, start, end, walls)
bqm = m.get_bqm()

# Submit BQM to a D-Wave sampler
sampler = EmbeddingComposite(DWaveSampler(solver={'qpu':True}))
result = sampler.sample(bqm, num_reads=1000, chain_strength=2)

# Interpret result
# Note: when grabbing the path, we are only grabbing path segments that have
#   been "selected" (i.e. indicated with a 1).
# Note2: in order construct the BQM such that the maze solution corresponds to
#   the ground energy, auxiliary variables
#   may have been included in the BQM. These auxiliary variables are no longer
#   useful once we have our result. Hence, we can just ignore them by filtering
#   them out with regex (i.e. re.match(r"^aux(\d+)$", k)])
path = [k for k, v in result.first.sample.items() if v==1
            and not re.match(r"^aux(\d+)$", k)]

# Visualize maze path
m.visualize(path)
print("\n")
print(result.first.sample)

'''
from dwave.system import LeapHybridSampler
from dimod import BinaryQuadraticModel
from collections import defaultdict
from copy import deepcopy

# Overall model variables: problem size
# binary variable q_nd is the assignment of nurse n to day d
n_nurses = 3      # count nurses n = 1 ... n_nurses
n_days = 11       # count scheduling days as d = 1 ... n_days
size = n_days * n_nurses

# Parameters for hard nurse constraint
# a is a positive correlation coefficient for implementing the hard nurse
# constraint - value provided by Ikeda, Nakamura, Humble
a = 3.5

# Parameters for hard shift constraint
# Hard shift constraint: at least one nurse working every day
# Lagrange parameter, for hard shift constraint, on workforce and effort
lagrange_hard_shift = 1.3
workforce = 1     # Workforce function W(d) - set to a constant for now
effort = 1        # Effort function E(n) - set to a constant for now

# Parameters for soft nurse constraint
# Soft nurse constraint: all nurses should have approximately even work
#                        schedules
# Lagrange parameter, for shift constraints, on work days is called gamma
# in the paper
# Minimum duty days 'min_duty_days' - the number of work days that each
# nurse wants
# to be scheduled. At present, each will do the minimum on average.
# The parameter gamma's value suggested by Ikeda, Nakamura, Humble
lagrange_soft_nurse = 0.3      # Lagrange parameter for soft nurse, gamma
preference = 1                 # preference function - constant for now
min_duty_days = int(n_days/n_nurses)


# Find composite index into 1D list for (nurse_index, day_index)
def get_index(nurse_index, day_index):
    return nurse_index * n_days + day_index


# Inverse of get_index - given a composite index in a 1D list, return the
# nurse_index and day_index
def get_nurse_and_day(index):
    nurse_index, day_index = divmod(index, n_days)
    return nurse_index, day_index


# Hard nurse constraint: no nurse works two consecutive days
# It does not have Lagrange parameter - instead, J matrix
# symmetric, real-valued interaction matrix J, whereas all terms are
# a or zero.
# composite indices i(n, d) and j(n, d) as functions of n and d
# J_i(n,d)j(n,d+1) = a and 0 otherwise.
J = defaultdict(int)
for nurse in range(n_nurses):
    for day in range(n_days - 1):
        nurse_day_1 = get_index(nurse, day)
        nurse_day_2 = get_index(nurse, day+1)
        J[nurse_day_1, nurse_day_2] = a

# Q matrix assign the cost term, the J matrix
Q = deepcopy(J)

# Hard shift constraint: at least one nurse working every day
# The sum is over each day.
# This constraint tries to make (effort * sum(q_i)) equal to workforce,
# which is set to a constant in this implementation, so that one nurse
# is working each day.
# Overall hard shift constraint:
# lagrange_hard_shift * sum_d ((sum_n(effort * q_i(n,d)) - workforce) ** 2)
#
# with constant effort and constant workforce:
# = lagrange_hard_shift * sum_d ( effort * sum_n q_i(n,d) - workforce ) ** 2
# = lagrange_hard_shift * sum_d [ effort ** 2 * (sum_n q_i(n,d) ** 2)
#                              - 2 effort * workforce * sum_n q_i(n,d)
#                              + workforce ** 2 ]
# The constant term is moved to the offset, below, right before we solve
# the QUBO
#
# Expanding and merging the terms ( m is another sum over n ):
# lagrange_hard_shift * (effort ** 2 - 2 effort * workforce) *
# sum_d sum_n q_i(n,d)
# + lagrange_hard_shift * effort ** 2 * sum_d sum_m sum_n q_i(n,d) q_j(m, d) #

# Diagonal terms in hard shift constraint, without the workforce**2 term
for nurse in range(n_nurses):
    for day in range(n_days):
        ind = get_index(nurse, day)
        Q[ind, ind] += lagrange_hard_shift * (effort ** 2 - (2 * workforce * effort))

# Off-diagonal terms in hard shift constraint
# Include only the same day, across nurses
for day in range(n_days):
    for nurse1 in range(n_nurses):
        for nurse2 in range(nurse1 + 1, n_nurses):

            ind1 = get_index(nurse1, day)
            ind2 = get_index(nurse2, day)
            Q[ind1, ind2] += 2 * lagrange_hard_shift * effort ** 2

# Soft nurse constraint: all nurses should have approximately even work
#                        schedules
# This constraint tries to make preference * sum(q_i) equal to min_duty_days,
# so that the nurses have the same number of days. The sum of the q_i,
# over the number of days, is each nurse's number of days worked in the
# schedule.
# Overall soft nurse constraint:
# lagrange_soft_nurse * sum_n ((sum_d(preference * q_i(n,d)) - min_duty_days) ** 2)
# with constant preference and constant min_duty_days:
# = lagrange_soft_nurse * sum_n ( preference * sum_d q_i(n,d) - min_duty_days ) ** 2
# = lagrange_soft_nurse * sum_n [ preference ** 2 * (sum_d q_i(n,d) ** 2)
#                              - 2 preference * min_duty_days * sum_d q_i(n,d)
#                              + min_duty_days ** 2 ]
# The constant term is moved to the offset, below, right before we solve
# the QUBO
#
# The square of the the sum_d term becomes:
# Expanding and merging the terms (d1 and d2 are sums over d):
# = lagrange_soft_nurse * (preference ** 2 - 2 preference * min_duty_days) * sum_n sum_d q_i(n,d)
# + lagrange_soft_nurse * preference ** 2 * sum_n sum_d1 sum_d2 q_i(n,d1)
#                      * q_j(n, d2)

# Diagonal terms in soft nurse constraint, without the min_duty_days**2 term
for nurse in range(n_nurses):
    for day in range(n_days):
        ind = get_index(nurse, day)
        Q[ind, ind] += lagrange_soft_nurse * (preference ** 2 - (2 * min_duty_days * preference))

# Off-diagonal terms in soft nurse constraint
# Include only the same nurse, across days
for nurse in range(n_nurses):
    for day1 in range(n_days):
        for day2 in range(day1 + 1, n_days):

            ind1 = get_index(nurse, day1)
            ind2 = get_index(nurse, day2)
            Q[ind1, ind2] += 2 * lagrange_soft_nurse * preference ** 2

# Solve the problem, and use the offset to scale the energy
e_offset = (lagrange_hard_shift * n_days * workforce ** 2) + (lagrange_soft_nurse * n_nurses * min_duty_days ** 2)
bqm = BinaryQuadraticModel.from_qubo(Q, offset=e_offset)
sampler = LeapHybridSampler()
results = sampler.sample(bqm)

# Get the results
smpl = results.first.sample
energy = results.first.energy
print("Size ", size)
print("Energy ", energy)


# Check the results by doing the sums directly
# J sum
sum_j = 0
for i in range(size):
    for j in range(size):
        sum_j += J[i, j] * smpl[i] * smpl[j]
print("Checking Hard nurse constraint ", sum_j)

# workforce sum
sum_w = 0
for d in range(n_days):
    sum_n = 0
    for n in range(n_nurses):
        sum_n += effort * smpl[get_index(n, d)]
    sum_w += lagrange_hard_shift * (sum_n - workforce) * (sum_n - workforce)
print("Checking Hard shift constraint ", sum_w)

# min_duty_days sum
sum_f = 0
for n in range(n_nurses):
    sum_d = 0
    for d in range(n_days):
        sum_d += preference * smpl[get_index(n, d)]
    sum_f += lagrange_soft_nurse * (sum_d - min_duty_days) * (sum_d - min_duty_days)
print("Checking Soft nurse constraint ", sum_f)

# Graphics
sched = [get_nurse_and_day(j) for j in range(size) if smpl[j] == 1]
str_header_for_output = " " * 11
str_header_for_output += "  ".join(map(str, range(n_days)))
print(str_header_for_output)
for n in range(n_nurses):
    str_row = ""
    for d in range(n_days):
        outcome = "X" if (n, d) in sched else " "
        if d > 9:
            outcome += " "
        str_row += "  " + outcome
    print("Nurse ", n, str_row)
    '''
