"""Overnight UCB optimizer for the MAZE race (race_maze_hard).

Maximizes the number of cells that escape the maze, sweeping the food-gathering
levers + the boids params (see race_opt.py). No time limit; saves
race_maze_hard_winner_<k>.gif whenever the escaped count beats the best by >= 10, and
rewrites WINNERS_race_maze_hard.md.

    PYTHONPATH=../src python opt_maze.py               # run until killed
    PYTHONPATH=../src python opt_maze.py 36000         # optional: stop after N seconds
    bsub ... "python opt_maze.py"                      # cluster
"""
import sys
import race_opt

if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else float("inf")
    race_opt.main("race_maze_hard", budget)
