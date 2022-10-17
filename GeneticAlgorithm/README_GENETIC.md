# Genetic Algorithms

## Idea

Come up with a bunch of features that represent game state, each will be given a weight associated with it

---

## Genetic Algorithm Components

1. **Initial Population:** Initialise the weights for each feature to be random
1. **Fitness Function:** Can look at win/lose/draw + how many turns taken
1. **Selection:** Pairs are selected based on fitness score
1. **Crossover:** Select crossover point, and swap features between each to create children - 2 children are created
1. **Mutation:** For certain children, can mutate with low probability
1. **Termination:** Offspring are not significantly different from previous generation

---

## Pseudocode

```
START
Generate the initial population
Compute fitness
REPEAT
    Selection
    Crossover
    Mutation
    Compute fitness
UNTIL population has converged
STOP
```

---

## Heuristics

We need to find the value of each move - Find the utility for each column we put in
We have to determine the weights of each heuristic (How much weight that heuristic adds to the value of placing in that column)
After calculating value of each column, we choose the one with the best value
Let the AI play against itself, choose top performing to mate

### Possible heuristics:

1. Selected column results in win
2. Selected column results in blocking opponent from winning next turn
3. Selected column leads to 3 tiles in a row, in any direction (line of 3, can go through empty spaces, can ignore empty, as long as not blocked by opponent)
4. Selected column leads to 2 tiles in a row, in any direction (line of 2, can go through empty spaces)
5. Selected column leads to situation of inevitable win - 2 in a row, both sides still have space for tiles
6. Number of possible directions to win from (left, right, up, diagonals are not blocked)
7. Selecting a column that will result in opponent winning the next turn (negative value)
8. Placing at column leads to 3 tiles in a row, in > 1 direction
9. Placing at column leads to 2 tiles in a row, in > 1 direction
10. Placing on columns 2, 4 and 6

**Note:** An inevitable win is a move which will result in 3 in a row, with empty spaces on either side of the set

---

## Inevitable Wins

Situation 1: Inevitable win for agent 1

```
0 0 0 0 0 0 0
0 2 0 0 0 0 0
2 2 0 1 1 1 0
```
