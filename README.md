# ud820-proj

Coding Resources for the Udacity ud820 final project. In this project we are working with two domains - Gridworld and Pacman.

## Getting Started

To get started read the [Project Description][1].

## Try it

### Gridworld

For a full list of options:
```
python gridworld.py -h
```

For moving the agent in the gridworld manually using arrow keys:
```
python gridworld.py -m
```

Learn using value iteration:
```
python gridworld.py -a value -i 100 -k 10
```

Learn using Q learning from manual interactions:
```
python gridworld.py -a q -k 5 -m
```

Learn using Q learning using epsilon greedy selection (picks the action that maximizes Q, or a random action with probability epsilon):
```
python gridworld.py -a q -k 100
```

With a simple feature extractor:
```
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
```

## Documentation

Find the docs/ folder in the repository.

[1]:https://docs.google.com/document/d/1NN6shM9oB_sdppT6zsVFuQrSuJ077Jg5oyQQIV8TXgk/pub
