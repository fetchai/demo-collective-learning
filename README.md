A quick implementation of collective learning scheme (one worker trains, the others vote on the update) in pysyft.

- tutorial_6.py is a copy of Tutorial 6 from the pysyft tutorials.
- driver.py is the colearn demo. It creates n workers, selects a random one to train, then coordinates the voting. 