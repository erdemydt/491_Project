from Demon import Demon
from Tape import Tape
import numpy as np

class Simulation:
    def __init__(self, demon: Demon, tape: Tape, tau: float):
        self.demon = demon
        self.tape_arr = tape.tape_arr
        self.tau = tau
        self.N = tape.N

