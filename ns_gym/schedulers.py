from numpy import inf
import ns_gym.base as base
import numpy as np
"""
The schdulers can be initialized at different times to we need to maintian an internal clock as well.
"""
class RandomScheduler(base.Scheduler):
    """Random event scheduler: Events occur randomly with a given probability at each time step.
    """
    def __init__(self, probability: float = 0.5, start=0, end=np.inf, seed=None) -> None:
        """
        Args:
            probability (float): The probability of an event occurring at each time step.
            seed (int, optional): Random generator seed. Defaults to None.
        """
        super().__init__(start, end)
        self.probability = probability
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, t: int) -> bool:
        if self.start <= t <= self.end:
            return self.rng.random() < self.probability
        else:
            return False

class CustomScheduler(base.Scheduler):
    """Custom event scheduler: Allows for custom event logic based on a user-defined function.
    """
    def __init__(self, event_function, start=0, end=np.inf) -> None:
        """
        Args:
            event_function (function): A function that takes the current time step as input and returns a boolean indicating whether an event should occur.
        """
        super().__init__(start, end)
        self.event_function = event_function

    def __call__(self, t: int) -> bool:
        if self.start <= t <= self.end:
            return self.event_function(t)
        else:
            return False
        
class ContinuousScheduler(base.Scheduler):
    """Contnious Event Schedueler : At every time step return true
    """
    def __init__(self,
                 start = 0, 
                 end = np.inf) -> None:
        super().__init__(start, end)
    def __call__(self, t: int) -> bool:
        if self.start <= t <= self.end:
            return True 
        else:
            return False

class DiscreteScheduler(base.Scheduler):
    """A discrete event scheudler returns a bool indicating where the system should transitoin at this time step
    """
    def __init__(self, 
                 event_list:set,
                 start = 0,
                 end = np.inf) -> None:
        """
        Args:
            event_list (set): List of time steps to make a transition 
        """
        super().__init__(start,end)
        self.event_list = event_list
        assert min(event_list) >= start, "Scheduler start time occurs after first event in event list"
        assert max(event_list) <= end, "Scheduler end time occurs before last event in event list"

    def __call__(self,t:int) -> bool:
        if self.start <= t <= self.end:
            return t in self.event_list

class PeriodicScheduler(base.Scheduler):
    """Periodic event scheduler: At periodic steps return true. 
    """
    def __init__(self,
                 period: int,
                 start = 0,
                 end = np.inf ) -> None:
        """
        Args:
            period (int): Period of event trasition times. 
        """
        super().__init__(start,end)
        self.period = period
    def __call__(self,t:int) -> bool: 
        return t%self.period == 0


class MemorylessScheduler(base.Scheduler):
    """Memoryless Scheudler: Events happen at intervals according to a Geometric distribution
        This scheudler modles teh number of trials that must be run before a success. 
        The scheduler samples from a geometric distirbution then records the new time an event will occur. 
        After a transition we resample from the geometric distribution. 
    """
    def __init__(self,
                p: float,
                start = 0,
                end = np.inf,
                seed = None) -> None:
        """Memoryless Scheduler: Sample from a geometric distribution

        Args:
            p (float): The probability of success of an individual trial.
            seed (_type_, optional): Random generator seed. Defaults to None.
        """
        super().__init__(start,end)
        self.p = p 
        self.rng = np.random.default_rng(seed = seed)
        self.transition_time = self.rng.geometric(p = self.p, size = (1,))

    def __call__(self,t:int) -> bool:
        if t == self.transition_time:
            delta_t = self.rng.geometric(p = self.p ,size = (1,))
            self.transition_time = delta_t + t
            return True
        else: 
            return False

if __name__ == "__main__":
    print(base.Scheduler.__subclasses__())