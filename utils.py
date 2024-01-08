class LinearIterator:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
    
    def value(self, step):
        return self.start + (self.end - self.start) * step / self.steps