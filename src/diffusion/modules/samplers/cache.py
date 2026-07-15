
class _Cache():
    def __init__(self, level, cache_labels, optable):
        self.level = level
        self.keys = cache_labels    
        self.optable = optable
        self.reset()
        
    def update(self, items):
        for key, val in items.items():
            self.cache[key] = [val, *self.cache[key][:self.level-1]]

    def reset(self):
        self.cache = {
            k: [] for k in self.keys
        }
            
    def __call__(self, items):
        self.update(items)
        return self.optable(self.cache)