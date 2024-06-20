import numpy as np

class Time:
    def __init__(self, data):
        data_without_nan = data[~data['Time'].isnull()]
        self.time = data_without_nan["Time"]
        print(np.unique(self.get_numpy_time()))

    def get_numpy_time(self):
        return self.time.to_numpy()
