import numpy as np


class Episode:
    def __init__(self, data):
        data_without_nan = data[~data['Episode'].isnull()]
        self.episode = data_without_nan["Episode"]
        print(np.unique(self.get_numpy_episode()))

    def get_numpy_episode(self):
        return self.episode.to_numpy()
