import numpy as np


class Sender:
    def __init__(self, data):
        data_without_nan = data[~data['Sender'].isnull()]
        self.sender = data_without_nan["Sender"]
        print(np.unique(self.get_numpy_sender()))

    def get_numpy_sender(self):
        return self.sender.to_numpy()
