import numpy as np


class SecToIcu:
    def __init__(self, data):
        data_without_nan = data[~data['SecToIcu'].isnull()]
        self.sec_to_icu = data_without_nan["SecToIcu"]
        print(np.unique(self.get_numpy_sec_to_icu()))

    def get_numpy_sec_to_icu(self):
        return self.sec_to_icu.to_numpy()
