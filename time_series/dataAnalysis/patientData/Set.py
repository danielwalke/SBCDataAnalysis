import numpy as np
import matplotlib.pyplot as plt


class Set:
    def __init__(self, data):
        self.set_labels = data["Set"]
        self.sets = list(map(Set.set_classifier, self.set_labels))

    @staticmethod
    def set_classifier(center):
        if center == "Training":
            return 0
        if center == "Validation":
            return 1
        raise Exception("Invalid Set type " + center)

    def get_numpy_sets(self):
        np_sets = np.array(self.sets, dtype=np.uintc)
        return np_sets

    def get_number_of_trainings(self):
        return np.count_nonzero(self.get_numpy_sets() == 0)

    def get_number_of_validations(self):
        return np.count_nonzero(self.get_numpy_sets() == 1)

    def visualize_sets(self):
        fig = plt.figure(figsize=(10, 7))
        plt.pie([self.get_number_of_trainings(), self.get_number_of_validations()], labels=["Training", "Validation"],
                autopct=lambda pct: f"{round(pct,1)} %")
        plt.show()