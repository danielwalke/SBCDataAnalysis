import numpy as np
import matplotlib.pyplot as plt


class Center:
    def __init__(self, data):
        self.center_labels = data["Center"]
        self.centers = list(map(Center.center_classifier, self.center_labels))

    @staticmethod
    def center_classifier(center):
        if center == "Greifswald":
            return 0
        if center == "Leipzig":
            return 1
        raise Exception("Invalid Center " + center)

    def get_numpy_centers(self):
        np_centers = np.array(self.centers, dtype=np.uintc)
        return np_centers

    def get_number_of_leipzig(self):
        return np.count_nonzero(self.get_numpy_centers() == 1)

    def get_numer_of_greifswald(self):
        return np.count_nonzero(self.get_numpy_centers() == 0)

    def visualize_diagnoses(self):
        fig = plt.figure(figsize=(10, 7))
        plt.pie([self.get_number_of_leipzig(), self.get_numer_of_greifswald()], labels=["Leipzig", "Greifswald"],
                autopct=lambda pct: f"{round(pct,1)} %")
        plt.show()