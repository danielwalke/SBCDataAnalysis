import numpy as np
import matplotlib.pyplot as plt


class Age:
    def __init__(self, data):
        self.ages = data["Age"]
        sepsis_data = data[~data["Diagnosis"].str.contains("Sepsis")]
        control_data = data[~data["Diagnosis"].str.contains("Control")]
        self.sepsis_ages = sepsis_data["Age"]
        self.control_ages = control_data["Age"]

    def get_numpy_ages(self):
        np_ages = self.ages.to_numpy(dtype=np.uintc)
        return np_ages

    def get_numpy_sepsis_ages(self):
        np_ages = self.sepsis_ages.to_numpy(dtype=np.uintc)
        return np_ages

    def get_numpy_control_ages(self):
        np_ages = self.control_ages.to_numpy(dtype=np.uintc)
        return np_ages


    def get_avg_age(self):
        avg = np.average(self.get_numpy_ages())
        return avg

    def get_media_age(self):
        return np.median(self.get_numpy_ages())

    def visualize_age(self):
        ages = self.get_numpy_ages()
        plt.hist(ages, bins=len(np.unique(ages)))
        plt.ylabel('frequency')
        plt.xlabel('age')
        plt.show()

    def compare_ages(self):
        sepsis_ages = self.get_numpy_sepsis_ages()
        control_ages = self.get_numpy_control_ages()
        plt.subplot(1, 2, 1)
        plt.hist(sepsis_ages, bins=len(np.unique(sepsis_ages)))
        plt.ylabel('frequency')
        plt.xlabel('age')
        plt.subplot(1, 2, 2)
        plt.hist(control_ages, bins=len(np.unique(control_ages)))
        plt.ylabel('frequency')
        plt.xlabel('age')
        plt.show()

