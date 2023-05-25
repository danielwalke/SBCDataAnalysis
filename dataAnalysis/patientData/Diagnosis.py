import numpy as np
import matplotlib.pyplot as plt


class Diagnosis:
    def __init__(self, data):
        self.diagnosis_label = data["Diagnosis"]
        self.diagnoses = list(map(Diagnosis.diagnosis_classifier, self.diagnosis_label))

    @staticmethod
    def diagnosis_classifier(diagnosis):
        if diagnosis == "Control":
            return 0
        if diagnosis == "Sepsis":
            return 1
        if diagnosis == "SIRS":
            return 2
        raise Exception("Invalid Diagnosis" + diagnosis)


    def get_numpy_diagnoses(self):
        np_diagnoses = np.array(self.diagnoses, dtype=np.uintc)
        return np_diagnoses

    def get_number_of_controls(self):
        return np.count_nonzero(self.get_numpy_diagnoses() == 0)

    def get_numer_of_cases(self):
        return np.count_nonzero(self.get_numpy_diagnoses() == 1)

    def get_numer_of_sirs(self):
        return np.count_nonzero(self.get_numpy_diagnoses() == 2)

    def visualize_diagnoses(self):
        fig = plt.figure(figsize=(10, 7))
        plt.pie([self.get_number_of_controls(), self.get_numer_of_cases(), self.get_numer_of_sirs()], labels=["Control", "Sepsis", "SIRS"],
                autopct=lambda pct: f"{round(pct,1)} %")
        plt.show()
