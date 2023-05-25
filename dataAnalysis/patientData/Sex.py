import numpy as np
import matplotlib.pyplot as plt


class Sex:
    def __init__(self, data):
        self.genderChars = data["Sex"]
        self.genders = list(map(Sex.sex_classifier, self.genderChars))

    @staticmethod
    def sex_classifier(sex):
        if sex == "M":
            return 0
        if sex == "W":
            return 1
        raise Exception("Invalid Sex " + sex)

    def get_numpy_sex(self):
        np_sex = np.array(self.genders, dtype=np.uintc)
        return np_sex

    def get_number_of_males(self):
        return np.count_nonzero(self.get_numpy_sex() == 0)

    def get_numer_of_females(self):
        return np.count_nonzero(self.get_numpy_sex() == 1)

    def visualize_sex(self):
        fig = plt.figure(figsize=(10, 7))
        plt.pie([self.get_number_of_males(), self.get_numer_of_females()], labels=["Males", "Females"],
                autopct=lambda pct: f"{round(pct,1)} %")
        plt.show()
