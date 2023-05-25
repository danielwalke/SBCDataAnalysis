import numpy as np
import matplotlib.pyplot as plt
from dataAnalysis.patientData.bloodValues.BloodValue import BloodValue

## TODO normalize?
class WBC(BloodValue):
    def __init__(self, data):
        print(len(data.query("WBC > 100", engine='python')))
        # data = data.query("WBC < 100", engine='python')
        self.wbc = data["WBC"]
        sepsis_data = data[data["Diagnosis"].str.contains("Sepsis")]
        control_data = data[data["Diagnosis"].str.contains("Control")]
        self.wbc_sepsis = sepsis_data["WBC"]
        self.wbc_control = control_data["WBC"]
        BloodValue.__init__(self, data, "WBC")

    def get_numpy_wbc(self):
        np_wbc = np.array(self.wbc, dtype=np.double)
        # np_wbc_numbers = np_wbc[~np.isnan(np_wbc)]
        return np_wbc

    def get_numpy_wbc_sepsis(self):
        np_wbc = np.array(self.wbc_sepsis, dtype=np.double)
        return np_wbc

    def get_numpy_wbc_control(self):
        np_wbc = np.array(self.wbc_control, dtype=np.double)
        return np_wbc

    def get_median_wbc(self):
        return np.median(self.get_numpy_wbc())

    def get_average_wbc(self):
        return np.average(self.get_numpy_wbc())

    def visualize_wbc_comparison(self):
        sepsis_wbc = self.get_numpy_wbc_sepsis()
        control_wbc = self.get_numpy_wbc_control()
        print(30*"%")
        print(np.max(control_wbc))
        print(np.min(control_wbc))
        print(np.min(sepsis_wbc))
        plt.subplot(1, 2, 1)
        plt.hist(sepsis_wbc, bins=len(np.unique(sepsis_wbc)), range=(0, np.max(sepsis_wbc)))
        plt.ylabel('frequency')
        plt.xlabel('wbc')
        plt.subplot(1, 2, 2)
        plt.hist(control_wbc, bins=len(np.unique(control_wbc)))
        plt.ylabel('frequency')
        plt.xlabel('wbc')
        plt.show()


