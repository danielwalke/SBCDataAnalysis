from dataAnalysis.patientData.bloodValues.BloodValue import BloodValue


class HGB(BloodValue):
    def __init__(self, data):
        BloodValue.__init__(self, data, "HGB")
