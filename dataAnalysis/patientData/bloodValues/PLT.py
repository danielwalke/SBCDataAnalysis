from dataAnalysis.patientData.bloodValues.BloodValue import BloodValue


class PLT(BloodValue):
    def __init__(self, data):
        BloodValue.__init__(self, data, "PLT")