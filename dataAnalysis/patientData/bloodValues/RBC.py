from dataAnalysis.patientData.bloodValues.BloodValue import BloodValue


class RBC(BloodValue):
    def __init__(self, data):
        BloodValue.__init__(self, data, "RBC")