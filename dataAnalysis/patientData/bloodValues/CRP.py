from dataAnalysis.patientData.bloodValues.BloodValue import BloodValue


class CRP(BloodValue):
    def __init__(self, data):
        BloodValue.__init__(self, data, "CRP")