from dataAnalysis.patientData.bloodValues.BloodValue import BloodValue


class MCV(BloodValue):
    def __init__(self, data):
        BloodValue.__init__(self, data, "MCV")