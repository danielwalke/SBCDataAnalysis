from dataAnalysis.patientData.bloodValues.BloodValue import BloodValue


class PCT(BloodValue):
    def __init__(self, data):
        BloodValue.__init__(self, data, "PCT")