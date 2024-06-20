from dataAnalysis.data.Filter import Filter


class MIMIC(Filter):
    def __init__(self, data):
        mimic_validation_data = data.query("Center == 'MIMIC-IV' & Set == 'Validation'")
        Filter.__init__(self, mimic_validation_data)
