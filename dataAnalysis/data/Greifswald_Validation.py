from dataAnalysis.data.Filter import Filter


class GreifswaldValidation(Filter):
    def __init__(self, data):
        greifswald_validation_data = data.query("Center == 'Greifswald' & Set == 'Validation'")
        Filter.__init__(self, greifswald_validation_data)
