from dataAnalysis.data.Filter import Filter


class Validation(Filter):
    def __init__(self, data):
        leipzig_validation_data = data.query("Center == 'Leipzig' & Set == 'Validation'")
        Filter.__init__(self, leipzig_validation_data)
