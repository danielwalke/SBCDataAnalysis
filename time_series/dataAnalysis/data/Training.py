from dataAnalysis.data.Filter import Filter


class Training(Filter):
    def __init__(self, data):
        leipzig_training_data = data.query("Center == 'Leipzig' & Set == 'Training'")
        Filter.__init__(self, leipzig_training_data)
