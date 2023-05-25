class TargetIcu:
    def __init__(self, data):
        data_without_nan_target_icus = data[~data['TargetIcu'].isnull()]
        self.target_icus = data_without_nan_target_icus["TargetIcu"].unique()
