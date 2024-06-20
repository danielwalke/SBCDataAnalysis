import matplotlib.pyplot as plt


class BloodValue:
    def __init__(self, data, column):
        print(f"Length of {column} values above 100: " + str(len(data.query(f"{column} > 100", engine='python'))))
        # data = data.query(f"{column} < 100", engine='python')
        data = data.query(f"~{column}.isnull()", engine='python')
        self.blood_value = data["HGB"]
        sepsis_data = data[data["Diagnosis"].str.contains("Sepsis")]
        control_data = data[data["Diagnosis"].str.contains("Control")]
        print(len(sepsis_data.query(f"{column} > 100", engine='python')))
        print(len(control_data.query(f"{column} > 100", engine='python')))
        self.blood_value_sepsis = sepsis_data[column]
        self.blood_value_control = control_data[column]
        self.column = column

    def violin_plot(self):
        fig, ax = plt.subplots()

        # Create a plot
        ax.violinplot([self.blood_value_sepsis, self.blood_value_control], showmedians=True)
        ax.set_title('Sepsis vs Control')
        # ax2.violinplot(self.blood_value_control, showmedians=True)
        # ax2.set_title('Control')
        fig.suptitle(self.column)
        fig.supxlabel('frequency')
        fig.supylabel('value')
        plt.show()