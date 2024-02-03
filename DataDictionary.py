from MLData import MLData

class DataDictionary:
    def __init__(self):
        self.datanames = ["SoyBean"]

    def dataobject(self, name):
        return MLData(*self.metadata(name))

    def metadata(self, name):
        if name == "ForestFires": return self.forestfires()
        if name == "StudentPerformance": return self.studentperformance()

    def forestfires(self):
        name = "ForestFires"
        file = 'raw_data/forestfires.csv'
        columns = [
          'X', # For Forest Fires
          'Y',
          'Month',
          'Day',
          'FFMC',
          'DMC',
          'DC',
          'ISI',
          'Temp',
          'RH',
          'Wind',
          'Rain',
          'Area'  #Target
        ]
        replace = None
        target_name = 'Area'
        classification = False
        header = False
        return (name, file, columns, target_name, replace, classification, header)

    def studentperformance(self):
        name = "StudentPerformance"
        file = "raw_data/Student_Performance.csv"
        columns = [
            "Hours Studied",
            "Previous Scores",
            "Extracurricular Activities",
            "Sleep Hours",
            "Sample Question Papers Practiced",
            "Performance Index" #Target
        ]
        replace = None
        target_name = "Performance Index"
        classification = False
        header = True
        return (name, file, columns, target_name, replace, classification, header)