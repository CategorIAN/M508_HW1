from MLData import MLData

class DataDictionary:
    def __init__(self):
        self.datanames = ["SoyBean"]

    def dataobject(self, name):
        return MLData(*self.metadata(name))

    def metadata(self, name):
        if name == "SoyBean": return self.soybean()
        if name == "ForestFires": return self.forestfires()


    def soybean(self):
        name = "SoyBean"
        file = 'raw_data/soybean-small.csv'
        columns =  ['Date',  # For Soy Bean
         'Plant-Stand',
         'Precip',
         'Temp',
         'Hail',
         'Crop-Hist',
         'Area-Damaged',
         'Severity',
         'Seed-TMT',
         'Germination',
         'Plant-Growth',
         'Leaves',
         'Leafspots-Halo',
         'Leafspots-Marg',
         'Leafspot-Size',
         'Leaf-Shread',
         'Leaf-Malf',
         'Leaf-Mild',
         'Stem',
         'Lodging',
         'Stem-Cankers',
         'Canker-Lesion',
         'Fruiting-Bodies',
         'External Decay',
         'Mycelium',
         'Int-Discolor',
         'Sclerotia',
         'Fruit-Pods',
         'Fruit Spots',
         'Seed',
         'Mold-Growth',
         'Seed-Discolor',
         'Seed-Size',
         'Shriveling',
         'Roots',
         'Class'  #Target
         ]
        replace = None
        target_name = 'Class'
        classification = True
        return (name, file, columns, target_name, replace, classification)

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
        return (name, file, columns, target_name, replace, classification)