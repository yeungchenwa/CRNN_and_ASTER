from importlib.abc import Loader
import yaml

class Yaml:
    def __init__(self, yaml_file):
        super(Yaml, self).__init__()
        self.yaml_file = yaml_file

    def read_yaml(self):
        """read yaml file"""
        with open(self.yaml_file, 'r', encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return config
    
    def writeYaml(self,dict):
        '''write yaml file'''
        with open(self.yamlFile,'a',encoding="utf-8") as f:
            try:
                yaml.dump(data=dict,stream=f,encoding="utf-8",allow_unicode=True)
            except Exception as e:
                print(e)

file = Yaml('./configs/CRNN.yaml')
config = file.read_yaml()
print(config['task'])