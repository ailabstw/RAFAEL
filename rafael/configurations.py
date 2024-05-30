from ruamel.yaml import YAML
from ruamel.yaml.constructor import SafeConstructor


# YamlLoader needs refactoring towards a set of functions rather than a class
class YamlLoader:
    def __init__(self):
        SafeConstructor.add_constructor(u'tag:yaml.org,2002:map', self.yaml_map)
        self.yml = YAML(typ='safe')

    @staticmethod
    def yaml_map(self, node):
        data = []
        yield data
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=True)
            val = self.construct_object(value_node, deep=True)
            data.append((key, val))

    @staticmethod
    def format_pipeline(content):
        wrapped = []
        for uc, methods in content:
            for meth in methods:
                wrapped.append((uc, meth))
        return wrapped

    @staticmethod
    def format_dataloader(content):
        wrapped = []
        for uc, params in content:
            wrapped.append((uc, dict(params)))
        return wrapped

    def format(self, config):
        if any(map(lambda x: x[0] == "config", config)):
            formatted = {}
            for key, content in config:
                if key == 'pipeline':
                    formatted[key] = self.format_pipeline(content)
                elif key in ("clients", "servers", "compensators"):
                    formatted[key] = list(map(dict, content))
                elif key == 'dataloader':
                    formatted[key] = self.format_dataloader(content)
                else:
                    formatted[key] = dict(content)
            return formatted
        else:
            return {key:dict(content) for key, content in config}

    @classmethod
    def load(cls, yml):
        return cls().format(cls().yml.load(yml))

def load_yml(yml_path):
    with open(yml_path, "r") as stream:
        config = YamlLoader.load(stream)
    return config