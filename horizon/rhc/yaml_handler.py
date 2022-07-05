import yaml


class YamlParser:

    @classmethod
    def load(cls, yaml_file):
        with open(yaml_file, 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # parse required fields
        required_fields = ['constraints', 'costs'] # 'solver'
        required_fields_dict = cls._parse_required_fields(parsed_yaml, required_fields)

        # parse tasks
        field_names = {'constraints': 'constraint', 'costs': 'residual'}
        task_list = list()
        for task_name, task_desc in parsed_yaml.items():
            task_desc['name'] = task_name

            for field in required_fields:
                if task_name in required_fields_dict[field]:
                    task_desc['fun_type'] = field_names[field]
                    task_list.append(task_desc)

        return task_list #, required_fields_dict['solver']

    @staticmethod
    def _parse_required_fields(parsed_yaml, required_fields):

        required_fields_dict = dict()
        for field in required_fields:
            if field in parsed_yaml:
                required_fields_dict[field] = parsed_yaml.pop(field)
            else:
                raise ValueError(f"'{field}' field is required.")

        return required_fields_dict


    @staticmethod
    def resolve(task_desc, shortcuts):
        '''
        substitute values in the task description with the corresponding values in a shortcut dictionary
        shortcut: dict of dict ({task_key: {old_value: new_value}})
        '''
        task_desc_copy = task_desc.copy()
        # for each key, value in the task_description
        for k, v in task_desc_copy.items():
            # if the key of the task description is in the dictionary keys
            # if the value of the task description is in the dictionary values,
            # then substitute
            if isinstance(v, str) and k in shortcuts and v in shortcuts[k]:
                task_desc_copy[k] = shortcuts[k][v]

        return task_desc_copy


if __name__ == '__main__':

    yaml_file = "../playground/spot/task_interface_playground/config_walk.yaml"
    task_list = YamlParser.load(yaml_file)
    for task in task_list:
        print(task)
    shortcuts = {
        'nodes': {'final': [5], 'all': 'thesingleladies'},
        'indices': {'all': 'stars'}
    }
    print("--------------------------------------")
    task_list_new = YamlParser.resolve(task_list[0], shortcuts)

    print(task_list_new)
