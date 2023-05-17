import os
import json

with open('config.json') as config_file:
    config = json.load(config_file)

data_dir = config['data_dir']
arg_list = []

for trace_path in os.listdir(data_dir + 'traffic\\undefence'):
    arg_list.append([data_dir + 'traffic\\undefence', trace_path])

base_dir = os.getcwd()

for arg in arg_list:
    old_name = base_dir + '\\' + arg[0] + '\\' + arg[1]
    new_name = base_dir + '\\' + arg[0] + '\\' + str(int(arg[1].split('-')[0]) + 50) + '-' + arg[1].split('-')[1]
    os.rename(old_name, new_name)
