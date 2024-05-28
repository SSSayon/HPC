import os
import re

data_dir = 'data'

config_files = [f for f in os.listdir(data_dir) if f.endswith('.config')]

data = []

pattern = re.compile(r'(\d+)\s+(\d+)')

for config_file in config_files:
    dataset_name = config_file.replace('.config', '')
    with open(os.path.join(data_dir, config_file), 'r') as file:
        content = file.read().strip()
        match = pattern.match(content)
        if match:
            data.append(f"{dataset_name} {match.group(1)} {match.group(2)}")

with open('dataset.txt', 'w') as file:
    for line in data:
        file.write(line + '\n')
