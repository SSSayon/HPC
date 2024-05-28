import re

input_file = f"raw_trunc.txt"
output_file = f"parse_trunc.txt"

with open(input_file, 'r') as file:
    data = file.read()

# Define the regex patterns
step_pattern = re.compile(r'TRUNCATED_STEP=(\d+)')
opt_pattern = re.compile(r'\[.*?\] SpMMTest.opt_performance\s*\[.*?\]\s*time = ([\d.]+)')

# Find all matches for TRUNCATED_STEP and opt_performance time
steps = step_pattern.findall(data)
opt_times = opt_pattern.findall(data)

# Multiply the opt_performance time by 1000000
opt_times = [f'{float(time) * 1e6:.2f}' for time in opt_times]

# Check that we have the same number of steps and opt times
assert len(steps) == len(opt_times), "Mismatch in number of TRUNCATED_STEP and opt_performance times"

# Write the results to a new text file
with open(output_file, 'w') as file:
    for step, time in zip(steps, opt_times):
        file.write(f'TRUNCATED_STEP={step}, opt_time={time}\n')
