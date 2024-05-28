import re

with open('raw_1_trunc.txt', 'r') as file:
    data = file.read()

dset_pattern = re.compile(r'dset = "(.*?)"')
cusparse_pattern = re.compile(r'\[.*?\] SpMMTest.cusparse_performance\s*\[.*?\]\s*time = ([\d.]+)')
opt_pattern = re.compile(r'\[.*?\] SpMMTest.opt_performance\s*\[.*?\]\s*time = ([\d.]+)')

dsets = dset_pattern.findall(data)
cusparse_times = cusparse_pattern.findall(data)
opt_times = opt_pattern.findall(data)

cusparse_times = [f'{float(time) * 1e6:.2f}' for time in cusparse_times]
opt_times = [f'{float(time) * 1e6:.2f}' for time in opt_times]

data_dict = {dset: (cusparse_time, opt_time) for dset, cusparse_time, opt_time in zip(dsets, cusparse_times, opt_times)}
output_order = [
    "arxiv", "collab", "citation", "ddi", "protein", "ppa", "reddit.dgl", 
    "products", "youtube", "amazon_cogdl", "yelp", "wikikg2", "am"
]

dset_width = max(len(dset) for dset in output_order) + 4  # +4 for some padding
cusparse_width = max(len(cusparse_time) for cusparse_time in cusparse_times) + 4
opt_width = max(len(opt_time) for opt_time in opt_times) + 4

with open('parsed_data_1_trunc.txt', 'w') as file:
    file.write(f'{"dataset".ljust(dset_width)} {"cusparse (us)".ljust(cusparse_width)} {"opt (us)".ljust(opt_width)}\n')
    for dset in output_order:
        cusparse_time, opt_time = data_dict[dset]
        file.write(f'{dset.ljust(dset_width)} {cusparse_time.ljust(cusparse_width)} {opt_time.ljust(opt_width)}\n')
