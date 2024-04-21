import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = """
naive 32 1 Exec-time: 9.6549 ms                                                                                                                                       
shared_memory 32 1 Exec-time: 10.1296 ms                                                                                                                                    
naive 32 2 Exec-time: 7.61549 ms                                                                                                                                            
shared_memory 32 2 Exec-time: 6.54308 ms                                                                                                                                    
naive 32 3 Exec-time: 7.62618 ms                                                                                                                                            
shared_memory 32 3 Exec-time: 5.82668 ms
naive 32 4 Exec-time: 7.64226 ms
shared_memory 32 4 Exec-time: 5.53416 ms
naive 32 5 Exec-time: 7.69929 ms
shared_memory 32 5 Exec-time: 5.80498 ms
naive 32 6 Exec-time: 7.70586 ms
shared_memory 32 6 Exec-time: 5.72863 ms
naive 32 7 Exec-time: 7.69232 ms
shared_memory 32 7 Exec-time: 5.59486 ms
naive 32 8 Exec-time: 7.70245 ms
shared_memory 32 8 Exec-time: 5.49189 ms
naive 32 9 Exec-time: 7.72213 ms
shared_memory 32 9 Exec-time: 5.54273 ms
naive 32 10 Exec-time: 7.75607 ms
shared_memory 32 10 Exec-time: 5.63446 ms
naive 32 11 Exec-time: 7.8517 ms
shared_memory 32 11 Exec-time: 5.98292 ms
naive 32 12 Exec-time: 7.77013 ms
shared_memory 32 12 Exec-time: 5.64665 ms
naive 32 13 Exec-time: 8.01355 ms
shared_memory 32 13 Exec-time: 6.22466 ms
naive 32 14 Exec-time: 7.85587 ms
shared_memory 32 14 Exec-time: 5.88883 ms
naive 32 15 Exec-time: 7.83356 ms
shared_memory 32 15 Exec-time: 5.68719 ms
naive 32 16 Exec-time: 7.78897 ms
shared_memory 32 16 Exec-time: 5.4996 ms
naive 32 17 Exec-time: 8.19757 ms
shared_memory 32 17 Exec-time: 6.35012 ms
naive 32 18 Exec-time: 8.03361 ms
shared_memory 32 18 Exec-time: 6.0316 ms
naive 32 19 Exec-time: 8.02569 ms
shared_memory 32 19 Exec-time: 5.95293 ms
naive 32 20 Exec-time: 7.93004 ms
shared_memory 32 20 Exec-time: 5.76848 ms
naive 32 21 Exec-time: 7.92963 ms
shared_memory 32 21 Exec-time: 5.7162 ms
naive 32 22 Exec-time: 8.40936 ms
shared_memory 32 22 Exec-time: 7.07015 ms
naive 32 23 Exec-time: 8.37462 ms
shared_memory 32 23 Exec-time: 6.81402 ms
naive 32 24 Exec-time: 8.3143 ms
shared_memory 32 24 Exec-time: 6.8199 ms
naive 32 25 Exec-time: 8.32229 ms
shared_memory 32 25 Exec-time: 6.56487 ms
naive 32 26 Exec-time: 8.22698 ms
shared_memory 32 26 Exec-time: 6.48723 ms
naive 32 27 Exec-time: 7.99165 ms
shared_memory 32 27 Exec-time: 6.22684 ms
naive 32 28 Exec-time: 7.99382 ms
shared_memory 32 28 Exec-time: 6.29176 ms
naive 32 29 Exec-time: 7.99935 ms
shared_memory 32 29 Exec-time: 6.00086 ms
naive 32 30 Exec-time: 7.95858 ms
shared_memory 32 30 Exec-time: 6.07832 ms
naive 32 31 Exec-time: 7.94616 ms
shared_memory 32 31 Exec-time: 5.91429 ms
naive 32 32 Exec-time: 7.91295 ms
shared_memory 32 32 Exec-time: 5.99207 ms
naive 64 1 Exec-time: 7.61334 ms
shared_memory 64 1 Exec-time: 6.45561 ms
naive 64 2 Exec-time: 7.6343 ms
shared_memory 64 2 Exec-time: 5.51423 ms
naive 64 3 Exec-time: 7.67956 ms
shared_memory 64 3 Exec-time: 5.5653 ms
naive 64 4 Exec-time: 7.68717 ms
shared_memory 64 4 Exec-time: 5.32023 ms
naive 64 5 Exec-time: 7.74445 ms
shared_memory 64 5 Exec-time: 5.59893 ms
naive 64 6 Exec-time: 7.77179 ms
shared_memory 64 6 Exec-time: 5.61611 ms
naive 64 7 Exec-time: 7.88524 ms
shared_memory 64 7 Exec-time: 5.88723 ms
naive 64 8 Exec-time: 7.81628 ms
shared_memory 64 8 Exec-time: 5.407 ms
naive 64 9 Exec-time: 8.08105 ms
shared_memory 64 9 Exec-time: 6.09354 ms
naive 64 10 Exec-time: 7.96521 ms                                                                                                                                           
shared_memory 64 10 Exec-time: 5.69933 ms                                                                                                                                   
naive 64 11 Exec-time: 8.50713 ms                                                                                                                                           
shared_memory 64 11 Exec-time: 7.09308 ms                                                                                                                                   
naive 64 12 Exec-time: 8.4148 ms
shared_memory 64 12 Exec-time: 6.72138 ms
naive 64 13 Exec-time: 8.2899 ms
shared_memory 64 13 Exec-time: 6.28401 ms
naive 64 14 Exec-time: 8.02314 ms
shared_memory 64 14 Exec-time: 6.03858 ms
naive 64 15 Exec-time: 8.00696 ms
shared_memory 64 15 Exec-time: 5.89134 ms
naive 64 16 Exec-time: 7.95456 ms
shared_memory 64 16 Exec-time: 5.77371 ms
naive 96 1 Exec-time: 7.61965 ms
shared_memory 96 1 Exec-time: 5.84236 ms
naive 96 2 Exec-time: 7.65463 ms
shared_memory 96 2 Exec-time: 5.48378 ms
naive 96 3 Exec-time: 7.67149 ms
shared_memory 96 3 Exec-time: 5.31839 ms
naive 96 4 Exec-time: 7.73689 ms
shared_memory 96 4 Exec-time: 5.48797 ms
naive 96 5 Exec-time: 7.79727 ms
shared_memory 96 5 Exec-time: 5.55982 ms
naive 96 6 Exec-time: 8.01007 ms
shared_memory 96 6 Exec-time: 5.99579 ms
naive 96 7 Exec-time: 7.88573 ms
shared_memory 96 7 Exec-time: 5.54423 ms
naive 96 8 Exec-time: 8.40608 ms
shared_memory 96 8 Exec-time: 6.64055 ms
naive 96 9 Exec-time: 8.02189 ms
shared_memory 96 9 Exec-time: 6.09084 ms
naive 96 10 Exec-time: 7.99592 ms
shared_memory 96 10 Exec-time: 5.75318 ms
naive 128 1 Exec-time: 7.62821 ms
shared_memory 128 1 Exec-time: 5.61884 ms
naive 128 2 Exec-time: 7.66435 ms
shared_memory 128 2 Exec-time: 5.24336 ms
naive 128 3 Exec-time: 7.74364 ms
shared_memory 128 3 Exec-time: 5.50878 ms
naive 128 4 Exec-time: 7.76569 ms
shared_memory 128 4 Exec-time: 5.29681 ms
naive 128 5 Exec-time: 7.8832 ms
shared_memory 128 5 Exec-time: 5.60223 ms
naive 128 6 Exec-time: 8.30181 ms
shared_memory 128 6 Exec-time: 6.55951 ms
naive 128 7 Exec-time: 8.00047 ms
shared_memory 128 7 Exec-time: 5.92741 ms
naive 128 8 Exec-time: 7.94554 ms
shared_memory 128 8 Exec-time: 5.59808 ms
naive 160 1 Exec-time: 7.64049 ms
shared_memory 160 1 Exec-time: 5.70778 ms
naive 160 2 Exec-time: 7.69635 ms
shared_memory 160 2 Exec-time: 5.39962 ms
naive 160 3 Exec-time: 7.77515 ms
shared_memory 160 3 Exec-time: 5.47784 ms
naive 160 4 Exec-time: 7.83998 ms
shared_memory 160 4 Exec-time: 5.55288 ms
naive 160 5 Exec-time: 8.13818 ms
shared_memory 160 5 Exec-time: 6.38778 ms
naive 160 6 Exec-time: 7.92325 ms
shared_memory 160 6 Exec-time: 5.71061 ms
naive 192 1 Exec-time: 7.66093 ms
shared_memory 192 1 Exec-time: 5.66992 ms
naive 192 2 Exec-time: 7.73119 ms
shared_memory 192 2 Exec-time: 5.41419 ms
naive 192 3 Exec-time: 7.9479 ms
shared_memory 192 3 Exec-time: 5.9083 ms
naive 192 4 Exec-time: 8.15952 ms
shared_memory 192 4 Exec-time: 6.51154 ms
naive 192 5 Exec-time: 7.95247 ms
shared_memory 192 5 Exec-time: 5.61443 ms
naive 224 1 Exec-time: 7.64659 ms
shared_memory 224 1 Exec-time: 5.52284 ms
naive 224 2 Exec-time: 7.79547 ms
shared_memory 224 2 Exec-time: 5.65093 ms
naive 224 3 Exec-time: 7.81677 ms
shared_memory 224 3 Exec-time: 5.37665 ms
naive 224 4 Exec-time: 7.91669 ms
shared_memory 224 4 Exec-time: 5.94417 ms
naive 256 1 Exec-time: 7.66617 ms
shared_memory 256 1 Exec-time: 5.46993 ms
naive 256 2 Exec-time: 7.76395 ms
shared_memory 256 2 Exec-time: 5.26065 ms
naive 256 3 Exec-time: 8.10612 ms
shared_memory 256 3 Exec-time: 6.32311 ms
naive 256 4 Exec-time: 7.91895 ms                                                                                                                                           
shared_memory 256 4 Exec-time: 5.34964 ms                                                                                                                                   
naive 288 1 Exec-time: 7.6622 ms                                                                                                                                            
shared_memory 288 1 Exec-time: 5.49191 ms                                                                                                                                   
naive 288 2 Exec-time: 7.92544 ms                                                                                                                                           
shared_memory 288 2 Exec-time: 5.86523 ms                                                                                                                                   
naive 288 3 Exec-time: 7.97438 ms                                                                                                                                           
shared_memory 288 3 Exec-time: 5.89619 ms                                                                                                                                   
naive 320 1 Exec-time: 7.70398 ms                                                                                                                                           
shared_memory 320 1 Exec-time: 5.59603 ms                                                                                                                                   
naive 320 2 Exec-time: 7.86675 ms
shared_memory 320 2 Exec-time: 5.51238 ms
naive 320 3 Exec-time: 7.92881 ms
shared_memory 320 3 Exec-time: 5.42176 ms
naive 352 1 Exec-time: 7.77521 ms
shared_memory 352 1 Exec-time: 5.90592 ms
naive 352 2 Exec-time: 8.10898 ms
shared_memory 352 2 Exec-time: 6.5482 ms
naive 384 1 Exec-time: 7.73361 ms
shared_memory 384 1 Exec-time: 5.63142 ms
naive 384 2 Exec-time: 8.05234 ms
shared_memory 384 2 Exec-time: 6.16834 ms
naive 416 1 Exec-time: 7.87557 ms
shared_memory 416 1 Exec-time: 6.17452 ms
naive 416 2 Exec-time: 8.01365 ms
shared_memory 416 2 Exec-time: 5.91316 ms
naive 448 1 Exec-time: 7.8102 ms
shared_memory 448 1 Exec-time: 5.8471 ms
naive 448 2 Exec-time: 7.90284 ms
shared_memory 448 2 Exec-time: 5.66267 ms
naive 480 1 Exec-time: 7.74896 ms
shared_memory 480 1 Exec-time: 5.64222 ms
naive 480 2 Exec-time: 7.84932 ms
shared_memory 480 2 Exec-time: 5.4831 ms
naive 512 1 Exec-time: 7.74085 ms
shared_memory 512 1 Exec-time: 5.50976 ms
naive 512 2 Exec-time: 7.87412 ms
shared_memory 512 2 Exec-time: 5.22932 ms
naive 544 1 Exec-time: 7.97807 ms
shared_memory 544 1 Exec-time: 6.27975 ms
naive 576 1 Exec-time: 7.92897 ms
shared_memory 576 1 Exec-time: 6.06185 ms
naive 608 1 Exec-time: 7.93454 ms
shared_memory 608 1 Exec-time: 6.01422 ms
naive 640 1 Exec-time: 7.86967 ms
shared_memory 640 1 Exec-time: 5.82734 ms
naive 672 1 Exec-time: 7.82395 ms
shared_memory 672 1 Exec-time: 5.63915 ms
naive 704 1 Exec-time: 8.13373 ms
shared_memory 704 1 Exec-time: 6.82694 ms
naive 736 1 Exec-time: 8.0601 ms
shared_memory 736 1 Exec-time: 6.56047 ms
naive 768 1 Exec-time: 8.0651 ms
shared_memory 768 1 Exec-time: 6.5532 ms
naive 800 1 Exec-time: 7.97572 ms
shared_memory 800 1 Exec-time: 6.3674 ms
naive 832 1 Exec-time: 8.012 ms
shared_memory 832 1 Exec-time: 6.30066 ms
naive 864 1 Exec-time: 7.95589 ms
shared_memory 864 1 Exec-time: 6.28185 ms
naive 896 1 Exec-time: 7.91665 ms
shared_memory 896 1 Exec-time: 6.05369 ms
naive 928 1 Exec-time: 7.84449 ms
shared_memory 928 1 Exec-time: 5.99153 ms
naive 960 1 Exec-time: 7.91389 ms
shared_memory 960 1 Exec-time: 5.98065 ms
naive 992 1 Exec-time: 7.99591 ms
shared_memory 992 1 Exec-time: 5.9717 ms
naive 1024 1 Exec-time: 7.84316 ms
shared_memory 1024 1 Exec-time: 5.7537 ms
"""

# 解析数据并存储在 DataFrame 中
data = data.split('\n')[1:-1]
data_naive = [line.split() for line in data if line.startswith('naive')]  # 以 'naive' 开头的行
data_share = [line.split() for line in data if line.startswith('shared_memory')]  # 以 'shared_memory' 开头的行
df_naive = pd.DataFrame(data_naive, columns=['Type', 'block_size_x', 'block_size_y', 'Exec', 'Time', 'Unit'])  # DataFrame
df_share = pd.DataFrame(data_share, columns=['Type', 'block_size_x', 'block_size_y', 'Exec', 'Time', 'Unit'])
df_naive['block_size_x'] = df_naive['block_size_x'].astype(int)
df_naive['block_size_y'] = df_naive['block_size_y'].astype(int)
df_naive['Time'] = df_naive['Time'].astype(float)
df_share['block_size_x'] = df_share['block_size_x'].astype(int)
df_share['block_size_y'] = df_share['block_size_y'].astype(int)
df_share['Time'] = df_share['Time'].astype(float)

pivot_table_naive = df_naive.pivot(index='block_size_y', columns='block_size_x', values='Time')
pivot_table_share = df_share.pivot(index='block_size_y', columns='block_size_x', values='Time')
diff_table = pivot_table_naive - pivot_table_share

# plot 
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table_naive, annot=False, fmt=".2f", cmap='viridis')
plt.title('Execution Time of Different Block Size (naive)')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table_share, annot=False, fmt=".2f", cmap='viridis')
plt.title('Execution Time of Different Block Size (shared_memory)')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(diff_table, annot=False, fmt=".2f", cmap='viridis')
plt.title('Difference in Execution Time Between Naive and Shared Memory')
plt.show()
