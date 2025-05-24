import argparse
import main

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="示例：读取命令行参数")

# 添加参数
parser.add_argument('--work_path', type=str, default='/mnt/e/data', help='工作目录')
parser.add_argument('--h5ad_path', type=str, default='./网络熵/singlecell/GSE200981/filter_data.h5ad', help='单细胞h5ad路径')
parser.add_argument('--net_path', type=str, default='./PIN/STRING/string.csv', help='PPI路径')
parser.add_argument('--device', type=str, default='GPU', help='计算设备')
parser.add_argument('--save_path', type=str, default='None', help='计算设备')

# 解析参数
args = parser.parse_args()

# 使用参数
work_space = args.work_path
h5ad_path = args.h5ad_path
net_path = args.net_path
device = args.device
save_path = args.save_path

main.run_main(work_space,h5ad_path,net_path,device,save_path)