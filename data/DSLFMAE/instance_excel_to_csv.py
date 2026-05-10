"""
excel算例特定数据格式转为csv算例特定数据格式
"""
import openpyxl, os, csv
from utilities import DataComputer


class DataCsvWrite(DataComputer):
    """
    excel数据读取: data
    """
    def __init__(self, path, file_name, book_name_xlsx, input_data, machine_class_data, time_machine_class_data, class_machine_data):
        super().__init__(book_name_xlsx, input_data, machine_class_data, time_machine_class_data, class_machine_data)
        # 文件写入位置
        self.path = path  # 文件存储路径
        self.file_name = file_name  # 文件名
        # 问题特点
        self.DDT = 1.0  # 交期紧急度
        self.machine_count = self.machine_number  # 机器数
        self.machine_tuple = tuple(self.machine_list)  # 机器元组
        self.order_count = 1  # 新到达的订单总数
        self.order_tuple = tuple(s for s in range(self.order_count))  # 订单元组
        self.kind_count = self.number_job_kind  # 工件类型数
        self.kind_tuple = tuple(r for r in range(self.kind_count))
        self.task_kind = self.number_task
        # 工序元组索引，可选机器元组索引，在各机器上的加工时间，订单中各类型工件数量， 订单到达时间， 订单交期时间
        self.task_r_dict, self.machine_rj_dict, self.kind_task_m_dict, self.time_rjm_dict, self.count_sr_dict, \
        self.time_arrive_s_dict, self.time_delivery_s_dict, self.kind_task_tuple, self.time_mrj_dict = self.read_data()

    def read_data(self):
        """读取csv文件相关属性"""
        # 由类到工序的索引
        kind_task_class = [[i, j] for k in self.class_list for i in range(self.kind_count) for j in range(self.task_kind[i]) if self.class_kind[i][j] == k]
        kind_task_class_dict = {k: tuple(kind_task_class[k]) for k in self.class_list}  # 由类号-(r, j)的索引
        class_kind_task_dict = {value: key for key, value in kind_task_class_dict.items()}
        # 生成写入数据
        task_r_dict = {r: tuple(j for j in range(self.task_kind[r])) for r in self.kind_tuple}  # [r]对应工序元组
        kind_task_tuple = tuple((r, j) for r in self.kind_tuple for j in task_r_dict[r])  # 工序类型元组
        machine_rj_dict = {r: {j: tuple(self.machine_class[class_kind_task_dict[(r, j)]]) for j in task_r_dict[r]} for r in self.kind_tuple}  # [r][j]可选机器元组
        time_rjm_dict = {
            r: {j: {m: self.time_machine_class[class_kind_task_dict[(r, j)]][self.machine_class[class_kind_task_dict[(r, j)]].index(m)] for m in machine_rj_dict[r][j]}
                for j in task_r_dict[r]} for r in self.kind_tuple}  # [r][j][m]加工时间
        kind_task_m_dict = {m: tuple((r, j) for r in self.kind_tuple for j in task_r_dict[r] if m in machine_rj_dict[r][j]) for m in self.machine_tuple}
        time_mrj_dict = {m: {rj: time_rjm_dict[rj[0]][rj[1]][m] for rj in kind_task_m_dict[m]} for m in self.machine_tuple}
        # 各工序加工时间均值
        time_rj_dict = {r: {j: sum([time_rjm_dict[r][j][m] for m in machine_rj_dict[r][j]]) / len(machine_rj_dict[r][j]) for j in task_r_dict[r]} for r in
                        self.kind_tuple}
        # 订单信息
        count_sr_dict = {s: tuple(self.order_list[s][r] for r in self.kind_tuple) for s in self.order_tuple}  # [s][r]工件类型的数量
        time_gap_s_dict = {
            s: sum([time_rj_dict[r][j] * count_sr_dict[s][r] for r in self.kind_tuple for j in task_r_dict[r]]) * self.DDT / (self.machine_count * 2) for s in
            self.order_tuple}  # 各订单交期-到达时间差值
        time_interval_list = [0 for s in range(self.order_count - 1)]  # 各订单的间隔时间
        time_interval_list.insert(0, 0)
        time_arrive_s_dict = {s: int(sum(time_interval_list[:s + 1])) for s in self.order_tuple}  # 各订单的到达时间
        time_delivery_list = [time_arrive_s_dict[s] + time_gap_s_dict[s] for s in self.order_tuple]
        time_delivery_list.sort()
        time_delivery_s_dict = {s: int(time_delivery_list[s]) for s in self.order_tuple}  # 各订单的交期时间

        return task_r_dict, machine_rj_dict, kind_task_m_dict, time_rjm_dict, count_sr_dict, time_arrive_s_dict, \
               time_delivery_s_dict, kind_task_tuple, time_mrj_dict

    def write_file(self):
        """写入csv文件"""
        file_path = self.path
        os.makedirs(os.path.join(file_path, self.file_name), exist_ok=True)  # 新建实例文件夹
        file_csv = {'based_data.csv': ['kind_count', 'machine_count', 'order_count', 'DDT'],
                    'process_data.csv': ['kind', 'task', 'machine_selectable', 'process_time'],
                    'order_data.csv': ['order', 'time_arrive', 'time_delivery', 'kind_number']}

        for csv_name, header in file_csv.items():
            data_file = os.path.join(file_path, self.file_name, csv_name)
            with open(data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                rows = []  # 初始化写入数据
                if csv_name == 'based_data.csv':
                    rows.append([self.kind_count, self.machine_count, self.order_count, self.DDT])
                elif csv_name == 'process_data.csv':
                    for r in self.kind_tuple:
                        for j in self.task_r_dict[r]:
                            time_machine_tuple = tuple(self.time_rjm_dict[r][j][m] for m in self.machine_rj_dict[r][j])
                            rows.append([r, j, self.machine_rj_dict[r][j], time_machine_tuple])
                else:
                    for s in self.order_tuple:
                        rows.append([s, self.time_arrive_s_dict[s], self.time_delivery_s_dict[s], self.count_sr_dict[s]])
                writer.writerows(rows)
        print("写入完成")


# 测试
if __name__ == '__main__':
    file_name_list = \
        ['M10R5N5', 'M10R5N10', 'M10R5N15', 'M10R10N5', 'M10R10N10', 'M10R10N15', 'M10R15N5', 'M10R15N10', 'M10R15N15',
         'M15R5N5', 'M15R5N10', 'M15R5N15', 'M15R10N5', 'M15R10N10', 'M15R10N15', 'M15R15N5', 'M15R15N10', 'M15R15N15',
         'M20R5N5', 'M20R5N10', 'M20R5N15', 'M20R10N5', 'M20R10N10', 'M20R10N15', 'M20R15N5', 'M20R15N10', 'M20R15N15']
    path_save = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/data/DSLFMAE'
    for file_name in file_name_list:
        path = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/algorithms/data_process/instance_data/{}.xlsx'.format(file_name)
        read_object = DataCsvWrite(path=path_save, file_name=file_name, book_name_xlsx=path, input_data='算例基础数据', machine_class_data='各类可选加工机器',
                                   time_machine_class_data='各类在各机器加工时间', class_machine_data='各机器可加工类')
        read_object.write_file()
        print('写入文件：', file_name)