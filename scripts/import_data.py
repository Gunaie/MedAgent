import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import os


class BuildGraph():
    def __init__(self, data_dir: str = None):
        """
        初始化图构建器。

        Args:
            data_dir: neo4j_data 目录的绝对或相对路径。
                     为 None 时自动推导（假设本文件位于 scripts/ 下）。
        """
        if data_dir is None:
            # 自动推导项目根目录：scripts/import_data.py -> 向上两级
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            data_dir = os.path.join(project_root, "neo4j_data")

        self.data_dir = os.path.abspath(data_dir)
        self.medical_json_path = os.path.join(self.data_dir, "medical.json")
        self.nodes_dir = os.path.join(self.data_dir, "nodes")
        self.relations_dir = os.path.join(self.data_dir, "relations")

        self.nodes = defaultdict(list)
        self.relations = defaultdict(list)
        self.parse_raw_data()

    def parse_raw_data(self):
        if not os.path.exists(self.medical_json_path):
            raise FileNotFoundError(
                f"找不到医疗数据文件: {self.medical_json_path}\n"
                f"请确保 {self.data_dir}/medical.json 存在。"
            )

        with open(self.medical_json_path, encoding='utf-8') as file:
            lines = file.readlines()
            for line in tqdm(lines, desc='parse_data'):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] 跳过损坏的 JSON 行: {e}")
                    continue

                disease_name = row['name']

                # 疾病节点
                disease_node = {
                    'name': disease_name,
                    'desc': row.get('desc', ''),
                    'prevent': row.get('prevent', ''),
                    'cause': row.get('cause', ''),
                    'yibao_status': row.get('yibao_status', ''),
                    'get_prob': row.get('get_prob', ''),
                    'get_way': row.get('get_way', ''),
                    'cure_lasttime': row.get('cure_lasttime', ''),
                    'cured_prob': row.get('cured_prob', ''),
                    'cost_money': row.get('cost_money', ''),
                }
                if disease_node not in self.nodes['Disease']:
                    self.nodes['Disease'].append(disease_node)

                # 分类节点
                for category_name in row.get('category', []):
                    category_node = {'name': category_name}
                    if category_node not in self.nodes['Category']:
                        self.nodes['Category'].append(category_node)
                    rel = (disease_name, category_name)
                    if rel not in self.relations['DISEASE_CATEGORY']:
                        self.relations['DISEASE_CATEGORY'].append(rel)

                # 症状节点
                for symptom_name in row.get('symptom', []):
                    symptom_node = {'name': symptom_name}
                    if symptom_node not in self.nodes['Symptom']:
                        self.nodes['Symptom'].append(symptom_node)
                    rel = (disease_name, symptom_name)
                    if rel not in self.relations['DISEASE_SYMPTOM']:
                        self.relations['DISEASE_SYMPTOM'].append(rel)

                # 并发症
                for acompany_name in row.get('acompany', []):
                    rel = (disease_name, acompany_name)
                    if rel not in self.relations['DISEASE_ACOMPANY']:
                        self.relations['DISEASE_ACOMPANY'].append(rel)

                # 科室
                for department_name in row.get('cure_department', []):
                    department_node = {'name': department_name}
                    if department_node not in self.nodes['Department']:
                        self.nodes['Department'].append(department_node)
                    rel = (disease_name, department_name)
                    if rel not in self.relations['DISEASE_DEPARTMENT']:
                        self.relations['DISEASE_DEPARTMENT'].append(rel)

                # 治疗方法
                for cureway_name in row.get('cure_way', []):
                    cureway_node = {'name': cureway_name}
                    if cureway_node not in self.nodes['Cureway']:
                        self.nodes['Cureway'].append(cureway_node)
                    rel = (disease_name, cureway_name)
                    if rel not in self.relations['DISEASE_CUREWAY']:
                        self.relations['DISEASE_CUREWAY'].append(rel)

                # 检查项
                for check_name in row.get('check', []):
                    check_node = {'name': check_name}
                    if check_node not in self.nodes['Check']:
                        self.nodes['Check'].append(check_node)
                    rel = (disease_name, check_name)
                    if rel not in self.relations['DISEASE_CHECK']:
                        self.relations['DISEASE_CHECK'].append(rel)

                # 药物
                for drug_name in row.get('recommand_drug', []) + row.get('common_drug', []):
                    drug_node = {'name': drug_name}
                    if drug_node not in self.nodes['Drug']:
                        self.nodes['Drug'].append(drug_node)
                    rel = (disease_name, drug_name)
                    if rel not in self.relations['DISEASE_DRUG']:
                        self.relations['DISEASE_DRUG'].append(rel)

                # 食物
                for food_name in row.get('do_eat', []) + row.get('not_eat', []):
                    food_node = {'name': food_name}
                    if food_node not in self.nodes['Food']:
                        self.nodes['Food'].append(food_node)
                for food_name in row.get('do_eat', []):
                    rel = (disease_name, food_name)
                    if rel not in self.relations['DISEASE_DO_EAT']:
                        self.relations['DISEASE_DO_EAT'].append(rel)
                for food_name in row.get('not_eat', []):
                    rel = (disease_name, food_name)
                    if rel not in self.relations['DISEASE_NOT_EAT']:
                        self.relations['DISEASE_NOT_EAT'].append(rel)

                # 菜谱
                for dishes_name in row.get('recommand_eat', []):
                    dishes_node = {'name': dishes_name}
                    if dishes_node not in self.nodes['Dishes']:
                        self.nodes['Dishes'].append(dishes_node)
                    rel = (disease_name, dishes_name)
                    if rel not in self.relations['DISEASE_DISHES']:
                        self.relations['DISEASE_DISHES'].append(rel)

    def dump_nodes(self):
        os.makedirs(self.nodes_dir, exist_ok=True)
        for label, node_list in self.nodes.items():
            df = pd.DataFrame(node_list)
            output_path = os.path.join(self.nodes_dir, f"{label}.csv")
            df.to_csv(output_path, encoding='utf-8', index=False)
            print(f"[DUMP] Nodes/{label}.csv: {len(node_list)} rows")

    def dump_relations(self):
        os.makedirs(self.relations_dir, exist_ok=True)
        for rel, relation_list in self.relations.items():
            df = pd.DataFrame(relation_list, columns=['from', 'to'])
            output_path = os.path.join(self.relations_dir, f"{rel}.csv")
            df.to_csv(output_path, encoding='utf-8', index=False)
            print(f"[DUMP] Relations/{rel}.csv: {len(relation_list)} rows")


if __name__ == '__main__':
    # 默认自动推导路径
    bg = BuildGraph()
    bg.dump_nodes()
    bg.dump_relations()
    print(f"[DONE] 数据已导出到: {bg.data_dir}")