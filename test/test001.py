"""
测试单个模板查询
"""

from neo4j_store import get_medical_graph
graph = get_medical_graph()
print(graph.query_by_template("symptom", "鼻炎"))
print(graph.query_by_template("cure_way", "鼻炎"))
print(graph.query_by_template("check", "高血压"))


"""
[Neo4j] 实际关系类型: {'DISEASE_ACOMPANY', 'DISEASE_NOT_EAT', 'DISEASE_CHECK', 'DISEASE_CUREWAY', 'DISEASE_CATEGORY', 'DISEASE_DISHES', 'DISEASE_DEPARTMENT', 'DISEASE_SYMPTOM', 'DISEASE_DO_EAT', 'DISEASE_DRUG'}
[Neo4j] 药物相关关系: {'DISEASE_DRUG'}
【鼻炎】的症状包括：深侧鼻塞、流鼻涕、鼻粘膜苍白水肿、鼻塞排出脓性或...、上额窦囊肿、鼻出血、鼻粘膜干燥发痒、鼻子糜烂、鼻子上火、鼻酸。
【鼻炎】可用药物：孟鲁司特钠片、匹多莫德口服溶液、马来酸氯苯那敏片、鼻窦炎口服液、匹多莫德片、氨苄西林胶囊、富马酸酮替芬片、盐酸羟甲唑啉喷雾剂、丙酸氟替卡松鼻喷雾剂、益鼻喷雾剂、孟鲁司特钠咀嚼片、盐酸西替利嗪片、藿胆滴丸、富马酸酮替芬胶囊、苍夷滴鼻剂、注射用头孢唑林钠、鼻炎灵片、伤风停胶囊、祛风止痒口服液、头孢克洛颗粒。具体用药请遵医嘱。
【高血压】的检查项目：血清白细胞介素3、动态血压监测(ABPM)、血压、速尿激发试验、心电图、红细胞聚集性、甘油三酯、血管紧张素Ⅱ、白细胞介素4（iL-4）、紧张度与动脉壁状态。
"""