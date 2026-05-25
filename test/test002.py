"""
测试端到端 Agent
"""

from service import ChatService
svc = ChatService("test")
print(svc.answer("鼻炎会有哪些症状？"))
print(svc.answer("鼻炎吃什么药好得快？"))
print(svc.answer("高血压要做哪些检查？"))

"""
[DEBUG] Model: qwen3.5-flash
[KG] Cache incomplete (missing {'高血压', '糖尿病', '阿司匹林', '鼻炎'}), refreshing from Neo4j...
[Neo4j] 实际关系类型: {'DISEASE_DEPARTMENT', 'DISEASE_DRUG', 'DISEASE_CUREWAY', 'DISEASE_DISHES', 'DISEASE_NOT_EAT', 'DISEASE_CHECK', 'DISEASE_CATEGORY', 'DISEASE_DO_EAT', 'DISEASE_ACOMPANY', 'DISEASE_SYMPTOM'}
[Neo4j] 药物相关关系: {'DISEASE_DRUG'}
[KG] Loaded 18215 entities from Neo4j (Trie)
[KG] Sample entities: ['(奥松)肚痛泻丸', '125Ala', '125Ala注射液', '125Ser', '15AA', '18AA-Ⅰ', '18注射液', '19AA-Ⅰ', '21', '23', '29', '2∶1', '30R', '4:1', '4:1分散片']
[KG] 模板精确命中: symptom, 实体: 鼻炎
【鼻炎】的症状包括：深侧鼻塞、流鼻涕、鼻粘膜苍白水肿、鼻塞排出脓性或...、上额窦囊肿、鼻出血、鼻粘膜干燥发痒、鼻子糜烂、鼻子上火、鼻酸。
[KG] 模板精确命中: cure_way, 实体: 鼻炎
【鼻炎】可用药物：孟鲁司特钠片、匹多莫德口服溶液、马来酸氯苯那敏片、鼻窦炎口服液、匹多莫德片、氨苄西林胶囊、富马酸酮替芬片、盐酸羟甲唑啉喷雾剂、丙酸氟替卡松鼻喷雾剂、益鼻喷雾剂、孟鲁司特钠咀嚼片、盐酸西替利嗪片、藿胆滴丸、富马酸酮替芬胶囊、苍夷滴鼻剂、注射用头孢唑林钠、鼻炎灵片、伤风停胶囊、祛风止痒口服液、头孢克洛颗粒。具体用药请遵医嘱。
[KG] 模板精确命中: check, 实体: 高血压
【高血压】的检查项目：血清白细胞介素3、动态血压监测(ABPM)、血压、速尿激发试验、心电图、红细胞聚集性、甘油三酯、血管紧张素Ⅱ、白细胞介素4（iL-4）、紧张度与动脉壁状态。
"""