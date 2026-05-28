# fix_code.py
import re

# ========== 修复 service.py ==========
with open('service.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复 result[:200] 切片
content = content.replace(
    'logger.info(f"Result: {result[:200]}...")',
    '''result_text = result.content if hasattr(result, "content") else str(result)
        logger.info(f"Result: {result_text[:200]}...")'''
)

# 修复 return result，确保返回字符串
content = content.replace(
    '''        self.history.add_user_message(original_message)
        self.history.add_ai_message(result)
        return result''',
    '''        # 确保 result 是字符串
        final_result = result.content if hasattr(result, "content") else str(result)
        self.history.add_user_message(original_message)
        self.history.add_ai_message(final_result)
        return final_result'''
)

with open('service.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✅ service.py 修复完成")

# ========== 修复 tools.py 关系类型 ==========
with open('tools.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 把所有 TREATED_BY 替换为 DISEASE_DRUG
content = content.replace('TREATED_BY', 'DISEASE_DRUG')

with open('tools.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✅ tools.py 关系类型修复完成 (TREATED_BY -> DISEASE_DRUG)")