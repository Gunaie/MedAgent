# fix_service.py
with open('service.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到并修复 try/except 块
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]

    # 修复错误的 try 块结构
    if 'result_text = result.content if hasattr(result, "content") else str(result)' in line:
        # 确保 logger.info 也在 try 块内（缩进相同）
        new_lines.append(line)  # result_text = ...
        i += 1
        if i < len(lines) and 'logger.info(f"Result:' in lines[i]:
            # 修正 logger.info 的缩进，让它和 result_text 对齐
            indent = len(line) - len(line.lstrip())
            new_lines.append(' ' * indent + lines[i].lstrip())
            i += 1
        continue

    new_lines.append(line)
    i += 1

with open('service.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ service.py 缩进修复完成")