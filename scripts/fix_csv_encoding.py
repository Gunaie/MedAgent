import pandas as pd
import os
import chardet


def detect_encoding(filepath):
    """检测文件编码"""
    with open(filepath, 'rb') as f:
        raw = f.read(10000)  # 读前10KB检测
        result = chardet.detect(raw)
        return result['encoding'], result['confidence']


def fix_csv_encoding(filepath):
    """修复单个 CSV 的编码"""
    try:
        # 检测编码
        encoding, confidence = detect_encoding(filepath)
        print(f"\n检测: {os.path.basename(filepath)} -> {encoding} (置信度: {confidence:.2f})")

        # 用检测到的编码读取
        df = pd.read_csv(filepath, encoding=encoding)

        # 打印第一行看看中文是否正常
        first_col = df.columns[0]
        first_val = df.iloc[0, 0] if len(df) > 0 else "EMPTY"
        print(f"  列名: {first_col}")
        print(f"  第一行: {first_val}")

        # 重新保存为 UTF-8
        df.to_csv(filepath, encoding='utf-8', index=False)
        print(f"  ✓ 已修复为 UTF-8")
        return True

    except Exception as e:
        print(f"  ✗ 修复失败: {e}")
        return False


# 修复所有节点 CSV
print("=" * 50)
print("修复节点 CSV")
print("=" * 50)
node_dir = '../neo4j_data/nodes'
for filename in sorted(os.listdir(node_dir)):
    if filename.endswith('.csv'):
        fix_csv_encoding(os.path.join(node_dir, filename))

# 修复所有关系 CSV
print("\n" + "=" * 50)
print("修复关系 CSV")
print("=" * 50)
rel_dir = '../neo4j_data/relations'
for filename in sorted(os.listdir(rel_dir)):
    if filename.endswith('.csv'):
        fix_csv_encoding(os.path.join(rel_dir, filename))

print("\n全部完成！")