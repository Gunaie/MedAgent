import pandas as pd
import os


def force_fix_with_gbk(filepath):
    """强制用 GBK 读取，再保存为 UTF-8"""
    try:
        print(f"\n修复: {os.path.basename(filepath)}")

        # 方法1：直接用 GBK 读取
        try:
            df = pd.read_csv(filepath, encoding='gbk')
            first_val = df.iloc[0, 0] if len(df) > 0 else "EMPTY"
            print(f"  GBK读取成功，第一行: {first_val}")
        except:
            # 方法2：用 GB18030
            df = pd.read_csv(filepath, encoding='gb18030')
            first_val = df.iloc[0, 0] if len(df) > 0 else "EMPTY"
            print(f"  GB18030读取成功，第一行: {first_val}")

        # 保存为 UTF-8
        df.to_csv(filepath, encoding='utf-8', index=False)
        print(f"  ✓ 已修复为 UTF-8")
        return True

    except Exception as e:
        print(f"  ✗ 修复失败: {e}")
        return False


# 需要修复的文件列表
files_to_fix = [
    r'D:\Projects\MedAgent\neo4j_data\nodes\Cureway.csv',
    r'D:\Projects\MedAgent\neo4j_data\nodes\Disease.csv',
    r'D:\Projects\MedAgent\neo4j_data\nodes\Drug.csv',
    r'D:\Projects\MedAgent\neo4j_data\relations\DISEASE_ACOMPANY.csv',
    r'D:\Projects\MedAgent\neo4j_data\relations\DISEASE_CHECK.csv',
    r'D:\Projects\MedAgent\neo4j_data\relations\DISEASE_DEPARTMENT.csv',
    r'D:\Projects\MedAgent\neo4j_data\relations\DISEASE_DO_EAT.csv',
]

print("=" * 50)
print("强制用 GBK 修复乱码 CSV")
print("=" * 50)

for filepath in files_to_fix:
    if os.path.exists(filepath):
        force_fix_with_gbk(filepath)
    else:
        print(f"\n跳过（不存在）: {filepath}")

print("\n全部完成！")