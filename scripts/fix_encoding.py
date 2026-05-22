#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
disease.csv 编码修复脚本
原文件是GBK编码，但包含少量损坏的字节
"""

import csv
import re

def fix_disease_csv(input_path, output_path):
    """修复disease.csv的编码问题"""

    # 1. 读取原始文件（二进制）
    with open(input_path, 'rb') as f:
        raw_data = f.read()

    print(f"原始文件大小: {len(raw_data)} 字节")

    # 2. 用GBK解码，无法解码的字符用�替换
    text = raw_data.decode('gbk', errors='replace')

    # 3. 统计损坏的字符
    bad_chars = text.count('�')
    print(f"发现 {bad_chars} 个无法解码的字符")

    # 4. 修复已知的损坏模式
    # "地尔硫��片" → "地尔硫䓬片" (盐酸地尔硫䓬片)
    if '地尔硫��片' in text:
        text = text.replace('地尔硫��片', '地尔硫䓬片')
        print("✓ 修复: 地尔硫??片 → 地尔硫䓬片")

    # 5. 删除剩余的无法解码字符（或保留为?）
    # 这里选择删除，避免影响CSV解析
    text = text.replace('�', '')

    # 6. 保存为UTF-8
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"✓ 修复完成，已保存到: {output_path}")

    # 7. 验证
    with open(output_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        print(f"✓ CSV验证通过，列数: {len(headers)}")
        print(f"  列名: {headers}")

        # 统计行数
        row_count = sum(1 for _ in reader)
        print(f"  数据行数: {row_count}")

    return True

if __name__ == '__main__':
    import sys

    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = 'neo4j_data/disease.csv'
        output_file = 'neo4j_data/disease_utf8.csv'

    fix_disease_csv(input_file, output_file)