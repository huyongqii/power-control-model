import re

def analyze_sql_file_alternative(file_path):
    """
    通过解析SQL文件内容来获取表结构信息
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # 查找CREATE TABLE语句
        create_table_pattern = r"CREATE TABLE.*?\((.*?)\)"
        columns = []
        
        # 查找INSERT语句来计算行数
        insert_count = len(re.findall(r"INSERT INTO", content, re.IGNORECASE))
        
        # 提取列名
        matches = re.findall(create_table_pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            # 从第一个CREATE TABLE语句中提取列定义
            columns_def = matches[0]
            # 提取列名
            column_matches = re.findall(r"(\w+)\s+[\w\(\)]+", columns_def)
            columns = column_matches
        
        print(f"文件 {file_path} 的分析结果：")
        print(f"列名：{columns}")
        print(f"预计行数（基于INSERT语句数）：{insert_count}")
        
        return columns, insert_count
    
    except Exception as e:
        print(f"处理文件时出错：{str(e)}")
        return None, None

if __name__ == "__main__":
    sql_file = "wm1_database-20241210.sql"
    # 尝试两种方法
    try:
        analyze_sql_file(sql_file)
    except:
        print("\n尝试替代方法：")
        analyze_sql_file_alternative(sql_file)
