import os
import json
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, TextIO

def is_binary_file(file_path: Path) -> bool:
    """判断文件是否为二进制文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return False
    except (UnicodeDecodeError, UnicodeError):
        return True

def generate_tree(base_path: Path, ignored_dirs: List[str]) -> Tuple[Dict, List]:
    """
    生成项目目录树结构
    :param base_path: 根目录路径
    :param ignored_dirs: 忽略的文件夹名称列表
    :return: (目录树结构, 代码块列表)
    """
    code_blocks = []
    tree = _build_tree_node(base_path, base_path, code_blocks, ignored_dirs)
    return tree, code_blocks

def _build_tree_node(path: Path, base_path: Path, code_blocks: List[Dict], ignored_dirs: List[str]) -> Dict:
    """递归构建目录树节点"""
    tree_node = {"name": str(path.relative_to(base_path)), "directories": [], "files": []}
    
    for entry in sorted(path.iterdir(), key=lambda e: e.name):
        if entry.name in ignored_dirs:
            continue
            
        if entry.name in ["output.json", "output.txt"]:
            continue
            
        if entry.is_dir():
            subtree = _build_tree_node(entry, base_path, code_blocks, ignored_dirs)
            tree_node["directories"].append(subtree)
        else:
            if not is_binary_file(entry):
                content = entry.read_text(encoding='utf-8')
                code_blocks.append({
                    "filename": str(entry.relative_to(base_path)),
                    "content": content
                })
                tree_node["files"].append({
                    "name": entry.name,
                    "content_index": len(code_blocks) - 1
                })
                
    return tree_node

def render_tree_to_json(tree: Dict, code_blocks: List, output_file: Path):
    """将目录树渲染为JSON格式"""
    data = {
        "tree": tree,
        "code_blocks": code_blocks
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def render_tree_to_txt(tree: Dict, code_blocks: List, output_file: Path):
    """将目录树渲染为TXT格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        _render_tree_structure(tree, f)
        _render_file_contents(tree, code_blocks, f)

def _render_tree_structure(node: Dict, f: TextIO, prefix: str = "", last: bool = False):
    """递归渲染目录结构"""
    connector = "└──" if last else "├──"
    f.write(f"{prefix}{connector} {node['name']}/\n")
    new_prefix = prefix + ("    " if last else "│   ")

    # 写入当前目录下的文件
    for i, file in enumerate(node['files']):
        is_last = i == len(node['files']) - 1
        file_connector = "└──" if is_last else "├──"
        f.write(f"{new_prefix}{file_connector} {file['name']}\n")

    # 递归写入子目录
    children = node['directories']
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        _render_tree_structure(child, f, new_prefix, is_last)

def _render_file_contents(node: Dict, code_blocks: List, f: TextIO):
    """递归渲染文件内容"""
    for file in node['files']:
        idx = file['content_index']
        filename = code_blocks[idx]['filename']
        content = code_blocks[idx]['content']
        f.write(f"=== FILE: {filename} ===\n")
        f.write(f"{content}\n")
        f.write("=== END ===\n\n")

    for child in node['directories']:
        _render_file_contents(child, code_blocks, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成项目结构及代码块内容')
    parser.add_argument('--format', '-f', type=str, default='json', choices=['json', 'txt'],
                        help='输出格式: json 或 txt')
    args = parser.parse_args()

    base_path = Path('.')
    output_filename = 'output.json' if args.format == 'json' else 'output.txt'
    output_file = base_path / output_filename

    ignored_dirs = [
        ".git", "log", "__pycache__", "node_modules",
        "allPath.py", "output.json", "output.txt", ".vscode"
    ]

    tree, code_blocks = generate_tree(base_path, ignored_dirs)

    if args.format == 'json':
        render_tree_to_json(tree, code_blocks, output_file)
    else:
        render_tree_to_txt(tree, code_blocks, output_file)

    print(f"输出已保存到 {output_file}")