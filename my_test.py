import os
import sys

# 尝试导入 OpenSTL 的加载函数
try:
    from openstl.utils import load_config

    HAS_OPENSTL = True
except ImportError:
    print("⚠️ 未找到 openstl 环境，将使用原生 Python 方式进行模拟加载测试...")
    HAS_OPENSTL = False


def test_load_config(file_path):
    print(f"🔍 开始测试加载配置文件: {os.path.abspath(file_path)}\n")

    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}，请检查路径是否正确！")
        return

    # 方法一：使用 OpenSTL 官方内置的 load_config 测试
    if HAS_OPENSTL:
        print("--- 正在使用 OpenSTL 原生 load_config 测试 ---")
        try:
            config = load_config(file_path)
            if not config:
                print("❌ 失败: OpenSTL 返回了空字典，说明文件存在但解析失败（可能有语法错误）。")
            else:
                print("✅ OpenSTL 加载成功！提取到的参数如下:")
                for k, v in config.items():
                    print(f"  👉 {k}: {v}")
        except Exception as e:
            print(f"❌ OpenSTL 解析时发生崩溃: {e}")

    print("\n------------------------------------------------\n")

    # 方法二：使用 Python 原生执行测试 (模拟 OpenSTL 底层行为)
    print("--- 正在使用原生 Python exec 测试 (严格语法检查) ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # 准备一个空字典来装载文件里定义的变量
        mock_globals = {}
        exec(code, mock_globals)

        # 过滤掉 Python 自带的 __builtins__ 等内置变量
        clean_config = {k: v for k, v in mock_globals.items() if not k.startswith('__')}

        print("✅ Python 语法解析完美通过！")
        if not HAS_OPENSTL:
            for k, v in clean_config.items():
                print(f"  👉 {k}: {v}")

    except SyntaxError as e:
        print(f"❌ 失败: 配置文件中存在 Python 语法错误！")
        print(f"    报错详情: {e}")
    except Exception as e:
        print(f"❌ 失败: 配置文件执行时发生错误: {e}")


if __name__ == '__main__':
    # 指向你的配置文件路径
    target_config_path = './configs/cikm/exprecast.py'
    test_load_config(target_config_path)