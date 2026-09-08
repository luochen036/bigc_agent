"""可选环境检查：在仓库根目录运行 python check_env.py。

只检查核心依赖是否能导入，不加载 .env，也不请求模型接口。
完成环境安装后可先跳过本文件，按 README 的顺序学习 src 中的主流程。
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import sys


DEPENDENCIES = {
    "sentence-transformers": "sentence_transformers",
    "chromadb": "chromadb",
    "openai": "openai",
    "python-dotenv": "dotenv",
}


def main() -> int:
    # Python 包的安装名和导入名可能不同，例如 python-dotenv 对应 dotenv。
    print(f"Python {sys.version.split()[0]}")
    failed = False

    for package_name, module_name in DEPENDENCIES.items():
        try:
            import_module(module_name)
            package_version = version(package_name)
        except (ImportError, PackageNotFoundError) as exc:
            # 包未安装时记录失败，但继续检查后面的依赖，一次列出所有缺失项。
            failed = True
            print(f"[MISSING] {package_name}: {exc}")
        except Exception as exc:
            failed = True
            print(f"[ERROR] {package_name}: {exc}")
        else:
            print(f"[OK] {package_name} {package_version}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
