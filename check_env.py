from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import sys


DEPENDENCIES = {
    "sentence-transformers": "sentence_transformers",
    "chromadb": "chromadb",
    "openai": "openai",
    "gradio": "gradio",
    "python-dotenv": "dotenv",
}


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    failed = False

    for package_name, module_name in DEPENDENCIES.items():
        try:
            import_module(module_name)
            package_version = version(package_name)
        except (ImportError, PackageNotFoundError) as exc:
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
