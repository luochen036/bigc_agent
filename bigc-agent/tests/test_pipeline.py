"""可选的自动化检查，不属于学习主流程，也不会调用真实聊天接口。

unittest 是 Python 自带的测试工具，assert 用来核对实际结果是否符合预期。
Mock 创建替身对象，patch 临时替换函数，使测试无需下载模型或使用真实密钥。
初次学习可先只看 src/test.py，掌握主流程后再阅读这里。
"""

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import chromadb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import agent
import languagemodel
from build_index import build_index
from data_loader import DOCS_DIR, load_docs
from retriever import retrieve
from splitter import split_text
from store import index_docs


class PipelineTests(unittest.TestCase):
    def setUp(self):
        # 每个测试开始前准备两段假资料：相似度分别为 0.9 和 0.2。
        self.collection = Mock()
        self.collection.count.return_value = 2
        self.collection.query.return_value = {
            "documents": [["First fact", "Second fact"]],
            "metadatas": [[{"source": "first.txt"}, {"source": "second.txt"}]],
            "distances": [[0.1, 0.8]],
        }
        self.model = Mock()
        self.model.encode.return_value = np.array([[1.0, 0.0]])

    def test_loader_default_does_not_depend_on_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                docs = load_docs()
            finally:
                os.chdir(original)
        self.assertEqual({doc["source"] for doc in docs}, {p.name for p in DOCS_DIR.glob("*.txt")})
        self.assertTrue(docs)

    def test_retrieve_caps_k_and_filters_scores(self):
        hits = retrieve(self.collection, "question", self.model, k=5, min_score=0.3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["source"], "first.txt")
        self.assertAlmostEqual(hits[0]["score"], 0.9)
        self.assertEqual(self.collection.query.call_args.kwargs["n_results"], 2)

    def test_empty_collection_skips_encoding(self):
        self.collection.count.return_value = 0
        self.assertEqual(retrieve(self.collection, "question", self.model), [])
        self.model.encode.assert_not_called()

    def test_invalid_query_options(self):
        for kwargs in ({"query": " "}, {"query": "q", "k": 0}, {"query": "q", "min_score": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                retrieve(self.collection, embedder=self.model, **kwargs)

    def test_prompt_contains_only_retrieved_material(self):
        hits = retrieve(self.collection, "question", self.model)
        prompt = agent.build_prompt("question", hits)
        self.assertIn("[1]", prompt)
        self.assertIn("first.txt", prompt)
        self.assertIn("First fact", prompt)
        self.assertNotIn("Second fact", prompt)

    @patch("agent.generate_answer")
    def test_no_relevant_hits_skips_remote_call(self, generate):
        self.collection.query.return_value["distances"] = [[0.9, 0.9]]
        result = agent.answer_question("question", self.collection, self.model)
        self.assertEqual(result["hits"], [])
        self.assertTrue(result["answer"])
        generate.assert_not_called()

    @patch("agent.generate_answer", return_value="Answer [1]")
    def test_answer_uses_retrieved_context(self, generate):
        result = agent.answer_question("question", self.collection, self.model)
        self.assertEqual(result["answer"], "Answer [1]")
        generate.assert_called_once_with(
            agent.SYSTEM_PROMPT, agent.build_prompt("question", result["hits"])
        )

    def test_missing_config_never_constructs_client(self):
        with patch.dict(os.environ, {}, clear=True), patch("languagemodel.load_dotenv"), patch("languagemodel.OpenAI") as client:
            with self.assertRaisesRegex(ValueError, "LLM_BASE_URL"):
                languagemodel.generate_answer("system", "user")
            client.assert_not_called()

    def test_env_file_loads_and_existing_environment_takes_priority(self):
        # 临时文件使用假配置，真实 .env 不会被读取或修改。
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "LLM_BASE_URL=https://example.invalid/v1\nLLM_API_KEY=test-key\nLLM_MODEL=file-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"LLM_MODEL": "environment-model"}, clear=True), patch("languagemodel.ENV_PATH", env_file):
                config = languagemodel.load_llm_config()
        self.assertEqual(config["LLM_MODEL"], "environment-model")
        self.assertEqual(config["LLM_API_KEY"], "test-key")

    def test_model_call_and_empty_response(self):
        config = {"LLM_BASE_URL": "https://example.invalid/v1", "LLM_API_KEY": "test-key", "LLM_MODEL": "test-model"}
        with patch("languagemodel.load_llm_config", return_value=config), patch("languagemodel.OpenAI") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=" Answer [1] "))]
            )
            self.assertEqual(languagemodel.generate_answer("system", "user"), "Answer [1]")
            messages = client.chat.completions.create.call_args.kwargs["messages"]
            self.assertEqual(messages, [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}])
            client.chat.completions.create.return_value = SimpleNamespace(choices=[])
            with self.assertRaises(RuntimeError):
                languagemodel.generate_answer("system", "user")

    def test_reindex_removes_obsolete_chunks(self):
        # 使用内存集合验证真实的数据库行为，避免改动本地学习用的索引。
        collection = chromadb.EphemeralClient().create_collection(
            "pipeline_test", metadata={"hnsw:space": "cosine"}
        )
        try:
            with patch("store.get_collection", return_value=collection):
                index_docs(["one", "two"], ["a.txt", "b.txt"], [[1.0, 0.0], [0.0, 1.0]])
                index_docs(["updated"], ["a.txt"], [[1.0, 0.0]])
                self.assertEqual(collection.count(), 1)
                self.assertEqual(collection.get()["documents"], ["updated"])
                with self.assertRaises(ValueError):
                    index_docs([], [], [])
                self.assertEqual(collection.count(), 1)
        finally:
            chromadb.EphemeralClient().delete_collection("pipeline_test")

    def test_empty_docs_preserve_index_and_skip_model(self):
        with tempfile.TemporaryDirectory() as directory, patch("build_index.load_model") as model, patch("build_index.index_docs") as index:
            with self.assertRaises(ValueError):
                build_index(data_dir=directory)
            model.assert_not_called()
            index.assert_not_called()

    def test_zero_overlap_does_not_repeat_previous_chunk(self):
        # 没有标点的两行也能分块；overlap=0 时不应把上一块整块带进下一块。
        self.assertEqual(split_text("abc\ndef", chunk_size=3, overlap=0), ["abc", "def"])


if __name__ == "__main__":
    unittest.main()
