import pytest
from openalpha_brain.utils.tool_factory import ToolFactory, AlphaTool


async def _stub_embed_fn(text):
    return [0.1, 0.2, 0.3]


async def _stub_embed_fn_same(text):
    return [0.5, 0.5, 0.5]


class TestRecordFixPattern:
    def test_tracks_pattern_count(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"))
        factory.record_fix_pattern("LOW_SHARPE", "add ts_decay_linear", fix_success=True, direction="momentum")
        key = "LOW_SHARPE::add ts_decay_linear"
        assert factory._pattern_counter.get(key) == 1

    def test_increments_existing_pattern(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"))
        factory.record_fix_pattern("LOW_SHARPE", "add ts_decay_linear", fix_success=True, direction="momentum")
        factory.record_fix_pattern("LOW_SHARPE", "add ts_decay_linear", fix_success=True, direction="momentum")
        key = "LOW_SHARPE::add ts_decay_linear"
        assert factory._pattern_counter.get(key) == 2


class TestCreateToolFromPattern:
    @pytest.mark.asyncio
    async def test_creates_tool_with_embed_fn(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"), embed_fn=_stub_embed_fn)

        tool = await factory.create_tool_from_pattern("LOW_SHARPE", "add ts_decay_linear", direction="momentum")

        assert tool is not None
        assert "low_sharpe" in tool.name
        assert "momentum" in tool.name
        assert tool.embedding is not None

    @pytest.mark.asyncio
    async def test_creates_tool_without_embed_fn(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"))
        tool = await factory.create_tool_from_pattern("SELF_CORRELATION", "change rank to zscore", direction="value")

        assert tool is not None
        assert tool.embedding is None
        assert "zscore" in tool.parameters or "rank" in tool.parameters

    @pytest.mark.asyncio
    async def test_returns_none_for_duplicate_name(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"))
        tool1 = await factory.create_tool_from_pattern("LOW_SHARPE", "add ts_decay_linear", direction="momentum")
        assert tool1 is not None

        tool2 = await factory.create_tool_from_pattern("LOW_SHARPE", "add ts_decay_linear", direction="momentum")
        assert tool2 is None

    @pytest.mark.asyncio
    async def test_returns_none_for_high_similarity_embedding(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"), embed_fn=_stub_embed_fn_same)

        tool1 = await factory.create_tool_from_pattern("LOW_SHARPE", "add ts_decay_linear", direction="momentum")
        assert tool1 is not None

        tool2 = await factory.create_tool_from_pattern("LOW_SHARPE", "add smoothing", direction="momentum")
        assert tool2 is None


class TestSearchTools:
    @pytest.mark.asyncio
    async def test_returns_relevant_tools(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"), embed_fn=_stub_embed_fn)

        await factory.create_tool_from_pattern("LOW_SHARPE", "add ts_decay_linear", direction="momentum")

        results = await factory.search_tools("fix low sharpe", top_k=3)
        assert len(results) >= 1
        assert "tool" in results[0]
        assert "similarity" in results[0]

    @pytest.mark.asyncio
    async def test_returns_empty_without_embed_fn(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"))
        results = await factory.search_tools("fix low sharpe")
        assert results == []


class TestApplyTool:
    def test_records_success(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"))
        tool = AlphaTool(name="test_tool", fix_logic="do something")
        factory._tools[tool.tool_id] = tool

        factory.apply_tool(tool.tool_id, success=True)
        assert tool.success_count == 1
        assert tool.fail_count == 0

    def test_records_failure(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"))
        tool = AlphaTool(name="test_tool", fix_logic="do something")
        factory._tools[tool.tool_id] = tool

        factory.apply_tool(tool.tool_id, success=False)
        assert tool.success_count == 0
        assert tool.fail_count == 1

    def test_noop_for_unknown_tool_id(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"))
        factory.apply_tool("nonexistent", success=True)
        assert len(factory._tools) == 0


class TestDetectConflicts:
    def test_finds_similar_tools(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"))
        tool_a = AlphaTool(name="tool_a", fix_logic="fix1", embedding=[0.9, 0.1, 0.0])
        tool_b = AlphaTool(name="tool_b", fix_logic="fix2", embedding=[0.91, 0.11, 0.01])
        factory._tools[tool_a.tool_id] = tool_a
        factory._tools[tool_b.tool_id] = tool_b

        conflicts = factory.detect_conflicts()
        assert len(conflicts) == 1

    def test_no_conflicts_for_dissimilar_tools(self, tmp_path):
        factory = ToolFactory(path=str(tmp_path / "tools.json"))
        tool_a = AlphaTool(name="tool_a", fix_logic="fix1", embedding=[1.0, 0.0, 0.0])
        tool_b = AlphaTool(name="tool_b", fix_logic="fix2", embedding=[0.0, 1.0, 0.0])
        factory._tools[tool_a.tool_id] = tool_a
        factory._tools[tool_b.tool_id] = tool_b

        conflicts = factory.detect_conflicts()
        assert len(conflicts) == 0


class TestToolFactoryPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "tools.json")
        factory = ToolFactory(path=path)
        tool = AlphaTool(name="persisted_tool", fix_logic="do_fix", applicable_conditions="LOW_SHARPE + momentum")
        factory._tools[tool.tool_id] = tool
        factory._pattern_counter["LOW_SHARPE::do_fix"] = 3
        factory._save()

        factory2 = ToolFactory(path=path)
        assert len(factory2._tools) == 1
        loaded = list(factory2._tools.values())[0]
        assert loaded.name == "persisted_tool"
        assert factory2._pattern_counter.get("LOW_SHARPE::do_fix") == 3
