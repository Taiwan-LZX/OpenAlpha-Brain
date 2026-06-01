"""测试跨 Session 学习持久化功能"""
from __future__ import annotations

import json
from pathlib import Path

from openalpha_brain.learning.experience_distiller import ExperienceDistiller
from openalpha_brain.learning.mab import HierarchicalMAB


class TestMABPersistence:
    """测试 MAB bandit 状态持久化"""

    def test_save_and_load_state(self, tmp_path: Path):
        """测试保存和加载 MAB 状态"""
        mab = HierarchicalMAB()
        mab._update_count = 42

        save_path = tmp_path / "test_mab.json"

        result = mab.save_state(path=save_path)
        assert result is True
        assert save_path.exists()

        loaded_data = json.loads(save_path.read_text(encoding="utf-8"))
        assert "hierarchical_mab" in loaded_data
        assert loaded_data["version"] == "1.0"
        assert "saved_at" in loaded_data

    def test_load_state_restores_data(self, tmp_path: Path):
        """测试加载状态恢复数据"""
        original = HierarchicalMAB()
        original._update_count = 100

        save_path = tmp_path / "test_mab_restore.json"
        original.save_state(path=save_path)

        restored = HierarchicalMAB()
        result = restored.load_state(path=save_path)

        assert result is True
        assert restored._update_count == 100

    def test_load_state_file_not_found(self, tmp_path: Path):
        """测试文件不存在时返回 False"""
        mab = HierarchicalMAB()
        result = mab.load_state(path=tmp_path / "nonexistent.json")
        assert result is False

    def test_load_state_invalid_json(self, tmp_path: Path):
        """测试无效 JSON 文件时返回 False"""
        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text("{invalid json}", encoding="utf-8")

        mab = HierarchicalMAB()
        result = mab.load_state(path=invalid_path)
        assert result is False


class TestExperienceDistillerPersistence:
    """测试 Experience 卡片持久化"""

    def test_save_and_load_cards(self, tmp_path: Path):
        """测试保存和加载经验卡片"""
        distiller = ExperienceDistiller(path=str(tmp_path / "test_cards.json"))

        card1 = distiller.record_single_failure(
            failure_pattern="test_pattern_1",
            fix_strategy="fix_strategy_1",
            direction="momentum",
        )
        assert card1 is not None

        card2 = distiller.record_single_failure(
            failure_pattern="test_pattern_2",
            fix_strategy="fix_strategy_2",
            direction="mean_reversion",
        )
        assert card2 is not None

        result = distiller.save_cards(path=tmp_path / "test_saved_cards.json")
        assert result is True

        new_distiller = ExperienceDistiller(path=str(tmp_path / "test_empty.json"))
        count = new_distiller.load_cards(path=tmp_path / "test_saved_cards.json")

        assert count == 2
        assert len(new_distiller._cards) == 2

    def test_load_cards_file_not_found(self, tmp_path: Path):
        """测试文件不存在时返回 0"""
        distiller = ExperienceDistiller(path=str(tmp_path / "test_empty.json"))
        count = distiller.load_cards(path=tmp_path / "nonexistent.json")
        assert count == 0

    def test_save_cards_creates_directory(self, tmp_path: Path):
        """测试保存时自动创建目录"""
        distiller = ExperienceDistiller(path=str(tmp_path / "test_empty.json"))
        nested_path = tmp_path / "nested" / "dir" / "cards.json"

        result = distiller.save_cards(path=nested_path)
        assert result is True
        assert nested_path.exists()

    def test_persistence_preserves_card_attributes(self, tmp_path: Path):
        """测试持久化保留卡片属性"""
        distiller = ExperienceDistiller(path=str(tmp_path / "test_attr.json"))

        card = distiller.record_single_failure(
            failure_pattern="pattern_with_attrs",
            fix_strategy="strategy_with_attrs",
            direction="value",
        )
        assert card is not None
        card.confidence = 0.85
        card.usage_count = 10
        card.success_count = 8

        save_path = tmp_path / "test_attr_saved.json"
        distiller.save_cards(path=save_path)

        new_distiller = ExperienceDistiller(path=str(tmp_path / "test_new.json"))
        new_distiller.load_cards(path=save_path)

        assert len(new_distiller._cards) == 1
        restored_card = new_distiller._cards[0]
        assert restored_card.failure_pattern == "pattern_with_attrs"
        assert restored_card.fix_strategy == "strategy_with_attrs"
        assert restored_card.confidence == 0.85
        assert restored_card.usage_count == 10
        assert restored_card.success_count == 8
