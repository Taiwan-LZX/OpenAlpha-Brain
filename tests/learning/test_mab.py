from openalpha_brain.learning.mab import HierarchicalMAB


class TestMABExplorer:
    def test_select_and_update(self):
        mab = HierarchicalMAB()
        result = mab.select()
        assert isinstance(result, dict)
        assert "direction" in result
        direction = result["direction"]
        assert direction in mab._outer._arms
        mab.update(direction, operators=[], fields=[], reward=1.0)

    def test_set_initial_bias(self):
        mab = HierarchicalMAB()
        mab.set_initial_bias("momentum", 3.0)
        arm = mab._outer._arms.get("momentum")
        assert arm is not None
        assert arm.alpha > 1.0
