import os, sys
sys.path.insert(0, '.')
os.environ['ALPHA_AGENT_LLM_GEN_ENABLED'] = 'true'
from dotenv import load_dotenv
load_dotenv()

from alpha_agent.config import ModelConfig
from alpha_agent.llm_client import LLMClient
from alpha_agent.exploration_grid import ExplorationGrid
from alpha_agent.expression_generator import ExpressionGenerator, JaccardDiversityGate
from alpha_agent.rag_spec import DeterministicValidator, RAGSpecEncoder
from pathlib import Path

cfg = ModelConfig(
    provider='openai',
    model='deepseek-ai/deepseek-v4-flash',
    base_url='https://integrate.api.nvidia.com/v1',
    api_key_env='OPENAI_API_KEY',
    temperature=0.1,
)
client = LLMClient(cfg)
gate = JaccardDiversityGate(threshold=0.3)
validator = DeterministicValidator.from_fields_summary(Path('data/field_summaries.json'))
encoder = RAGSpecEncoder.from_fields_summary(Path('data/field_summaries.json'))

generator = ExpressionGenerator(
    validator=validator,
    rag_encoder=encoder,
    diversity_gate=gate,
    llm_client=client,
    llm_gen_config=None,
)

grid = ExplorationGrid()
cells = grid.select_cells(budget=2)
print(f"Selected {len(cells)} cells:")
for cell in cells:
    print(f"\n=== {cell.cell_id()} ===")
    print(f"Fields: {cell.candidate_fields[:5]}")
    print(f"Thesis count: {len(cell.thesis)}")
    print(f"Hypothesis: {cell.build_hypothesis()}")
    generated = generator.generate_from_cell(cell, frontier_expressions=[])
    for gc in generated:
        print(f"  Expression: {gc.expression}")
        print(f"  Idea: {gc.idea_name}")
        print(f"  Valid: {gc.validation_passed}")
        print(f"  Diversity: {gc.diversity_score:.2f}")
        print()
