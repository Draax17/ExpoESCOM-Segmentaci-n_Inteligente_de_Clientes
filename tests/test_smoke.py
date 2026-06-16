from src.pipeline.data_loader import generate_demo_customer_dataset
from src.pipeline.simulator import PipelineSimulator


def test_pipeline_simulator_runs() -> None:
    df = generate_demo_customer_dataset(n_samples=50)
    simulator = PipelineSimulator(k_min=2, k_max=4)
    result = simulator.run(df)

    assert not result.cleaned_data.empty
    assert result.selected_features
    assert "best_k" in result.metadata
