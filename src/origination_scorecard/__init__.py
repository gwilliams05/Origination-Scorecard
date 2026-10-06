"""Public interface for the synthetic origination-scorecard data generator."""

from .generator import GeneratorConfig, generate_project_data, write_project_data

__all__ = ["GeneratorConfig", "generate_project_data", "write_project_data"]
