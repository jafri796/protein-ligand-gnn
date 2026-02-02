# Contributing to Protein-Ligand GNN

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training)
- Git

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/jafri796/protein-ligand-gnn.git
cd protein-ligand-gnn
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Verify installation:
```bash
pytest tests/ -v
```

## How to Contribute

### Reporting Bugs

- Check existing issues to avoid duplicates
- Use the bug report template
- Include:
  - Python version
  - PyTorch/PyG versions
  - Minimal reproducible example
  - Full error traceback

### Suggesting Features

- Open an issue with the feature request template
- Explain the use case and expected behavior
- Discuss before implementing major changes

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Format code: `black . && isort .`
6. Commit with descriptive messages
7. Push and create a Pull Request

## Code Style

### Python

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Add type hints where practical
- Write docstrings for public functions

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- First line: 50 characters max, imperative mood
- Body: explain what and why, not how

Example:
```
Add LP-PDBBind split validation

- Verify no sequence similarity leakage between splits
- Add unit tests for similarity computation
- Update documentation with split statistics
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_models.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for setup/teardown
- Aim for meaningful coverage, not 100%

## Scientific Contributions

### Model Improvements

- Document the scientific rationale
- Include relevant citations
- Provide benchmark comparisons
- Ensure SE(3)-equivariance is preserved

### Data Processing

- Validate feature dimensions match documentation
- Test on edge cases (missing atoms, unusual residues)
- Ensure LP-PDBBind compatibility

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions
- Update config examples if parameters change
- Include usage examples for new features

## Questions?

- Open a discussion on GitHub
- Tag maintainers in issues if needed

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
