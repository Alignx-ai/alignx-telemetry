# Contributing to Alignx Telemetry

Thank you for your interest in contributing to Alignx Telemetry! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already
4. Install dependencies: `uv sync --active`
5. Install pre-commit hooks: `pre-commit install`

## Development Setup

### Prerequisites
- Python 3.12 or higher
- Git
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Fast Python package installer and resolver

### Setting up virtaul env
```bash
uv venv

source .venv/bin/activate (Unix/MacOS)
.venv\Scripts/activate
```


### Installing Dependencies
```bash
uv sync --active
```

### Running Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black .
uv run isort .
```

### Linting
```bash
uv run flake8
uv run mypy .
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Use meaningful variable and function names

## Testing

- Write unit tests for new functionality
- Ensure test coverage doesn't decrease
- Use pytest for testing
- Mock external dependencies appropriately

## Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions and classes
- Update CHANGELOG.md for significant changes

## Issues

- Use GitHub issues for bug reports and feature requests
- Provide clear, detailed descriptions
- Include reproduction steps for bugs
- Tag issues appropriately

## License

By contributing to Alignx Telemetry, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, please open an issue or contact the maintainers.
