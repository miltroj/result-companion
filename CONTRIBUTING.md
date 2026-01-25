# Contributing to Result Companion

This document outlines the contribution guidelines for Result Companion.

## Types of Contributions

We accept the following contribution types:

- **Bug Reports** - Report issues you encounter
- **Feature Requests** - Suggest new functionality
- **Code Contributions** - Submit bug fixes or features
- **Documentation** - Improve docs, examples, or docstrings
- **Examples & Tutorials** - Add usage examples
- **Community Support** - Answer questions in Issues/Discussions
- **Testing & Feedback** - Test features and provide feedback

## Getting Started

### Development Setup

See the [README](README.md) for basic installation instructions. For development:

```bash
git clone https://github.com/miltroj/result-companion.git
cd result-companion
poetry install --with=dev
```

## Code Standards

Result Companion follows strict code quality principles. Key requirements:

- **Type hints** on all functions (mandatory)
- **Google-style docstrings** for all public functions
- **Maximum nesting depth**: 2 levels
- **Maximum cyclomatic complexity**: 10
- **Single responsibility** - one function, one job
- **Testable code** - separate I/O from logic, use dependency injection
- **Tools**: ruff (linting), black (formatting), pytest (testing)

### Pre-Submission Checklist

Before submitting a PR, verify:

- [ ] Type hints present on all functions
- [ ] Google docstrings on all public functions
- [ ] Nesting depth ≤ 2 levels
- [ ] One function = one job
- [ ] Dependencies passed as parameters (not hardcoded)
- [ ] Tests included (mandatory)
- [ ] Tests pass locally
- [ ] Code coverage maintained/improved
- [ ] Documentation updated (if applicable)
- [ ] Linting passes (`poetry run ruff check .`)
- [ ] Formatting applied (`poetry run black .`)

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** - suggested format: `feature/description` or `bugfix/description`
3. **Make changes** - follow code standards, write tests
4. **Commit** - use clear, descriptive commit messages
5. **Push** and open a Pull Request
6. **Link to issue** - reference related issues when applicable
7. **CI checks** - automated tests, linting, and coverage checks must pass

## Reporting Issues

### Bug Reports

Include the following information:

**Required:**
- Python version (e.g., 3.10, 3.11)
- Operating system (Linux/macOS/Windows)
- Result Companion version
- Steps to reproduce
- Expected vs actual behavior
- `result_companion.log` file (if available)

**Example:**

```markdown
**Python**: 3.11.5
**OS**: macOS 14.0
**Version**: 0.0.2

**Steps to reproduce:**
1. Run `result-companion analyze -o output.xml`
2. Error occurs when...

**Expected**: Should complete analysis
**Actual**: Crashes with error: ...

**Log file attached**: result_companion.log
```

### Feature Requests

Provide the following:

- **Use case** - what problem does this solve?
- **Proposed solution** - how should it work?
- **Benefits** - who benefits and how?
- **Alternatives considered** - other approaches you've thought about

## Testing Requirements

- **Tests are mandatory** for all code contributions
- **Minimum coverage**: Maintain or improve existing coverage threshold (see `.coverage_threshold`)
- **Test style**: Follow patterns in existing tests
- Use simple fakes over complex mocks
- One test = one scenario
- Descriptive test names: `test_<function>_<scenario>_<expected>`

## Documentation Requirements

Documentation is mandatory for:

- **New features** - update README, add examples if needed
- **API changes** - update docstrings
- **Bug fixes** - update docs if behavior changes
- **All public functions** - must have Google-style docstrings

## Communication

- **GitHub Issues** - bug reports, feature requests, questions
- **Pull Requests** - code contributions and discussion

## License

By contributing to Result Companion, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

All contributions must be your original work or properly attributed with compatible licensing.
