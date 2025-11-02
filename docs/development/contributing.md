# Contributing

*Guidelines for contributing to Visual Next Token.*

## Welcome!

Thank you for your interest in contributing to Visual Next Token! This project explores curiosity-driven reinforcement learning for image navigation, and we welcome contributions of all kinds.

## Ways to Contribute

### 1. 🐛 Report Bugs

Found a bug? Please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)
- Error messages or logs

### 2. 💡 Suggest Features

Have an idea? Open an issue with:
- Use case description
- Proposed solution
- Alternative approaches considered
- Potential challenges

### 3. 📝 Improve Documentation

Documentation contributions are highly valued:
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve API documentation
- Create guides for common tasks

### 4. 🔬 Contribute Research

Share your experiments:
- Novel reward formulations
- Ablation studies
- Baseline comparisons
- New application domains

### 5. 💻 Submit Code

Code contributions welcome:
- Bug fixes
- New features
- Performance improvements
- Test coverage

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/visual-next-token.git
cd visual-next-token

# Add upstream remote
git remote add upstream https://github.com/georgepearse/visual-next-token.git
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black isort mypy
```

### 3. Create Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## Code Style

### Python Style

We follow PEP 8 with some modifications:

```bash
# Format code with black
black techniques/ experiments/

# Sort imports with isort
isort techniques/ experiments/

# Type check with mypy (coming soon)
mypy techniques/
```

### Conventions

- **Line length**: 88 characters (black default)
- **Imports**: Sorted with isort
- **Type hints**: Encouraged for all public functions
- **Docstrings**: Google style

Example:
```python
def compute_reward(
    features_t: torch.Tensor,
    features_t1: torch.Tensor,
    action: torch.Tensor
) -> torch.Tensor:
    """Compute rolling-window prediction accuracy reward.

    Args:
        features_t: Current semantic features (batch_size, feature_dim)
        features_t1: Next semantic features (batch_size, feature_dim)
        action: Action taken (batch_size,)

    Returns:
        reward: Computed reward (batch_size,)
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_encoder.py

# Run with coverage
pytest --cov=techniques tests/
```

### Writing Tests

Place tests in `tests/` directory:

```python
# tests/test_encoder.py
import pytest
import torch
from techniques.rl_navigation import SemanticEncoder

def test_encoder_freeze():
    encoder = SemanticEncoder(model_name="dinov2_vits14")
    encoder.freeze()

    # Check all parameters frozen
    for param in encoder.parameters():
        assert not param.requires_grad

def test_encoder_unfreeze():
    encoder = SemanticEncoder(model_name="dinov2_vits14")
    encoder.freeze()
    encoder.unfreeze_top_layers(n_layers=2)

    # Check some parameters unfrozen
    unfrozen_params = [p for p in encoder.parameters() if p.requires_grad]
    assert len(unfrozen_params) > 0
```

## Pull Request Process

### 1. Update Your Branch

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Make Your Changes

- Write clear, concise commit messages
- Keep commits focused and atomic
- Add tests for new features
- Update documentation

### 3. Run Checks

```bash
# Format code
black techniques/ experiments/
isort techniques/ experiments/

# Run tests
pytest

# Check types (if mypy configured)
mypy techniques/
```

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots/examples if applicable
- Checklist of completed items

### PR Template

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123

## Changes Made
- [ ] Added feature X
- [ ] Updated documentation
- [ ] Added tests
- [ ] Ran formatting (black, isort)

## Testing
Describe how you tested the changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Commit Message Guidelines

Use conventional commits format:

```
type(scope): short description

Longer description if needed

Fixes #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(encoder): add support for DINOv2 vitg14 model

Adds support for the 1.1B parameter DINOv2 model variant.
Includes configuration preset and memory optimization.

Fixes #45
```

```
fix(environment): correct rolling-window reward computation

Previous implementation used prediction error instead of accuracy.
Updated to match documented rolling-window accuracy formulation.

Fixes #67
```

## Documentation

### Building Docs Locally

```bash
# Install MkDocs
pip install mkdocs-material

# Serve locally
mkdocs serve

# Open http://127.0.0.1:8000
```

### Adding New Pages

1. Create `.md` file in `docs/`
2. Add to `mkdocs.yml` navigation
3. Link from relevant pages
4. Build and verify

## Research Contributions

### Sharing Experiments

Create an issue or discussion with:
- Research question
- Methodology
- Results (with visualizations)
- Code/config used
- Conclusions

### Proposing Research Directions

See [TODO list](todo.md) for:
- Planned experiments
- Open questions
- Research priorities

## Code Review Process

All contributions go through code review:

1. **Automated checks**: CI/CD runs tests and linting
2. **Maintainer review**: Code quality and design
3. **Feedback**: Request changes if needed
4. **Approval**: Merge when ready

**Review criteria**:
- Correctness
- Code quality
- Test coverage
- Documentation
- Performance impact

## Community Guidelines

### Be Respectful

- Welcoming to all contributors
- Constructive feedback
- Assume good intentions
- Focus on ideas, not people

### Ask Questions

- No question is too basic
- Use GitHub Discussions for questions
- Check existing issues first
- Provide context and examples

### Give Credit

- Acknowledge contributions
- Link to related work
- Cite relevant papers
- Thank reviewers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create an Issue
- **Security**: Email (add contact info)

## Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes
- Documentation acknowledgments

## Next Steps

- Check [TODO list](todo.md) for contribution ideas
- Review [open issues](https://github.com/georgepearse/visual-next-token/issues)
- Join discussions on GitHub
- Share your experiments!

Thank you for contributing to Visual Next Token! 🚀
