## v2.1.1 (2024-10-22)

### Fix

- trying to fix gitworkflows
- some commit msg

## v2.1.0 (2024-10-15)

### Feat

- cache manager is now a static class and accepts cache-filepath for his methods

### Fix

- __init__.py file now use absolute import which is necessary for package to be successfully imported
- OcrEngine abstract class can now be imported from prepCV package

### Refactor

- setup.py file is removed, thus now pyproject.toml us responsible for package building, small file refactorings.
- refactored __init__.py
