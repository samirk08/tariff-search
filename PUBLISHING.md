# Publishing tariff-search Package

## Quick Start: Publish to GitHub

1. **Update package metadata**:
   - Edit `setup.py`: Replace "Your Name" and email
   - Edit `pyproject.toml`: Update author info

2. **Create GitHub repository**:
   ```bash
   # Initialize git (if not already done)
   git init
   
   # Add all files
   git add .
   
   # Commit
   git commit -m "Initial commit of tariff-search package"
   
   # Create repo on GitHub, then:
   git remote add origin https://github.com/YOUR_USERNAME/tariff-search.git
   git push -u origin main
   ```

3. **Users can now install**:
   ```bash
   pip install git+https://github.com/YOUR_USERNAME/tariff-search.git
   ```

## Optional: Publish to PyPI

This allows `pip install tariff-search`:

1. **Create PyPI account**: https://pypi.org/account/register/

2. **Install build tools**:
   ```bash
   pip install build twine
   ```

3. **Build the package**:
   ```bash
   python -m build
   ```

4. **Upload to PyPI**:
   ```bash
   # Test upload first
   twine upload --repository testpypi dist/*
   
   # Then real PyPI
   twine upload dist/*
   ```

## Important Notes

- The Google Drive link in `download.py` is already configured
- Large data files (*.pkl, *.npy) are excluded from git via .gitignore
- Users will download data on first use via Google Drive
- Consider adding GitHub Actions for automated testing/publishing

## Package Structure Ready for Publishing

✅ setup.py - Package configuration
✅ pyproject.toml - Modern Python packaging
✅ README.md - Documentation
✅ LICENSE - MIT license
✅ .gitignore - Excludes data files
✅ Requirements specified
✅ CLI entry point configured
✅ Google Drive download integrated