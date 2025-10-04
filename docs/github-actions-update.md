# GitHub Actions Updates

## Fixed Deprecation Warning

**Problem**: GitHub Actions was failing with the error:
```
This request has been automatically failed because it uses a deprecated version of `actions/upload-artifact: v3`
```

## Changes Made

### Updated Action Versions

| Action | Old Version | New Version | Reason |
|--------|-------------|-------------|---------|
| `actions/upload-artifact` | v3 | v4 | v3 is deprecated |
| `actions/download-artifact` | v3 | v4 | v3 is deprecated |
| `actions/setup-python` | v4 | v5 | Latest stable |
| `pypa/cibuildwheel` | v2.16.2 | v2.21.2 | Latest version with bug fixes |
| `codecov/codecov-action` | v3 | v4 | Latest stable |

### Important Changes in v4 Artifacts

**WARNING - Breaking Change**: `actions/download-artifact@v4` behavior changed:

- **v3**: Downloads artifacts directly to specified path
- **v4**: Downloads each artifact into its own subdirectory by name

**Impact**: The PyPI upload job was updated to handle the new directory structure:

```yaml
- name: Download all artifacts
  uses: actions/download-artifact@v4
  with:
    path: dist/

- name: Flatten directory structure
  run: |
    mkdir -p final_dist/
    # In v4, artifacts are downloaded into subdirectories by name
    find dist/ -name "*.whl" -exec cp {} final_dist/ \;
    find dist/ -name "*.tar.gz" -exec cp {} final_dist/ \;
```

## Verification

After these changes, the GitHub Actions should run without deprecation warnings. You can verify by:

1. **Push to trigger CI**: Push these changes to trigger the workflows
2. **Check Actions tab**: Verify no deprecation warnings appear
3. **Monitor builds**: Ensure wheel building still works correctly

## Next Steps

The CI/CD pipeline is now updated and should work reliably. The workflows will:

1. **Build wheels** for Linux x86_64, Windows x86_64, and macOS (x86_64 + ARM64)
2. **Test wheels** on all target platforms and Python versions
3. **Upload to PyPI** on tagged releases (requires `PYPI_API_TOKEN` secret)

## Troubleshooting

If you encounter issues:

1. **Check the Actions tab** for detailed error logs
2. **Verify artifacts** are being uploaded/downloaded correctly
3. **Test locally** using the scripts in `scripts/` directory:
   ```bash
   ./scripts/build_wheels.sh    # Build wheels locally
   python scripts/test_wheel.py # Test wheel functionality
   ```

## Security Note

The `PYPI_API_TOKEN` secret warning in the linter is expected - this secret needs to be configured in your repository settings for PyPI publishing to work.
