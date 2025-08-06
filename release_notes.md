## OutlierCleaner v1.1.4 Release (2025-08-06)

### New Features
- **Comprehensive Type Hints**: Added complete type annotations to all methods and functions
  - Enhanced IDE support with better autocomplete and error detection
  - Improved code documentation through type annotations
  - MyPy compatibility for static type checking
  - Better developer experience and code maintainability

### Improvements
- Updated dependency management with complete requirements specification
- Enhanced null safety with proper error handling
- Improved code quality and professional standards
- Full backward compatibility maintained

### Dependencies
- Updated requirements.txt with scipy and tqdm dependencies
- Synchronized dependency versions across setup.py and requirements.txt

### Breaking Changes
None - All existing functionality is preserved with full backward compatibility.

---

## OutlierCleaner v1.0.1 Release (Previous)

### Changes
- Fixed author name spelling
- Updated documentation for consistency

### Previous Features (v1.0.0)
- Automatic method selection based on data distribution
- Multiple outlier detection methods (IQR, Z-score, Modified Z-score)
- Advanced distribution analysis and method recommendations
- Progress tracking for batch operations
- Index preservation options
- Outlier tracking and statistics
- Robust handling of edge cases

### Documentation
- Updated README with comprehensive examples using California Housing dataset
- Added detailed method descriptions
- Improved installation and dependency information

### Dependencies
- Updated minimum versions for all dependencies
- Added new dependencies: scipy, tqdm
- Added scikit-learn as optional dependency for examples

### Breaking Changes
None - All existing functionality is preserved with backward compatibility.

### Example Usage
The README now includes a complete example using the California Housing dataset, demonstrating all major features including:
- Distribution analysis
- Automatic method selection
- Progress tracking
- Index preservation
- Outlier tracking
- Visualization tools