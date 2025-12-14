import unittest
from pathlib import Path
import shutil
import tempfile
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collectors.exporter import JSONExporter, JSONExportError

class TestExporterPathValidation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.exporter = JSONExporter(output_directory=self.test_dir)
        
    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except OSError:
            pass
            
    def test_path_traversal_parent_directory(self):
        """Test that paths attempting to traverse to parent are rejected."""
        # ../output_malicious
        malicious_path = str(Path(self.test_dir).parent / "output_malicious")
        
        with self.assertRaises(JSONExportError) as cm:
            self.exporter._validate_custom_path(malicious_path)
        self.assertIn("escapes allowed directory", str(cm.exception))

    def test_path_traversal_complex(self):
        """Test complex traversal attempts."""
        # /tmp/allowed/../../../etc/passwd
        malicious_path = str(Path(self.test_dir) / ".." / ".." / "etc" / "passwd")
        
        with self.assertRaises(JSONExportError) as cm:
            self.exporter._validate_custom_path(malicious_path)
        self.assertIn("escapes allowed directory", str(cm.exception))

    def test_valid_subdir(self):
        """Test that valid subdirectories are accepted."""
        valid_path = str(Path(self.test_dir) / "subdir" / "data.json")
        # Should not raise
        self.exporter._validate_custom_path(valid_path)

    def test_valid_exact_dir(self):
        """Test that exact directory is accepted."""
        valid_path = str(Path(self.test_dir) / "data.json")
        # Should not raise
        self.exporter._validate_custom_path(valid_path)
        
    def test_broken_exception_logic_fixed(self):
        """Verify that general exceptions are wrapped correctly."""
        # Force a TypeError or similar by mocking (or simplest way: unlikely path error)
        # Using a really invalid path that might cause resolve to fail or something else?
        # Actually simplest is to ensure JSONExportError is raised for invalid inputs without crashing
        
        # Test 1: Exception wrapping check logic fix
        # If we pass something that causes resolve() to fail significantly
        try:
            # Null bytes usually raise ValueError or comparable in path
            self.exporter._validate_custom_path("\0")
        except JSONExportError as e:
            # Should be wrapped
            self.assertIn("Invalid output path", str(e))
        except Exception as e:
            self.fail(f"Should have raised JSONExportError, got {type(e)}")
        else:
            self.fail("Expected JSONExportError was not raised")

if __name__ == "__main__":
    unittest.main()
