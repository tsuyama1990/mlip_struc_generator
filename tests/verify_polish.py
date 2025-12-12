
import logging
import unittest
from unittest.mock import MagicMock
from nnp_gen.generators.ionic import validate_element
from nnp_gen.web_ui.tabs.config_tab import ConfigViewModel

class TestPolishImprovements(unittest.TestCase):

    def test_ionic_validation_rejects_X(self):
        """Test that element 'X' is rejected."""
        print("Testing ionic validation...")
        with self.assertRaises(ValueError) as cm:
            validate_element("X")
        print(f"Caught expected error: {cm.exception}")
        self.assertIn("'X' is a dummy element", str(cm.exception))

    def test_ui_progress_parsing(self):
        """Test that UI parses MD progress logs."""
        print("Testing UI progress parsing...")
        vm = ConfigViewModel()
        vm.job_manager = MagicMock()
        vm._last_job_id = "test_job"
        vm.job_manager.get_status.return_value = "running"
        
        # Simulating log content
        log_content = """
        [INFO] Step 1: Structure Generation
        [INFO] Step 2: Exploration
        [INFO] MD Progress: 100/1000
        """
        vm.job_manager.get_log_content.return_value = log_content

        # Initial state
        vm.progress_value = 0
        
        # Run update
        vm.update_logs()
        
        # "Step 2" sets base to 50.
        # MD Progress 100/1000 = 10%.
        # Range is 50 -> 75 (span 25).
        # 10% of 25 is 2.5.
        # Total = 50 + 2.5 = 52.5 -> 52.
        
        print(f"Progress Value: {vm.progress_value}")
        self.assertEqual(vm.progress_value, 52)

if __name__ == "__main__":
    unittest.main()
