import pytest
from nnp_gen.web_ui.tabs.config_tab import ConfigViewModel
from nnp_gen.core.config import AppConfig

def test_config_vm_to_pydantic():
    """Test conversion from ViewModel to Pydantic Config."""
    vm = ConfigViewModel()
    vm.system_type = "alloy"
    vm.elements_input = "Cu, Au"
    vm.alloy_lattice_constant = 3.61
    vm.temperature = 500

    config = vm.get_pydantic_config()

    assert isinstance(config, AppConfig)
    assert config.system.type == "alloy"
    assert config.system.elements == ["Cu", "Au"]
    assert config.system.lattice_constant == 3.61
    assert config.exploration.temperature == 500.0

def test_config_vm_dynamic_switch():
    """Test switching system type."""
    vm = ConfigViewModel()
    vm.system_type = "molecule"
    vm.molecule_smiles = "C6H6"
    vm.elements_input = "C, H" # Technically molecules infer elements, but config requires list

    config = vm.get_pydantic_config()
    assert config.system.type == "molecule"
    assert config.system.smiles == "C6H6"

def test_run_pipeline_trigger(mocker):
    """Test that running pipeline triggers JobManager."""
    vm = ConfigViewModel()

    # Mock JobManager singleton inside VM
    # Since JobManager is a singleton, we can mock the class method or the instance
    mock_submit = mocker.patch.object(vm.job_manager, 'submit_job', return_value="job_123")

    vm.run_pipeline()

    assert mock_submit.call_count == 1
    assert vm._last_job_id == "job_123"
    assert "Job job_123 submitted" in vm.status_message
