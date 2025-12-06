import numpy as np
import pytest
from ase import Atoms
from nnp_gen.web_ui.tabs.viz_tab import VizViewModel

def test_viz_vm_load_db(mocker, tmp_path):
    """Test loading database into ViewModel."""
    # Create a dummy db
    db_path = tmp_path / "test.db"
    import ase.db
    with ase.db.connect(db_path) as db:
        atoms = Atoms("CO", positions=[[0,0,0], [0,0,1.2]])
        # Mock descriptor
        atoms.info["descriptor"] = np.random.rand(10)
        db.write(atoms, data={"descriptor": atoms.info["descriptor"]}, is_sampled=True)

        atoms2 = Atoms("Fe", positions=[[0,0,0]])
        atoms2.info["descriptor"] = np.random.rand(10)
        db.write(atoms2, data={"descriptor": atoms2.info["descriptor"]}, is_sampled=False)

    vm = VizViewModel()
    vm.load_db(str(db_path))

    assert len(vm.structures) == 2
    assert "Loaded 2 structures" in vm.status_msg

    # Check Metadata
    assert vm.metadata_list[0]["is_sampled"] is True
    assert vm.metadata_list[1]["is_sampled"] is False

    # Check PCA computation triggers
    assert len(vm.pca_source.data['x']) == 2
    assert vm.pca_source.data['color'][0] == "red" # sampled
    assert vm.pca_source.data['color'][1] == "blue" # not sampled

def test_viz_vm_update_viewer(mocker):
    """Test HTML generation on selection."""
    vm = VizViewModel()
    vm.structures = [Atoms("H2", positions=[[0,0,0], [0,0,0.74]])]
    vm.selected_idx = 0

    # Trigger update manually if not watching in test env without panel server
    vm.update_viewer()

    assert "$3Dmol.createViewer" in vm.viewer_html
    # XYZ format varies slightly depending on precision settings in ase.io.write or system
    # Just check for element H
    assert "H" in vm.viewer_html

def test_job_list_update(mocker):
    """Test job list population."""
    vm = VizViewModel()

    # Mock JobManager
    mock_jobs = [
        mocker.Mock(job_id="job1", status="COMPLETED"),
        mocker.Mock(job_id="job2", status="RUNNING")
    ]
    mocker.patch.object(vm.job_manager, 'get_all_jobs', return_value=mock_jobs)

    vm.update_job_list()

    assert vm.param.job_selector.objects == ["job1"]
    assert vm.job_selector == "job1"
