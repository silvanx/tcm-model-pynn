import neo
from pathlib import Path
import numpy as np

def neo_analogsignals_identical(signal1, signal2):
    return False

def neo_spiketrains_identical(train1, train2):
    assert len(train1) == len(train2)
    struct_info_fields = [
        "t_start",
        "t_stop",
        "sampling_rate",
        "units",
        "annotations"]
    for i in range(len(train1)):
        assert np.all(train1[i].times == train2[i].times)
        for field in struct_info_fields:
            assert getattr(train1[i], field) == getattr(train2[i], field)
        
    return True

def neo_segments_identical(seg1, seg2):
    assert seg1.annotations == seg2.annotations
    assert seg1.description == seg2.description
    assert seg1.name == seg2.name
    assert neo_spiketrains_identical(seg1.spiketrains, seg2.spiketrains)
    assert len(seg1.analogsignals) == len(seg2.analogsignals)
    for i in range(len(seg1.analogsignals)):
        assert neo_analogsignals_identical(seg1.analogsignals, seg2.analogsignals)
    
    return True

def neo_blocks_identical(block1, block2):
    assert block1.name == block2.name
    assert block1.description == block2.description
    assert block1.annotations == block2.annotations
    assert len(block1.segments) == len(block2.segments)
    for i in range(len(block1.segments)):
        seg1 = block1.segments[i]
        seg2 = block2.segments[i]
        if not neo_segments_identical(seg1, seg2):
            return False
    return True

def neo_structs_almost_identical(file1, file2):
    struct1 = neo.NeoMatlabIO(file1).read()
    struct2 = neo.NeoMatlabIO(file2).read()
    if len(struct1) != len(struct2):
        return False
    for i in range(len(struct1)):
        block1 = struct1[i]
        block2 = struct2[i]
        if not neo_blocks_identical(block1, block2):
            return False
    return True


def test_d_layer():
    file_test = Path("Results/D_Layer.mat")
    file_original = Path("Results/original_D_Layer.mat")
    assert neo_structs_almost_identical(file_test, file_original)

def test_s_layer():
    raise NotImplementedError

def test_m_layer():
    raise NotImplementedError

def test_interneurons():
    raise NotImplementedError

def test_tcr_nucleus():
    raise NotImplementedError

def test_tr_nucleus():
    raise NotImplementedError