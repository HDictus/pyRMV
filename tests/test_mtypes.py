from pyrmv import mtypes


def test_resolve():
    assert mtypes.resolve("PC") == {
        "L2_IPC",
        "L2_TPC:A",
        "L2_TPC:B",
        "L3_TPC:A",
        "L3_TPC:C",
        "L4_TPC",
        "L4_UPC",
        "L5_TPC:A",
        "L5_TPC:B",
        "L5_TPC:C",
        "L5_UPC",
        "L6_BPC",
        "L6_HPC",
        "L6_IPC",
        "L6_TPC:A",
        "L6_TPC:C",
        "L6_UPC",
    }
    assert mtypes.resolve("NBC") == {"L23_NBC", "L4_NBC", "L5_NBC", "L6_NBC"}
    assert mtypes.resolve("BP") == {"L23_BP", "L4_BP", "L5_BP", "L6_BP"}
