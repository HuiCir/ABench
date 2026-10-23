"""KernelBench problem files (Ouyang et al. 2025, github.com/ScalingIntelligence/KernelBench, MIT licence) for K-KB-001
and K-FUS-002: the selected files are copied from a clone at COMMIT, checked against their sha256, renamed
L<level>_<file> and listed in problems.json with the shape overrides that keep every input within 1 GiB of fp32
(overrides halve the largest size global of a problem until its inputs fit; computed from get_inputs() shapes).
The held-out part is only seeds: weights, the five input distributions and the post-timing check per problem, from a secret seed."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from prepare.common import Frozen, sha256_file

REPO = "https://github.com/ScalingIntelligence/KernelBench"
COMMIT = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
N_SEED_COLS = 7                                    # weights, native, normal, scaled, sparse, large, post_timing
FILES = {                                         # path in the repository's KernelBench/ dir: (sha256, overrides)
    "level1/2_Standard_matrix_multiplication_.py": ("136a33c4d598d07cd7931ef8c0eff4e683c221af6fc43c2d89b6632d3f9902a0", {}),
    "level1/3_Batched_matrix_multiplication.py": ("310fb3bdf4f69ea178a5b5868bf204e83196cba71e4750be3fbbcb68b78d37f7", {'n': 1024}),
    "level1/4_Matrix_vector_multiplication_.py": ("5f1476dfcbd291f393d469a2645ff628f1e393bd720c2fe36d5fa4f9181dd8a1", {'K': 65536}),
    "level1/5_Matrix_scalar_multiplication.py": ("f4d52425a8f72b4b22375494064f2ab1675e2caf29376c47a102211ce66d6987", {'M': 16384}),
    "level1/9_Tall_skinny_matrix_multiplication_.py": ("06a133f60f0d704b70014b210031e3937d7a183c90ea11800357ce6b55c89c18", {}),
    "level1/18_Matmul_with_transposed_both.py": ("58f61b728890855d009329dbdbd0da4ac7d9390e46f7199b27dc353dd512ad19", {}),
    "level1/19_ReLU.py": ("cbbfc9409662168ee7a5d3e7f7a59bf56e0faf9d763197e7f6a41fb5942dd63a", {'dim': 49152}),
    "level1/20_LeakyReLU.py": ("5dea58d2392f446e0139e6cfb58bec3a1ac23cee90bb5c3e8a6ddab2909b7fbf", {'dim': 49152}),
    "level1/21_Sigmoid.py": ("8f3ae78284ddd70b5c2fff66e066e967780245dee262fbf8cd2432498833796a", {'dim': 49152}),
    "level1/28_HardSigmoid.py": ("fc773884c6c52af0fd6d41c610ca5bd15f66c4c47f0d4e9fad24564afea4c50a", {'dim': 49152}),
    "level1/32_HardTanh.py": ("b73a2ed250f2d9fec69a5899ddae3a15f41a9225b73a20ef6de5b8f8ef05fcf5", {'dim': 49152}),
    "level1/35_GroupNorm_.py": ("e4462f85b3eaadd05bdce8655305f5bcddc49dd31ef0a3bba1b85a1cbb2d19bb", {'dim1': 128, 'dim2': 256}),
    "level1/36_RMSNorm_.py": ("8f6ee1b9ee212f3cee1069eff2b7935482713c99596b1024801868065e5a2b1d", {'dim1': 128, 'dim2': 256}),
    "level1/37_FrobeniusNorm_.py": ("cf45f93a2a75102fdcc78a77833cbf6a2f40080fec5c7437ff1471b4092360ba", {'dim1': 128, 'dim2': 256}),
    "level1/38_L1Norm_.py": ("02d29f98ba28001a4274fe713bf0b985ffc088aef1e83581ad11e4dbe239dfda", {'dim': 16383, 'batch_size': 16384}),
    "level1/41_Max_Pooling_1D.py": ("65bbc7b4b8c999990dacd9be04823886fecfec981a5d7484ddcf53302b04584d", {'sequence_length': 16384}),
    "level1/43_Max_Pooling_3D.py": ("c684de6a8afdb3e555952e692f07744531394f2ccd8ea5f34b68793f7f71b709", {'dim1': 64, 'dim2': 64}),
    "level1/44_Average_Pooling_1D.py": ("2c55c057fff2439d7bf30ad1cd33b9e72da6b9996cfc2f2bbde981e114a5b774", {'input_length': 32768}),
    "level1/47_Sum_reduction_over_a_dimension.py": ("537d2ff381740bdb54b5208f82187564f6733a30df918edbbf7a49d64bef17ab", {'dim1': 1024, 'dim2': 2047}),
    "level1/51_Argmax_over_a_dimension.py": ("d6a2057c4c95a9737985b2b8ffda23595f812557c26a2263fec8edd594c61fbe", {'dim1': 1024, 'dim2': 2047}),
    "level1/52_Argmin_over_a_dimension.py": ("e1764caddce33679126f225b4da455a95978be85ccf79a372e3ef549ee35ba8f", {'dim1': 1024, 'dim2': 2047}),
    "level1/58_conv_transposed_3D__asymmetric_input__asymmetric_kernel.py": ("1673075fe1a9d3fa00f625f0059011c9aa7aa864d8030311673dab91fb9b3564", {}),
    "level1/60_conv_standard_3D__square_input__asymmetric_kernel.py": ("3f7204c138ec6d09c25c38f39be1d5b7f43f4fa5270d68fe0323ad9b3149ffb9", {}),
    "level1/64_conv_transposed_1D.py": ("0ad1a7811c8916691ae516c366a1ac888b011cf5699dd8b2e2a569cbef697c75", {'length': 32768}),
    "level1/71_conv_transposed_2D__asymmetric_input__square_kernel.py": ("02bb0b2e7744eb4c7d6576e135400b03da21efa4b999cef2ab4e5a44f49fd295", {}),
    "level1/80_conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__.py": ("4706128a7dfef02da3d2ebb72874eb81db833c6b6369b5413d40985e8ad0ea39", {}),
    "level1/84_conv_depthwise_2D_asymmetric_input_square_kernel.py": ("1af31dab803ef4f4e5b94f7bbe54260ef23930715610d1cbd8e7cdcd88e5c45c", {'width_in': 128}),
    "level1/90_cumprod.py": ("17408793b4846d75bef22fff5c2a16c7d836409eabc0476c8060f75183a64665", {'batch_size': 8192}),
    "level1/92_cumsum_exclusive.py": ("ec1551c196130f5d7fae707f0750016c35b988e76aa3f1657da4347f463ced86", {'batch_size': 8192}),
    "level1/97_ScaledDotProductAttention.py": ("a07ca5135657c2ea29f7b7a66ec1815560858ad69fa88221bfc0f82bd5648091", {'embedding_dimension': 256, 'sequence_length': 256}),
    "level2/6_Conv3d_Softmax_MaxPool_MaxPool.py": ("55322b483d9bf27e0fefcdfd0e25f53172c194da6b47154d40bd39f781c60a46", {}),
    "level2/7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd.py": ("cd0390e07dda478af0316a61bd1a407677992229ffdf3fa857d143e40a8533d7", {}),
    "level2/13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling.py": ("f33a027962c4ffa7b73bc48fe1e26e38d878c21b084e0c2a0a94c0bc290bd4f3", {}),
    "level2/23_Conv3d_GroupNorm_Mean.py": ("9677116401cf1ec4ac00eb19cb6f08762d63b87b137e6ed10f2c8ec3741310fb", {}),
    "level2/25_Conv2d_Min_Tanh_Tanh.py": ("15e3e645a6012d772f1197e988d8a0ea4f191ad2459229fecdbe3f1d9ae344b3", {}),
    "level2/28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py": ("5ed41cc50c8b2dfe65d03e642cc471dc3ad7cf6858608c65b911c235946ef00e", {}),
    "level2/36_ConvTranspose2d_Min_Sum_GELU_Add.py": ("e8a14171f06b41a2bb5067e1397d2b1afd928c95272916b3c16dfff811f98316", {}),
    "level2/37_Matmul_Swish_Sum_GroupNorm.py": ("fe2e8dc5418f1549cb3b6e34535b6f3dee0fc37ecce1b6be19a57ed8733156d4", {}),
    "level2/42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply.py": ("0eeb1e46e5bacfed11e5bfddffe6f4fdfd981332e0fb56c4e67b219d6faba3a6", {}),
    "level2/49_ConvTranspose3d_Softmax_Sigmoid.py": ("9386889fc80859671b9ef7f3bd3dd182bd0fb5b51067f88407c4e48fb1107f79", {}),
    "level2/58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp.py": ("d89693a52be9080caf580e0a21691a19d1cf9ef44c31a9466b6df97178ac9c21", {}),
    "level2/66_Matmul_Dropout_Softmax.py": ("7a8bcefb75586bbf0acfcdeb0e8391c8881548fe9b55844c0033a96e9f898a29", {}),
    "level2/69_Conv2d_HardSwish_ReLU.py": ("442df07c27315075294d997dcd7ce61310a7d41a8dff4033c33ef4cf61b4727e", {}),
    "level2/76_Gemm_Add_ReLU.py": ("88e932ace4f16e0477574c5c99b8d5665393e02f82f664dc99f2e22a2674ea42", {}),
    "level2/83_Conv3d_GroupNorm_Min_Clamp_Dropout.py": ("223d45d8423c55369bd0b1277abca1d358a28ad77cf7661f1c6e602c856a694a", {}),
    "level2/85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py": ("ba63b5b156a347f2f92a607c4a612e2bb1963dbcb9e7f04b3e4f0cb4a294fff3", {}),
    "level2/88_Gemm_GroupNorm_Swish_Multiply_Swish.py": ("eda8a47c825eb3fc49836d6cc0c9cd6ba596cc35ece1d22c871bba198009c6e7", {}),
    "level2/96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp.py": ("164edd0135adff496adade63a772074630ebf2c13feaa450307d98a9728da2e6", {}),
    "level2/98_Matmul_AvgPool_GELU_Scale_Max.py": ("8e5df29ff246f69cde03ab21ab53c706d58cd082014184641332cbb157607eb7", {}),
    "level2/100_ConvTranspose3d_Clamp_Min_Divide.py": ("1f9ce1ae75f5e9986156c1d6e40b525db04d89c11e825c159ef2cc108f7e132e", {}),
}


def sources(level: str) -> list[dict]:
    return [{"name": f"KernelBench {level} problem files (Ouyang et al. 2025)", "url": f"{REPO}/tree/{COMMIT}/KernelBench/{level}",
             "license": "MIT", "note": f"git clone {REPO} RAW/KernelBench-repo && git -C RAW/KernelBench-repo checkout {COMMIT}"}]


def _name(path: str) -> str:
    lv, f = path.split("/")
    return f"L{lv[-1]}_{Path(f).stem}"


def _problems(level: str) -> list[tuple[str, str, dict]]:
    return [(k, _name(k), ov) for k, (_, ov) in FILES.items() if k.startswith(level + "/")]


def _seeds(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(1, 2**31 - 1, size=(n, N_SEED_COLS), dtype=np.int64)


def build_level(raw: Path, level: str, seed_env: str) -> Frozen:
    seed = int(os.environ.get(seed_env, "0"))
    if not seed:
        raise SystemExit(f"set {seed_env} to a secret integer (weight and input seeds)")
    src = raw / "KernelBench-repo" / "KernelBench"
    files, listing, sums = {}, [], {"commit": COMMIT}
    for path, name, ov in _problems(level):
        p = src / path
        h = sha256_file(p)
        if h != FILES[path][0]:
            raise SystemExit(f"{p}: sha256 {h} differs from the pinned {FILES[path][0]}")
        files[f"{name}.py"] = p
        listing.append({"name": name, "file": f"{name}.py", "overrides": ov})
        sums[name] = h
    tmp = Path(tempfile.mkdtemp(prefix="kb-")) / "problems.json"
    tmp.write_text(json.dumps(listing, indent=1) + "\n", encoding="utf-8")
    files["problems.json"] = tmp
    return Frozen(public={}, hidden={"seeds": _seeds(seed, len(listing))}, sources_sha256=sums, raw_files=files)


def synthetic_level(level: str, n: int) -> Frozen:
    """Plumbing test without the repository: two tiny problems in KernelBench format."""
    d = Path(tempfile.mkdtemp(prefix="kb-syn-"))
    probs = {"S_relu": "torch.relu(x)", "S_scale": "x * 2.0"}
    listing = []
    for name, expr in probs.items():
        (d / f"{name}.py").write_text(
            "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n"
            f"    def forward(self, x):\n        return {expr}\n\nn = 1024\n\ndef get_inputs():\n    return [torch.randn(n, n)]\n\n"
            "def get_init_inputs():\n    return []\n", encoding="utf-8")
        listing.append({"name": name, "file": f"{name}.py", "overrides": {}})
    (d / "problems.json").write_text(json.dumps(listing), encoding="utf-8")
    files = {f"{k}.py": d / f"{k}.py" for k in probs} | {"problems.json": d / "problems.json"}
    return Frozen(public={}, hidden={"seeds": _seeds(1, len(listing))}, sources_sha256={"synthetic": level}, raw_files=files)
