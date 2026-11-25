# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek-V4.1 engram table: FP8 rows with UE8M0 scales in host memory, read by the GPU through UVA.

Taken from the engine's `ShardedFp8MultiHeadEmbedding` (`_torch/modules/engram/engram.py`) so the modeling_v2
targets own it: the row/head sharding, the pinned or NUMA-placed host allocation, and the node-wide sharing of
one copy between the processes of a node (`node_share.py`, `host_memory.py`). The gather kernel is the op
`ops/engram/engram_lookup.py`. The packed-row entry points of the engine class (used by its peer-memory
exchange) are not carried over.
"""

import ctypes
import hashlib
import math
import os
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from cuda.bindings import runtime as cudart
from torch import nn

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_lookup import (
    engram_lookup,
)
from tensorrt_llm._utils import CUASSERT, prefer_pinned
from tensorrt_llm.logger import logger

from .host_memory import (
    HostNumaMem,
    HostNumaOutOfMemoryError,
    HostRegisteredMem,
    ImportedHostMem,
    host_numa_mem,
    host_numa_node,
    node_gpu_numa_nodes,
)
from .node_share import NodeShare


def _bucket_offsets(list_of_N: List[int]) -> torch.Tensor:
    """Return per-head base rows shared by the dense and FP8 table layouts."""
    offsets = [0]
    for n in list_of_N[:-1]:
        offsets.append(offsets[-1] + n)
    return torch.tensor(offsets, dtype=torch.long)


class _CudaArrayInterfaceView:
    """Expose a device pointer as bytes; NumPy typestr has no FP8 spelling."""

    def __init__(self, ptr: int, shape: Tuple[int, ...]):
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": "|u1",
            "data": (int(ptr), False),
            "strides": None,
            "version": 3,
        }


def _uva_device_view(host_tensor: torch.Tensor) -> torch.Tensor:
    """Return a zero-copy CUDA view with the pinned tensor's shape and dtype.

    CUDA-allocated pinned memory shares its host and device address under UVA.
    The byte view requires contiguous single-byte storage to preserve strides.
    """
    if host_tensor.device.type != "cpu":
        raise ValueError(
            f"A UVA device view requires a pinned CPU tensor, got {host_tensor.device}."
        )
    if not host_tensor.is_pinned():
        raise ValueError(
            "A UVA device view requires page-locked host memory; this tensor is "
            "ordinary pageable memory, whose host address is not valid on the device."
        )
    if not host_tensor.is_contiguous():
        raise ValueError("A UVA device view requires a contiguous tensor.")
    if host_tensor.element_size() != 1:
        raise ValueError(
            f"A UVA device view is restricted to single-byte dtypes, got "
            f"{host_tensor.dtype} ({host_tensor.element_size()} bytes)."
        )
    view = torch.as_tensor(
        _CudaArrayInterfaceView(host_tensor.data_ptr(), host_tensor.shape),
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    return view.view(host_tensor.dtype)


def _host_numa_table(shape: Tuple[int, ...], dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Prefer host-NUMA VMM, with a registered host mapping if its node is full."""
    device = torch.cuda.current_device()
    numa_node = host_numa_node(device)
    if numa_node is None:
        return None
    nbytes = math.prod(shape) * dtype.itemsize
    # The driver API maps memory in the context current on this thread.
    CUASSERT(cudart.cudaSetDevice(device))
    try:
        host = host_numa_mem(nbytes, device, numa_node)
    except HostNumaOutOfMemoryError as error:
        if not _can_use_host_pointer(device):
            logger.warning(f"Engram: {error}; retrying with the default pinned allocator")
            return None
        logger.warning(
            f"Engram: {error}; retrying with CUDA-registered host memory under the process NUMA "
            "policy. Pages may span NUMA nodes; lookup performance may be lower."
        )
        host = HostRegisteredMem(nbytes, device)
    return _host_bytes(host, nbytes).view(dtype).view(shape)


def _can_use_host_pointer(device: int) -> bool:
    """``device`` reads registered host memory at its host address."""
    (supported,) = CUASSERT(
        cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrCanUseHostPointerForRegisteredMem, device
        )
    )
    return bool(supported)


def _host_bytes(host: object, nbytes: int) -> torch.Tensor:
    """The first ``nbytes`` of a host mapping as a CPU tensor that keeps the mapping alive."""
    raw = (ctypes.c_uint8 * nbytes).from_address(host.address)
    # The tensor's storage keeps ``raw``, and so the mapping, alive.
    raw.host_mem = host
    return torch.frombuffer(raw, dtype=torch.uint8)


# A node's workers reach a table seconds apart; one off the home node waits for home owners this long
# in all, so a node without one costs the wait once per process.
_OFF_HOME_CLAIM_WAIT_S = 120  # 2 min
_off_home_waited_s = 0.0


def _join_off_home(share: NodeShare) -> tuple[list[int], dict] | None:
    """``share.join`` for a worker off the table's home node, out of this process's wait budget."""
    global _off_home_waited_s
    start = time.monotonic()
    try:
        return share.join(max(0.0, _OFF_HOME_CLAIM_WAIT_S - _off_home_waited_s))
    finally:
        _off_home_waited_s += time.monotonic() - start


def _shared_table_home_node(own: Optional[int], tp_rank: int, tp_size: int) -> Optional[int]:
    """Host NUMA node a node-shared table belongs on: the owner's own (``own``) if the split has a
    shard per GPU, else the ``tp_rank``-th GPU host node (modulo), as workers on several may share it."""
    gpu_nodes = node_gpu_numa_nodes()
    homes = sorted({node for node in gpu_nodes if node >= 0})
    if own is None or len(homes) < 2 or tp_size >= len(gpu_nodes):
        return own
    return homes[tp_rank % len(homes)]


def _shareable_host_mem(nbytes: int, device: int) -> Union[HostNumaMem, HostRegisteredMem]:
    """Host memory other processes can map: VMM on the device's host NUMA node, else a memfd."""
    numa_node = host_numa_node(device)
    if numa_node is not None:
        try:
            return host_numa_mem(nbytes, device, numa_node, shareable=True)
        except HostNumaOutOfMemoryError as error:
            logger.warning(
                f"Engram: {error}; sharing CUDA-registered host memory under the process NUMA "
                "policy instead. Pages may span NUMA nodes; lookup performance may be lower."
            )
    if not _can_use_host_pointer(device):
        raise RuntimeError(f"device {device} cannot read registered host memory in place")
    return HostRegisteredMem(nbytes, device, shareable=True)


def _import_host_mem(
    fds: List[int], meta: Dict, nbytes: int, device: int
) -> Union[ImportedHostMem, HostRegisteredMem]:
    """Map the memory a node-share owner published, which must hold ``nbytes``."""
    try:
        if len(fds) != 1 or meta["size"] < nbytes:
            raise RuntimeError(f"its owner published {len(fds)} handles of {meta['size']} bytes")
        if meta["granularity"] is not None:
            return ImportedHostMem(fds[0], meta["size"], meta["granularity"], device)
        if not _can_use_host_pointer(device):
            raise RuntimeError(f"device {device} cannot read registered host memory in place")
        return HostRegisteredMem(meta["size"], device, fd=fds[0])
    finally:
        for fd in fds:
            os.close(fd)


def _pinned_host_table(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Allocate a page-locked host table for random-row UVA gathers.

    With base pages, nearly every row of a multi-GB table misses the GPU's
    address-translation caches, capping the gather far below the host-link
    bandwidth, so tables are mapped in large pages with ``HostNumaMem``. If that
    NUMA node cannot satisfy the allocation, a CUDA-registered anonymous mapping
    uses the process NUMA policy and requests transparent huge pages. Devices
    without host-NUMA virtual memory management use the default pinned allocator,
    as does construction under an init mode that defers host allocations to the
    meta device; such tables are materialized later through this function.
    """
    if not torch.empty(0, dtype=dtype, device="cpu").is_meta:
        table = _host_numa_table(shape, dtype)
        if table is not None:
            return table
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=prefer_pinned())


def _multi_processor_count(device_index: int) -> int:
    """Cache the fixed per-device SM count to avoid repeated property queries."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count



class ShardedFp8MultiHeadEmbedding(nn.Module):
    """Sharded E4M3 n-gram rows with one E8M0 scale per ``block_size`` lanes.

    With TP > 1 and ``num_heads % tp_size == 0``, each rank owns contiguous heads
    and their buckets. Lookup returns local heads for the caller's all-gather.
    Otherwise, ranks own contiguous row ranges, return zero for unowned rows,
    and the caller combines full-width contributions with a sum all-reduce.
    ``shard_heads`` selects the collective in ``DeepseekV41Engram.precompute``.

    Both shard layouts stay in pinned host memory; lookup reads them through UVA.
    The default scale block size is 32 lanes. Processes of one node and user that pass
    the same ``share_key`` map one copy, which ``load_weights`` elects an owner to load;
    the others' copy is complete after ``post_load_weights``.
    """

    def __init__(
        self,
        list_of_N: List[int],
        D: int,
        block_size: int = 32,
        dtype: Optional[torch.dtype] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        share_key: Optional[str] = None,
    ):
        super().__init__()
        if D % block_size != 0:
            raise ValueError(
                f"Engram embedding dim {D} is not a multiple of the fp8 block size "
                f"{block_size}; the checkpoint stores one exponent per block, so a "
                "partial trailing block has no scale to dequantize it with."
            )
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        self.block_size = block_size
        # The engine's table attribute its readers take the row format from (stock's MXFP4 Engram tables, #22414): these
        # rows are E4M3 bytes with one exponent per block.
        self.mxfp4 = False
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        # The HF loader fills pinned allocations in place; other loaders may replace them.
        self._requires_standard_hf_loading = True
        # Rebuild UVA views when host addresses change to avoid stale pointers.
        self._views: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._views_src: Optional[Tuple[int, int]] = None

        self.num_embeddings = sum(list_of_N)
        # Head all-gather requires equal widths; use row shards for uneven splits.
        self.shard_heads = tp_size > 1 and self.num_heads % tp_size == 0
        offsets = _bucket_offsets(list_of_N)

        if self.shard_heads:
            heads_per_rank = self.num_heads // tp_size
            self.head_start = tp_rank * heads_per_rank
            head_end = self.head_start + heads_per_rank
            self.num_local_heads = heads_per_rank
            self.vocab_start_idx = int(offsets[self.head_start])
            self.vocab_end_idx = self.vocab_start_idx + sum(list_of_N[self.head_start : head_end])
            self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
            # Owned heads use local row offsets; both layouts register the same buffer.
            offsets = offsets[self.head_start : head_end] - self.vocab_start_idx
        else:
            self.head_start = 0
            self.num_local_heads = self.num_heads
            # Match the reference's ceil-divided row ranges, with a short final shard.
            self.part_num_embeddings = (self.num_embeddings + tp_size - 1) // tp_size
            self.vocab_start_idx = tp_rank * self.part_num_embeddings
            self.vocab_end_idx = min(
                self.vocab_start_idx + self.part_num_embeddings, self.num_embeddings
            )

        self.register_buffer("offsets", offsets)

        self._share_name: Optional[str] = None
        self._share_spec = ""
        # The owner of a node-shared table serves it while this module lives.
        self._share: Optional[NodeShare] = None
        # An importer's checkpoint rows until its owner marks the table ready.
        self._share_source: Optional[Dict] = None
        if share_key is not None:
            self._share_spec = (
                f"{share_key} heads {self.num_heads} rows {self.num_embeddings} "
                f"tp {tp_rank}/{tp_size} dim {D} block {block_size}"
            )
            digest = hashlib.blake2b(f"{os.getuid()} {self._share_spec}".encode(), digest_size=16)
            self._share_name = f"trtllm-engram-{digest.hexdigest()}"

        # Override the model's CUDA construction context for table storage.
        if not prefer_pinned():
            raise RuntimeError("Engram tables require the pinned-memory runtime policy")
        weight_shape = (self.part_num_embeddings, D)
        scale_shape = (self.part_num_embeddings, D // block_size)
        if share_key is None:
            weight = _pinned_host_table(weight_shape, torch.float8_e4m3fn)
            scale = _pinned_host_table(scale_shape, torch.float8_e8m0fnu)
        else:
            # load_weights places a node-shared table once its owner is elected.
            weight = torch.empty(weight_shape, dtype=torch.float8_e4m3fn, device="meta")
            scale = torch.empty(scale_shape, dtype=torch.float8_e8m0fnu, device="meta")
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)

    def _materialize_pinned(self) -> None:
        """Materialize meta shards directly on the host, without an HBM staging copy."""
        for name in ("weight", "scale"):
            param = self._parameters[name]
            if param.device.type == "cpu" and param.is_pinned():
                continue
            if param.device.type != "meta":
                raise RuntimeError(
                    f"Engram {name} must be meta or pinned CPU memory, got {param.device}"
                )
            if not prefer_pinned():
                raise RuntimeError("Engram tables require the pinned-memory runtime policy")
            pinned = nn.Parameter(
                _pinned_host_table(tuple(param.shape), param.dtype), requires_grad=False
            )
            if not pinned.is_pinned():
                raise RuntimeError(f"Engram {name} allocation is not pinned")
            self.register_parameter(name, pinned)

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> "ShardedFp8MultiHeadEmbedding":
        """Preserve table dtypes and pinned host addresses across model transforms."""
        # Materialization and model.to("cuda") must preserve pinned tables while
        # offsets and other device buffers still follow the model.
        host_parameters = {name: self._parameters.pop(name) for name in ("weight", "scale")}
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            self._parameters.update(host_parameters)
        if self._share_name is None:
            self._materialize_pinned()
        return result

    def load_weights(self, weights: List[Dict]) -> None:
        """Load this rank's rows from full-height checkpoint weights and scales.

        Slice safetensors lazily so unowned rows are not read into host memory.
        """
        weights = weights[0] if isinstance(weights, list) else weights
        if self._share_name is not None and self._load_node_shared(weights):
            return
        self._materialize_pinned()
        self._load_rows(weights)

    def _load_node_shared(self, weights: Dict) -> bool:
        """Map the node's copy of this table, or allocate, serve and fill it if this process is
        elected its owner; False, after one warning, if this process must load a private copy.

        The owner serves the memory before it fills it, and importers wait for it only in
        ``post_load_weights``: a rank of a multi-rank worker never waits on another worker's table
        while its own peers may still need it in a collective of the weight load.
        """
        share = NodeShare(self._share_name)
        device = torch.cuda.current_device()
        # The driver API maps memory in the context current on this thread.
        CUASSERT(cudart.cudaSetDevice(device))
        try:
            own = host_numa_node(device)
            home = _shared_table_home_node(own, self.tp_rank, self.tp_size)
            # CUDA allocates host-NUMA memory only on the owner's node, so off-home workers let a home one claim.
            received = _join_off_home(share) if home != own else share.join(0.0)
            if received is None:
                host = _shareable_host_mem(self._table_bytes(), device)
                registered = isinstance(host, HostRegisteredMem)
                # A memfd has no VMM granularity; importers map it instead.
                share.publish(
                    [host.export_fd()],
                    {"size": host.size, "granularity": None if registered else host.granularity},
                )
            else:
                host = _import_host_mem(*received, self._table_bytes(), device)
            views = self._table_views(host)
        except (OSError, RuntimeError, ValueError) as error:
            share.close()
            logger.warning(
                f"Engram: cannot share table ({self._share_spec}): {error}; loading a private copy"
            )
            return False
        self._share = share
        for name, view in zip(("weight", "scale"), views):
            self.register_parameter(name, nn.Parameter(view, requires_grad=False))
        if received is not None:
            # Checked against this process's checkpoint once the owner marks the table ready.
            self._share_source = weights
            return True
        try:
            self._load_rows(weights)
        except BaseException:
            # Processes waiting for this owner elect another one.
            share.close()
            raise
        share.mark_ready()
        place = "in registered memory" if registered else f"on NUMA node {host.numa_node}"
        logger.info(
            f"Engram: serving node-shared table ({self._share_spec}), "
            f"{host.size / (1 << 30):.1f} GiB {place}"
        )
        return True

    def post_load_weights(self) -> None:
        self.wait_shared_ready()

    def wait_shared_ready(self) -> None:
        """Importer of a node-shared table: wait until its owner marks the table ready and check
        it against this process's checkpoint; elect again if the owner left first, and load a
        private copy if the rows differ or this process cannot share."""
        while self._share_source is not None:
            weights, self._share_source = self._share_source, None
            if self._share.wait_ready():
                if self._rows_match((self.weight.data, self.scale.data), weights):
                    logger.info(
                        f"Engram: mapped node-shared table ({self._share_spec}); "
                        "checkpoint rows match"
                    )
                    return
                logger.warning(
                    f"Engram: cannot share table ({self._share_spec}): its rows differ from this "
                    "process's checkpoint; loading a private copy"
                )
            else:
                logger.info(
                    f"Engram: the owner of node-shared table ({self._share_spec}) left before "
                    "filling it; electing again"
                )
                self._release_shared()
                if self._load_node_shared(weights):
                    continue
            self._release_shared()
            self._materialize_pinned()
            self._load_rows(weights)

    def _release_shared(self) -> None:
        """Drop this process's mapping of a node-shared table; the table is meta until reloaded."""
        if self._share is not None:
            self._share.close()
            self._share = None
        for name in ("weight", "scale"):
            param = self._parameters[name]
            meta = torch.empty(param.shape, dtype=param.dtype, device="meta")
            self.register_parameter(name, nn.Parameter(meta, requires_grad=False))
        self._views = self._views_src = None

    def _table_bytes(self) -> int:
        return self.part_num_embeddings * (
            self.embedding_dim + self.embedding_dim // self.block_size
        )

    def _table_views(self, host: object) -> Tuple[torch.Tensor, torch.Tensor]:
        """Weight rows, then scale rows, of one host mapping."""
        data = _host_bytes(host, self._table_bytes())
        split = self.part_num_embeddings * self.embedding_dim
        return (
            data[:split].view(torch.float8_e4m3fn).view(self.part_num_embeddings, -1),
            data[split:].view(torch.float8_e8m0fnu).view(self.part_num_embeddings, -1),
        )

    def _rows_match(self, views: Tuple[torch.Tensor, torch.Tensor], weights: Dict) -> bool:
        """64 spread rows of this shard, read by the GPU as lookups read them, equal the checkpoint."""
        start = self.vocab_start_idx
        count = self.vocab_end_idx - start
        picks = sorted({i * (count - 1) // 63 for i in range(64)})
        index = torch.tensor(picks, device="cuda")
        for name, view in zip(("weight", "scale"), views):
            rows = [torch.as_tensor(weights[name][start + row : start + row + 1]) for row in picks]
            expected = torch.cat(rows).to(view.dtype).view(torch.uint8)
            actual = _uva_device_view(view).view(torch.uint8)[index]
            if not torch.equal(actual, expected.to(actual.device)):
                return False
        return True

    def _load_rows(self, weights: Dict) -> None:
        start, end = self.vocab_start_idx, self.vocab_end_idx
        for name, param in (("weight", self.weight), ("scale", self.scale)):
            src = weights[name]
            shard = src[start:end]
            if not isinstance(shard, torch.Tensor):
                shard = torch.as_tensor(shard)
            rows = shard.shape[0]
            if rows != param.shape[0]:
                # Only the final row shard may be short by the ceil-divide remainder.
                # Other mismatches violate the hash bucket layout.
                if self.shard_heads or self.tp_rank != self.tp_size - 1 or rows > param.shape[0]:
                    raise ValueError(
                        f"Engram {name}: rank {self.tp_rank}/{self.tp_size} expected "
                        f"{param.shape[0]} rows in [{start}, {end}) of a "
                        f"{self.num_embeddings}-row table but the checkpoint tensor "
                        f"yielded {rows}."
                    )
                param.data[:rows].copy_(shard.to(param.dtype))
                # Valid hashes cannot reach padding; zero it defensively.
                param.data[rows:].zero_()
            else:
                param.data.copy_(shard.to(param.dtype))

    def storage(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached UVA ``(weight, scale)`` views, refreshed if host addresses change."""
        self.wait_shared_ready()
        src = (self.weight.data_ptr(), self.scale.data_ptr())
        if self._views_src != src:
            self._views = (
                _uva_device_view(self.weight.data),
                _uva_device_view(self.scale.data),
            )
            self._views_src = src
        assert self._views is not None
        return self._views

    def forward(self, input_ids: torch.Tensor, background: bool = False) -> torch.Tensor:
        """``[T, num_heads]`` bucket-local indices -> ``[T, num_local_heads, D]``.

        Head shards return only owned heads; row shards return all heads with
        unowned rows zeroed. The caller orders the matching collective after the
        lookup stream. Fused dequantization avoids a full FP32 intermediate.

        ``background`` caps side-stream occupancy to leave SMs for the main stream.
        """
        num_rows = input_ids.shape[0] * self.num_local_heads
        out = torch.empty(
            (input_ids.shape[0], self.num_local_heads, self.embedding_dim),
            dtype=self.dtype,
            device=input_ids.device,
        )
        if num_rows == 0:
            return out
        weight, scale = self.storage()
        num_sms = _multi_processor_count(input_ids.device.index)
        engram_lookup(
            weight.view(torch.uint8),
            scale.view(torch.uint8),
            input_ids,
            self.offsets,
            out,
            head_start=self.head_start,
            block_size=self.block_size,
            # Hashes must stay within each head's bucket. With rebased offsets,
            # head shards own every lookup and need no ownership mask.
            masked=not self.shard_heads,
            vocab_start=self.vocab_start_idx,
            vocab_end=self.vocab_end_idx,
            num_ctas=max(1, num_sms // 2) if background else num_sms,
        )
        return out
