"""Topology-aware CPU placement for CPU expert inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True)
class CPUPlacement:
    compute_cores: tuple[int, ...]
    shared_core: int
    compute_packages: tuple[int, ...]
    shared_package: int


@dataclass(frozen=True)
class _LogicalCPU:
    cpu: int
    package: int
    physical_core: int
    utilization: float


def _read_int(path: Path, default: int) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return default


def _logical_cpus(sample_interval: float) -> list[_LogicalCPU]:
    utilization = psutil.cpu_percent(percpu=True, interval=sample_interval)
    allowed = os.sched_getaffinity(0)
    result = []
    for cpu in sorted(allowed):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        result.append(
            _LogicalCPU(
                cpu=cpu,
                package=_read_int(topology / "physical_package_id", 0),
                physical_core=_read_int(topology / "core_id", cpu),
                utilization=utilization[cpu] if cpu < len(utilization) else 0.0,
            )
        )
    return result


def select_cpu_placement(total_cores: int, sample_interval: float = 0.2) -> CPUPlacement:
    """Select ``total_cores - 1`` compute CPUs plus one shared CPU.

    Compute CPUs prefer distinct physical cores in one package.  This avoids
    the two scaling hazards in the old lowest-utilization policy: selecting SMT
    siblings as independent cores and spreading a memory-bound expert GEMV
    across NUMA sockets.  The loading/background CPU is kept on an otherwise
    unused physical core, preferably in the compute package.
    """
    logical = _logical_cpus(sample_interval)
    if total_cores < 2:
        raise ValueError("cpu_cores must be at least 2 (1 compute + 1 shared)")
    if total_cores > len(logical):
        raise ValueError(
            f"cpu_cores={total_cores} exceeds the {len(logical)} CPUs allowed by this process"
        )

    # Score a physical core by its busiest sibling: an apparently idle sibling
    # is still a bad choice when its paired hardware thread is occupied.  Use
    # the lowest-numbered allowed sibling as a stable, compact representative.
    physical_members: dict[tuple[int, int], list[_LogicalCPU]] = {}
    for item in logical:
        key = (item.package, item.physical_core)
        physical_members.setdefault(key, []).append(item)

    physical: dict[tuple[int, int], _LogicalCPU] = {}
    for key, members in physical_members.items():
        representative = min(members, key=lambda item: item.cpu)
        physical[key] = _LogicalCPU(
            cpu=representative.cpu,
            package=representative.package,
            physical_core=representative.physical_core,
            utilization=max(item.utilization for item in members),
        )

    by_package: dict[int, list[_LogicalCPU]] = {}
    for item in physical.values():
        by_package.setdefault(item.package, []).append(item)
    for items in by_package.values():
        items.sort(key=lambda item: (item.utilization, item.cpu))

    compute_count = total_cores - 1
    capable_packages = [
        (sum(item.utilization for item in items[:compute_count]), package)
        for package, items in by_package.items()
        if len(items) >= compute_count
    ]

    if capable_packages:
        _, compute_package = min(capable_packages)
        selected = by_package[compute_package][:compute_count]
    else:
        # More compute threads than one socket has physical cores: fill whole
        # packages before using SMT siblings.
        package_order = sorted(
            by_package,
            key=lambda package: (
                sum(item.utilization for item in by_package[package])
                / len(by_package[package]),
                package,
            ),
        )
        selected = []
        for package in package_order:
            selected.extend(by_package[package])
            if len(selected) >= compute_count:
                selected = selected[:compute_count]
                break

    selected_ids = {item.cpu for item in selected}
    selected_physical = {(item.package, item.physical_core) for item in selected}
    compute_packages = {item.package for item in selected}

    shared_candidates = [
        item
        for item in physical.values()
        if item.cpu not in selected_ids
        and (item.package, item.physical_core) not in selected_physical
    ]
    shared_candidates.sort(
        key=lambda item: (
            item.package not in compute_packages,
            item.utilization,
            item.cpu,
        )
    )

    if not shared_candidates:
        # With more logical CPUs requested than physical cores, keep one
        # physical core for DMA/GPU submission when enough SMT slots remain.
        # Otherwise a compute worker and the loading thread share execution
        # resources even though the requested CPU budget permits isolation.
        if os.environ.get("SMOE_RESERVE_LOAD_CORE", "0") == "1":
            reservable = [item for item in selected if
                len(logical) - len(physical_members[(item.package, item.physical_core)])
                >= compute_count]
            if reservable:
                reserved = min(reservable, key=lambda item: (item.utilization, item.cpu))
                selected.remove(reserved)
                selected_ids = {item.cpu for item in selected}
                shared_candidates = [reserved]

    if not shared_candidates:
        # All physical cores are used.  Reserve the least busy unused SMT
        # sibling for loading rather than stealing a compute CPU.
        shared_candidates = [item for item in logical if item.cpu not in selected_ids]
        shared_candidates.sort(key=lambda item: (item.utilization, item.cpu))
    shared = shared_candidates[0]
    shared_key = (shared.package, shared.physical_core)
    isolate_shared = (os.environ.get("SMOE_RESERVE_LOAD_CORE", "0") == "1"
                      and len(logical) - len(physical_members[shared_key]) >= compute_count)

    # If the request exceeds the physical-core count, add unused logical CPUs
    # only after reserving the shared worker CPU.
    if len(selected) < compute_count:
        extras = [
            item
            for item in logical
            if item.cpu not in selected_ids and item.cpu != shared.cpu
            and (not isolate_shared or (item.package, item.physical_core) != shared_key)
        ]
        extras.sort(key=lambda item: (item.utilization, item.cpu))
        selected.extend(extras[: compute_count - len(selected)])

    return CPUPlacement(
        compute_cores=tuple(item.cpu for item in selected),
        shared_core=shared.cpu,
        compute_packages=tuple(sorted({item.package for item in selected})),
        shared_package=shared.package,
    )
