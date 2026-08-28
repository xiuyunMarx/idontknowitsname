"""Worker-side layer-wise weight offload / streaming load.

Mixed into the vLLM GPU worker via ``worker_extension_cls``; the ``lw_*``
methods are reachable with ``AsyncLLM.collective_rpc("lw_...")``.

Parameters are grouped into stages in execution order: pre (embeddings),
one per decoder layer, post (final norm + lm_head). ``lw_load`` re-allocates
the KV cache, then copies the stages back from pinned host memory on a side
stream and signals each one; every layer's forward is gated on its own stage,
so a prefill issued during the load runs right behind the loader front.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class _Stage:
    name: str
    params: list[tuple[str, nn.Parameter]] = field(default_factory=list)
    cpu_ready: threading.Event = field(default_factory=threading.Event)
    gpu_ready: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    nbytes: int = 0


class LayerwiseWorkerExtension:
    def lw_install(self) -> dict:
        model = self.model_runner.get_model()  # type: ignore[attr-defined]
        layers = _find_decoder_layers(model)
        n = len(layers)

        param_stage: dict[int, int] = {}
        for i, layer in enumerate(layers):
            for p in layer.parameters():
                param_stage[id(p)] = i + 1

        stages = [_Stage("pre")] + [_Stage(f"layer{i}") for i in range(n)] + [_Stage("post")]
        seen_layer = False
        for name, p in model.named_parameters():
            s = param_stage.get(id(p))
            if s is None:
                s = n + 1 if seen_layer else 0
            else:
                seen_layer = True
            stages[s].params.append((name, p))
            stages[s].nbytes += p.numel() * p.element_size()

        self._lw_stages = stages
        self._lw_cpu: dict[str, torch.Tensor] = {}
        self._lw_copy_stream = torch.cuda.Stream()
        self._lw_thread: threading.Thread | None = None
        self._lw_error: BaseException | None = None
        self._lw_resident = True
        for st in stages:
            st.cpu_ready.set()

        _gate(model, "forward", lambda: self._lw_wait(0))
        for i, layer in enumerate(layers):
            _gate(layer, "forward", lambda i=i: self._lw_wait(i + 1))
        if hasattr(model, "compute_logits"):
            _gate(model, "compute_logits", lambda: self._lw_wait(n + 1))

        return {
            "num_stages": len(stages),
            "total_gb": sum(st.nbytes for st in stages) / 2**30,
        }

    def lw_offload(self) -> dict:
        """Weights -> pinned host memory; weight and KV cache GPU storage freed."""
        self._lw_join()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        freed = self._lw_free_kv_cache()
        for st in self._lw_stages:
            st.cpu_ready.clear()
            for name, p in st.params:
                if name not in self._lw_cpu:
                    self._lw_cpu[name] = torch.empty_strided(
                        p.shape, p.stride(), dtype=p.dtype, pin_memory=True
                    )
                self._lw_cpu[name].copy_(p.data, non_blocking=True)
                freed += p.numel() * p.element_size()
        torch.cuda.synchronize()
        for st in self._lw_stages:
            for _, p in st.params:
                p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        torch.cuda.empty_cache()
        self._lw_resident = False
        return {"freed_gb": freed / 2**30, "seconds": time.perf_counter() - t0}

    def lw_load(self) -> dict:
        """Start streaming weights back in the background; returns at once."""
        if self._lw_resident:
            return {"started": False, "reason": "already resident"}
        if self._lw_thread is not None and self._lw_thread.is_alive():
            return {"started": False, "reason": "load in progress"}
        self._lw_error = None
        self._lw_alloc_kv_cache()
        self._lw_thread = threading.Thread(target=self._lw_loader, daemon=True)
        self._lw_thread.start()
        return {"started": True}

    def _lw_loader(self) -> None:
        try:
            with torch.cuda.stream(self._lw_copy_stream):
                for st in self._lw_stages:
                    for name, p in st.params:
                        src = self._lw_cpu[name]
                        dst = torch.empty_strided(
                            src.shape, src.stride(), dtype=src.dtype, device=p.device
                        )
                        dst.copy_(src, non_blocking=True)
                        p.data = dst
                    st.gpu_ready.record(self._lw_copy_stream)
                    # Signal only once the stage has landed: letting the CPU
                    # run ahead of the copies costs ~160 ms at the tail.
                    st.gpu_ready.synchronize()
                    st.cpu_ready.set()
            self._lw_copy_stream.synchronize()
            self._lw_resident = True
        except BaseException as e:
            self._lw_error = e
            for st in self._lw_stages:
                st.cpu_ready.set()

    def _lw_wait(self, idx: int) -> None:
        st = self._lw_stages[idx]
        st.cpu_ready.wait()
        if self._lw_error is not None:
            raise RuntimeError("layerwise load failed") from self._lw_error
        torch.cuda.current_stream().wait_event(st.gpu_ready)

    def lw_wait_loaded(self) -> dict:
        self._lw_join()
        if self._lw_error is not None:
            raise RuntimeError("layerwise load failed") from self._lw_error
        return {"resident": self._lw_resident}

    def lw_status(self) -> dict:
        return {
            "resident": self._lw_resident,
            "loading": self._lw_thread is not None and self._lw_thread.is_alive(),
            "stages_ready": sum(st.cpu_ready.is_set() for st in self._lw_stages),
            "num_stages": len(self._lw_stages),
        }

    def _lw_free_kv_cache(self) -> int:
        runner = self.model_runner  # type: ignore[attr-defined]
        freed = sum(t.numel() * t.element_size() for t in runner.kv_caches)
        runner.kv_caches.clear()
        runner.cross_layers_kv_cache = None
        for layer in runner.compilation_config.static_forward_context.values():
            if hasattr(layer, "kv_cache"):
                layer.kv_cache = torch.tensor([])
        return freed

    def _lw_alloc_kv_cache(self) -> None:
        runner = self.model_runner  # type: ignore[attr-defined]
        if runner.kv_caches:
            return
        if hasattr(runner, "kernel_block_sizes"):  # vllm.v1.worker.gpu.model_runner
            from vllm.v1.worker.gpu.attn_utils import init_kv_cache

            init_kv_cache(
                runner.kv_caches,
                runner.compilation_config.static_forward_context,
                runner.kv_cache_config,
                runner.attn_groups,
                runner.device,
                runner.cache_config.cache_dtype,
                runner.kernel_block_sizes,
                runner.vllm_config,
            )
        else:  # legacy gpu_model_runner
            runner.initialize_kv_cache_tensors(runner.kv_cache_config, runner._kernel_block_sizes)
        runner.post_kv_cache_wake_up()

    def _lw_join(self) -> None:
        if self._lw_thread is not None:
            self._lw_thread.join()
            self._lw_thread = None


def _find_decoder_layers(model: nn.Module) -> list[nn.Module]:
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        cands = [m for m in model.modules() if isinstance(m, nn.ModuleList)]
        if not cands:
            raise RuntimeError("cannot locate decoder layers")
        layers = max(cands, key=len)
    start = getattr(inner, "start_layer", 0)
    end = getattr(inner, "end_layer", len(layers))
    return list(layers)[start:end]


def _gate(obj, attr: str, wait) -> None:
    orig = getattr(obj, attr)

    def wrapped(*args, **kwargs):
        wait()
        return orig(*args, **kwargs)

    setattr(obj, attr, wrapped)
