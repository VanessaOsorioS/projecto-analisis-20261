# AGENTS.md

## Repository structure

Two independent Python projects for IIT/MIP analysis, each with its own `uv sync` environment:

```
GeoMIP/src/Method2_Dynamic_Programming_Reformulation/   # geometric (DP) strategy
QNodes/                                                    # classical PyPhi-based strategy
```

`GeoMIP/src/Method1_GPU_Accelerated/` **does not exist on disk** — references in README and `Dataset_Description.md` are stale.

## Commands

```bash
# QNodes (classical)
cd QNodes && uv sync && uv run exec.py
# Edit sample network page in QNodes/exec.py: aplicacion.set_pagina_red_muestra("A")
# Edit problem params in QNodes/src/main.py (estado_inicial, condiciones, alcance, mecanismo)

# GeoMIP Method2 (geometric)
cd GeoMIP/src/Method2_Dynamic_Programming_Reformulation && uv sync && uv run exec.py
# Input: GeoMIP/results/Pruebas_Metodo2.xlsx (sheet index 8, column B)
# Output: GeoMIP/results/resultados_Geometric.xlsx
```

Both projects use `uv` (not pip). Install with `pip install uv` if missing. Python 3.11+ required.

## KQNodes — k-partición MIP

New strategy at `QNodes/src/strategies/k_qnodes.py`. Extends QNodes (bipartition) to k-partitions (k ∈ {2,3,4,5}) with a **hybrid exhaustive/greedy** search:

```python
from src.strategies.k_qnodes import KQNodes
kn = KQNodes(tpm, k=3)                          # auto mode
kn = KQNodes(tpm, k=3, busqueda_exhaustiva=True) # force exhaustive
kn = KQNodes(tpm, k=3, busqueda_exhaustiva=False)# force greedy
sol = kn.aplicar_estrategia(estado_inicial, condicion, alcance, mecanismo)
```

- **k=2**: delegates to QNodes (Queyranne exacto, resultados idénticos).
- **k>2 — Modo exhaustivo** (S(2n,k) ≤ 1e5, o forzado con `busqueda_exhaustiva=True`):  
  Enumera todas las k-particiones mediante restricted growth strings y selecciona la de mínima pérdida EMD. Garantiza la k-MIP óptima global. Adecuado para n ≤ 5 (todos los k) o n ≤ 6 (k=3).
- **k>2 — Modo greedy** (S(2n,k) > 1e5, o forzado con `busqueda_exhaustiva=False`):  
  Partición jerárquica: bipartición óptima Queyranne inicial, luego subdivide iterativamente el grupo cuya división interna más reduce la pérdida. Escalable a sistemas mayores sin garantía de optimalidad global.
- **Evaluación**: `evaluate_k_partition()` marginaliza cada cubo a las dimensiones de mecanismo de su grupo. La distribución particionada es el vector de off-probabilities por nodo; pérdida = EMD efecto vs. distribución del subsistema. Idéntica en ambos modos.
- **Tensor product**: `tensor_product_k()` computa el producto exterior k-dimensional aplanado.
- **Formateo**: `fmt_kparte_q()` renderiza k grupos como `|purview||purview|...` / `|mechanism||mechanism|...`.

### Known limitation

Queyranne sobre subconjuntos de vértices (para splits internos en modo greedy) evalúa candidatos con `bipartir()`, que trata los vértices fuera del grupo como el "complemento" — objetivo sesgado. El resultado se re-evalúa correctamente con `evaluate_k_partition()`, por lo que el split es válido pero puede ser sub-óptimo. Una futura pasada de refinamiento local (intercambio de vértices entre grupos) mejoraría esto. El modo exhaustivo no tiene esta limitación.

## Quirks & gotchas

- **PyPhi cache**: Both projects write to `__pyphi_cache__/` directories under their root. Cached results are reused across runs and can mask code changes. Delete cache dir if results seem stale.
- **Fast completion**: If `phi=0`, the algorithm terminates early — this is normal, not a bug.
- **Profiling**: Enabled via `aplicacion.activar_profiling()` (QNodes) or `aplicacion.profiler_habilitado = True` (GeoMIP). Output HTML in `review/profiling/`.
- **No CI, no linter, no formatter, no typechecker, no test suite** configured for either project. The only test file is `QNodes/tests/__init__.py` (empty) plus an Excel file.
- TPM CSV samples live in `QNodes/src/.samples/` and `GeoMIP/data/samples/`.
- `Dataset_Description.md` at repo root documents the dataset but uses incorrect paths (e.g., `src/Method_1/`, `src/Method_2/`).

## Conventions

- Spanish identifiers throughout (aplicacion, gestor_redes, estado_inicial, etc.).
- No `.gitignore` for `__pyphi_cache__/` or `.logs/` — these are tracked in some places.
- Single-file entrypoints: `exec.py` at each project root delegates to `src/main.py`.
