'Capability and regression tests.'


import pytest

import contract
from tests.utiles import identificadores_del_proyecto, nombres_de_nodos

pytestmark = pytest.mark.reasoning

NOMBRES_REACT = {
    "reason", "reasoning", "think", "thought", "act", "action",
    "razonar", "razonamiento", "pensar", "actuar", "action",
}
NOMBRES_PLAN = {
    "planner", "plan", "executor", "execute", "replanner", "replan",
    "planificador", "planificar", "ejecutor", "ejecutar", "replanificador",
}
REDES_DE_SEGURIDAD = {
    "max_iterations", "max_steps", "max_iteraciones", "max_pasos",
    "limite_iteraciones", "limite_pasos", "recursion_limit",
}


def test_el_grafo_de_razonamiento_compila():
    grafo = contract.build_reasoning_graph()
    assert grafo is not None, "build_reasoning_graph() did not return a compiled graph"


def test_tiene_nodos_del_patron_elegido():
    nodos = nombres_de_nodos(contract.build_reasoning_graph())
    reales = {n for n in nodos if n not in ("__start__", "__end__")}
    assert reales & (NOMBRES_REACT | NOMBRES_PLAN), (
        f"No ReAct or Plan-and-Execute nodes were recognized. Nodes: {sorted(reales)}.\n"
        f"Use names such as {sorted(NOMBRES_REACT)} (ReAct) "
        f"o {sorted(NOMBRES_PLAN)} (Plan-and-Execute)."
    )


def test_tiene_red_de_seguridad():
    'Verify this behavior with deterministic test fixtures.'
    identificadores = identificadores_del_proyecto()
    assert identificadores & REDES_DE_SEGURIDAD, (
        "No safety limit was found in the code.\n"
        "A cyclic graph must have an explicit iteration or step limit. "
        f"Use one of these names: {sorted(REDES_DE_SEGURIDAD)}."
    )
