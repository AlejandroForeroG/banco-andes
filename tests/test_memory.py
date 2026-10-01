'Capability and regression tests.'


import uuid
from operator import add
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

import contract
from tests.utiles import estado_de, invocar

pytestmark = pytest.mark.memory


class _Estado(TypedDict):
    pasos: Annotated[list, add]


def _grafo_de_test(checkpointer):
    'Support the associated capability or regression scenario.'
    grafo = StateGraph(_Estado)
    grafo.add_node("registrar", lambda estado: {"pasos": ["visita"]})
    grafo.add_edge(START, "registrar")
    grafo.add_edge("registrar", END)
    return grafo.compile(checkpointer=checkpointer)


def test_el_checkpointer_se_construye():
    checkpointer = contract.build_checkpointer()
    assert checkpointer is not None, "build_checkpointer() returned nothing"
    assert hasattr(checkpointer, "get_tuple"), (
        "The result is not a LangGraph checkpointer. "
        "SqliteSaver.from_conn_string(...) is a context manager: "
        "enter it and return the initialized checkpointer."
    )


def test_el_store_se_construye():
    store = contract.build_store()
    assert store is not None, "build_store() returned nothing"


def test_un_mismo_thread_acumula_estado():
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}
    grafo = _grafo_de_test(contract.build_checkpointer())

    invocar(grafo, {"pasos": []}, config)
    estado = invocar(grafo, {"pasos": []}, config)

    assert len(estado["pasos"]) >= 2, (
        "Two calls on the same thread_id must accumulate state. "
        f"Received: {estado['pasos']}"
    )


def test_threads_distintos_no_comparten_estado():
    grafo = _grafo_de_test(contract.build_checkpointer())
    config_a = {"configurable": {"thread_id": f"test-a-{uuid.uuid4()}"}}
    config_b = {"configurable": {"thread_id": f"test-b-{uuid.uuid4()}"}}

    invocar(grafo, {"pasos": []}, config_a)
    invocar(grafo, {"pasos": []}, config_a)
    estado_b = invocar(grafo, {"pasos": []}, config_b)

    assert len(estado_b["pasos"]) == 1, (
        "Different thread IDs must not share history: thread B should have "
        f"one step but has {len(estado_b['pasos'])}"
    )


def test_la_memoria_sobrevive_al_reinicio():
    'Verify this behavior with deterministic test fixtures.'


    config = {"configurable": {"thread_id": f"test-disco-{uuid.uuid4()}"}}

    invocar(_grafo_de_test(contract.build_checkpointer()), {"pasos": []}, config)

    grafo_nuevo = _grafo_de_test(contract.build_checkpointer())
    recuperado = estado_de(grafo_nuevo, config)

    assert recuperado.values.get("pasos"), (
        "A new checkpointer instance did not recover persisted state. "
        "Memory must persist on disk using SqliteSaver or equivalent; "
        "MemorySaver loses data on restart."
    )
