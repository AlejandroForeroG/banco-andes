from types import SimpleNamespace
from datetime import date

import pytest
from click.testing import CliRunner
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import build_graph, _summary_source
from agent.review import needs_review
from cli.main import cli
from reviewer.critic import assemble_verdict, enforce_grounding


def test_documented_tools_command_runs_without_mcp():
    result = CliRunner().invoke(cli, ['tools', '--no-mcp'])
    assert result.exit_code == 0, result.output
    assert 'query_credit_catalog' in result.output


@pytest.mark.parametrize('question', [
    'What is my interest rate?', 'Explain APR', 'What is my debt?',
    'What is my income?', 'What does my score mean?',
    'Explain my borrowing capacity', 'What is my installment?',
    'Explain repayment', 'I need financing',
])
def test_english_financial_requests_require_review(question):
    assert needs_review([HumanMessage(content=question)], 'Here is the information.')


@pytest.mark.parametrize('draft, rejected', [
    ('This is the best option, but approval is not guaranteed.', True),
    ('This is right for you.', True),
    ('This is a good fit for your needs.', True),
    ('I recommend this loan.', True),
    ('This is not the best option.', False),
    ('I cannot determine the best choice without verified conditions.', False),
])
def test_approval_disclaimers_do_not_excuse_suitability_claims(draft, rejected):
    request = SimpleNamespace(scope='housing', draft=draft, evidence=[], case_date=date(2026, 10, 1))
    verdict = assemble_verdict({f'C{i:02}': 'pass' for i in range(1, 5)}, request)
    reviewed = enforce_grounding(verdict, request)
    criterion = next(c for c in reviewed.criteria_evaluated if c.criterion == 'C04')
    assert (criterion.result == 'fail') is rejected


def test_handoff_tool_is_excluded_from_model_tools(monkeypatch):
    import agent.graph as graph_module
    bound_names = []

    class Model:
        def bind_tools(self, tools):
            bound_names.extend(tool.name for tool in tools)
            return self

    monkeypatch.setattr(graph_module, 'ChatOllama', lambda **_: Model())
    build_graph(tools=[SimpleNamespace(name='register_handoff')])
    assert 'register_handoff' not in bound_names


@pytest.mark.parametrize('denial', [
    "I don't remember earlier messages.",
    "I can't access previous conversations.",
    'I have no memory of your request.',
    'Please sign in to access the conversation.',
    'Please log in before I can remember.',
])
def test_memory_denials_do_not_replace_thread_facts(denial):
    fact = HumanMessage(content='My housing budget is 180 million COP.')
    assert _summary_source([fact, AIMessage(content=denial)]) == [fact]


def test_truthful_session_observations_remain_in_history():
    fact = AIMessage(content='The authenticated session confirmed the requested profile query.')
    assert _summary_source([HumanMessage(content='Query my profile'), fact])[-1] == fact
