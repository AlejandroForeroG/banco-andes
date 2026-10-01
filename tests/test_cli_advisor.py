'Capability and regression tests.'

from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from cli import intent
from cli.main import _chat

pytestmark = pytest.mark.review


@pytest.mark.parametrize('content, expected', [
    ('{"requests_advisor": true}', True),
    ('{"requests_advisor": false}', False),
    ('{"requests_advisor": "true"}', False),
    ('{"requests_advisor": 1}', False),
    ('null', False), ('[]', False), ('I do not know', False),
])
async def test_classifier_requires_boolean(content, expected, monkeypatch):
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=content)))
    monkeypatch.setattr(intent, 'ChatOllama', lambda **_: model)
    assert await intent.requests_advisor('I want to speak to a person') is expected


async def test_classifier_failure_does_not_authorize(monkeypatch):
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=TimeoutError()))
    monkeypatch.setattr(intent, 'ChatOllama', lambda **_: model)
    with pytest.raises(RuntimeError, match='I could not interpret'):
        await intent.requests_advisor('I want an advisor')


@pytest.mark.parametrize('utterance', ['I want to speak to an advisor', '/advisor'])
@pytest.mark.parametrize('consent, authenticated, should_send', [
    ('yes', True, True), ('y', True, True), ('no', True, False),
    ('maybe', True, False), ('yes', False, False),
])
async def test_cli_keeps_review_and_requires_consent_and_session(
    utterance, consent, authenticated, should_send, monkeypatch,
):
    review = {'task_id': 'reviewed-case', 'status': 'validated', 'exchanges': [{
        'request': {'draft': 'Validated case guidance.', 'evidence': []},
        'response': {'findings': []},
    }]}
    entries = iter(['Query housing products', utterance, consent, '/exit'])
    prompts, displayed, calls, turns = [], [], [], []

    class Prompt:
        async def prompt_async(self, prompt):
            prompts.append(prompt)
            if 'Send this summary' in prompt:
                assert not calls
                assert any('reviewed-case' in output for output in displayed)
            return next(entries)

    class Graph:
        async def astream(self, inputs, **kwargs):
            turns.append(inputs['messages'][0].content)
            yield {'revisar': {'review': review, 'messages': [AIMessage(content='Reviewed guidance.')]}}

    @asynccontextmanager
    async def checkpointer():
        yield None

    async def invoke(args):
        calls.append(args)
        return {'id': 'ESC-TEST', 'cliente_id': 'CLI-TEST', 'creado_en': 'now'}

    def session():
        if not authenticated:
            raise RuntimeError('Sign in with /login')
        return SimpleNamespace(client_id='CLI-TEST')

    classifier = AsyncMock(side_effect=lambda text: text == utterance)
    tool = SimpleNamespace(name='register_handoff', ainvoke=invoke)
    monkeypatch.setattr('prompt_toolkit.PromptSession', Prompt)
    monkeypatch.setattr('prompt_toolkit.patch_stdout.patch_stdout', nullcontext)
    monkeypatch.setattr('agent.graph.build_graph', lambda **_: Graph())
    monkeypatch.setattr('agent.memory.checkpointer.open_async_checkpointer', checkpointer)
    monkeypatch.setattr('cli.main._reunir_tools', AsyncMock(return_value=[tool]))
    monkeypatch.setattr('cli.main.get_settings', lambda: SimpleNamespace(
        business_domain='finanzas', langsmith_tracing=False))
    monkeypatch.setattr('cli.main.ui.banner', lambda _: None)
    monkeypatch.setattr('cli.main.ui.info', displayed.append)
    monkeypatch.setattr('cli.main.ui.aviso', displayed.append)
    monkeypatch.setattr('cli.intent.requests_advisor', classifier)
    monkeypatch.setattr('agent.human.required_session', session)

    await _chat(True, 'test')

    assert turns == ['Query housing products']
    assert sum('Send this summary' in prompt for prompt in prompts) == 1
    assert len(calls) == int(should_send)
    if should_send:
        assert calls[0]['informacion_recopilada']['task_id'] == 'reviewed-case'
        assert calls[0]['informacion_recopilada']['reviewed_guidance'] == 'Validated case guidance.'
    assert all(call.args[0] != '/advisor' for call in classifier.call_args_list)
