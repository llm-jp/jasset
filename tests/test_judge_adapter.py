import json

import pytest

from jasset.evaluation.judge import Cache, fingerprint

REQUEST = dict(question='看板の文字は？', gold='入口', prediction='入口')


def test_user_judge_and_cache_identity(tmp_path):
    messages = []

    def judge(prompt):
        messages.append(prompt)
        return 'correct: yes'

    cache = Cache(tmp_path, tmp_path / 'cache', judge, 'model-a')
    assert cache.evaluate(REQUEST) == 1
    assert messages[0][0]['role'] == 'system'
    assert REQUEST['question'] in messages[0][1]['content']
    assert cache.evaluate(REQUEST) == 1
    assert len(messages) == 1
    other = Cache(tmp_path, tmp_path / 'cache', lambda _: 'correct: no', 'model-b')
    assert other.evaluate(REQUEST) == 0
    assert other.calls == 1


def test_archived_cache_can_be_read_without_model(tmp_path):
    archive = tmp_path / 'archive'
    archive.mkdir()
    (archive / (fingerprint(REQUEST) + '.json')).write_text(json.dumps(dict(
        request=REQUEST, attempts=[dict(content='correct: yes')], score=1,
    )))
    assert Cache(tmp_path / 'new', archive).evaluate(REQUEST) == 1
    fresh = Cache(tmp_path / 'new', archive, lambda _: 'correct: no', 'new-model')
    assert fresh.evaluate(REQUEST) == 0


def test_model_error_leaves_completed_results_available(tmp_path):
    def judge(messages):
        raise RuntimeError('model unavailable')

    cache = Cache(tmp_path, tmp_path / 'cache', lambda _: 'correct: yes', 'model')
    assert cache.evaluate(REQUEST) == 1
    cache.judge = judge
    with pytest.raises(RuntimeError, match='unavailable'):
        cache.evaluate(dict(REQUEST, prediction='出口'))
    assert cache.evaluate(REQUEST) == 1


def test_invalid_response_remains_unscored(tmp_path):
    cache = Cache(tmp_path, tmp_path / 'cache', lambda _: 'unknown', 'model')
    assert cache.evaluate(REQUEST) is None
    cache.judge = lambda _: {'content': 'correct: yes'}
    with pytest.raises(TypeError, match='str'):
        cache.evaluate(REQUEST)
