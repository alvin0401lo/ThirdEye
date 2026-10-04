import json
from queue import Empty, Queue

import numpy as np
import pytest

from thirdeye import realtime_audio
from thirdeye.audio import IncompleteSpeechError


class Socket:
    def __init__(self, fail=False):
        self.events = Queue()
        self.sent = []
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def close(self):
        self.closed = True

    def send(self, message):
        event = json.loads(message)
        self.sent.append(event)
        kind = event['type']
        if kind == 'session.update':
            self.events.put({'type': 'session.updated'})
        elif kind == 'input_audio_buffer.append':
            self.events.put({'type': 'conversation.item.input_audio_transcription.delta', 'delta': 'Find'})
        elif kind == 'input_audio_buffer.commit':
            self.events.put({'type': 'input_audio_buffer.committed', 'item_id': 'a'})
            if self.fail:
                self.events.put({'type': 'error', 'error': {'message': 'test failure'}})
            else:
                self.events.put({'type': 'conversation.item.input_audio_transcription.completed',
                                 'item_id': 'old', 'transcript': 'Wrong command'})
                self.events.put({'type': 'conversation.item.input_audio_transcription.completed',
                                 'item_id': 'a', 'transcript': 'Find my bottle.'})

    def recv(self, timeout):
        try:
            return json.dumps(self.events.get(timeout=timeout))
        except Empty:
            raise TimeoutError from None


@pytest.mark.parametrize('mode', ['success', 'silence', 'overflow', 'error'])
def test_realtime_commits_only_complete_audio(monkeypatch, capsys, mode):
    ws = Socket(fail=mode == 'error')
    monkeypatch.setattr(realtime_audio, 'connect', lambda *args, **kwargs: ws)

    def capture(device, **kwargs):
        assert 0 < kwargs['wait_timeout'] <= 30
        assert kwargs['require_complete']
        if mode == 'silence':
            return None
        samples = np.full(2400, 500, dtype=np.int16)
        kwargs['on_chunk'](samples, 24000)
        if mode == 'overflow':
            raise IncompleteSpeechError('overflow')
        return samples

    monkeypatch.setattr(realtime_audio, '_capture_utterance', capture)
    listener = realtime_audio.RealtimeTranscriber('test-key')
    if mode in {'overflow', 'error'}:
        with pytest.raises(RuntimeError):
            listener.listen_text(wait_timeout=30)
    else:
        assert listener.listen_text(wait_timeout=30) == (None if mode == 'silence' else 'Find my bottle.')
    committed = any(e['type'] == 'input_audio_buffer.commit' for e in ws.sent)
    assert committed == (mode in {'success', 'error'})
    assert ('[ASR FINAL]' in capsys.readouterr().out) == (mode == 'success')
    listener.close()


def test_followup_reuses_open_session(monkeypatch):
    connections = []

    def open_socket(*args, **kwargs):
        socket = Socket()
        connections.append(socket)
        return socket

    monkeypatch.setattr(realtime_audio, 'connect', open_socket)

    def capture(device, **kwargs):
        samples = np.full(2400, 500, dtype=np.int16)
        kwargs['on_chunk'](samples, 24000)
        return samples

    monkeypatch.setattr(realtime_audio, '_capture_utterance', capture)
    listener = realtime_audio.RealtimeTranscriber('test-key')
    assert listener.listen_text() == 'Find my bottle.'
    assert listener.listen_text(wait_timeout=30) == 'Find my bottle.'
    assert len(connections) == 1
    assert sum(event['type'] == 'session.update' for event in connections[0].sent) == 1
    listener.close()
    assert connections[0].closed


def test_requires_key():
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        realtime_audio.RealtimeTranscriber('')


def test_external_pcm_capture_uses_same_final_only_path(monkeypatch, capsys):
    ws = Socket()
    monkeypatch.setattr(realtime_audio, 'connect', lambda *args, **kwargs: ws)
    monkeypatch.setattr(realtime_audio, '_capture_utterance',
                        lambda *args, **kwargs: pytest.fail('PC microphone used'))

    def capture(device, **kwargs):
        assert device is None
        assert kwargs['require_complete']
        assert 0 < kwargs['wait_timeout'] <= 30
        chunk = np.full(320, 1200, dtype=np.int16)
        kwargs['on_chunk'](chunk, 16000)
        return chunk

    listener = realtime_audio.RealtimeTranscriber('test-key', capture_utterance=capture)
    assert listener.listen_text(wait_timeout=30) == 'Find my bottle.'
    appended = [event for event in ws.sent if event['type'] == 'input_audio_buffer.append']
    assert len(appended) == 1
    assert ws.sent[-1]['type'] == 'input_audio_buffer.commit'
    assert '[ASR PARTIAL]' in capsys.readouterr().out
    listener.close()
