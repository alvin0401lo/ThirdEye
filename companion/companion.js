const $ = id => document.getElementById(id);
let socket;
let stream;
let muted = false;
const peers = new Map();
let audioContext;
let nextAudioTime = 0;
let microphoneSource;
let microphoneProcessor;
let microphoneMute;
let microphoneFilter;
let microphoneCompressor;
let receivedMicPackets = 0;
let latestFallId;

function renderFallMonitor(message) {
  const status = $('fallStatus');
  status.className = 'fall-state';
  if (!message.connected) {
    status.textContent = 'Device disconnected';
    status.classList.add('unavailable');
  } else if (message.sensor_ok === false) {
    status.textContent = 'Sensor unavailable';
    status.classList.add('unavailable');
  } else if (message.sensor_ok === null) {
    status.textContent = 'Waiting for sensor';
  } else if (message.stale) {
    status.textContent = 'Sensor status stale';
    status.classList.add('unavailable');
  } else {
    status.textContent = 'Monitoring';
    status.classList.add('ready');
  }
  const event = message.latest_fall;
  if (!event || event.event_id === latestFallId) return;
  latestFallId = event.event_id;
  $('fallSection').classList.add('alert');
  $('fallMessage').textContent = 'Suspected fall detected. Check on the wearer.';
  $('fallTime').textContent = new Date(event.received_at * 1000).toLocaleString();
  $('fallImpact').textContent = `${event.peak_g.toFixed(2)} g`;
  $('fallTilt').textContent = `${event.tilt_deg.toFixed(1)} degrees`;
  $('dismissFall').disabled = false;
}

function toast(message) {
  $('toast').textContent = message;
  $('toast').classList.add('show');
  setTimeout(() => $('toast').classList.remove('show'), 3000);
}

function send(message) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

async function makePeer(id, initiator) {
  const peer = new RTCPeerConnection({ iceServers: [] });
  peer.initiator = initiator;
  peers.set(id, peer);
  addLocalTracks(peer);
  peer.onicecandidate = event => event.candidate && send({ type: 'ice', target: id, candidate: event.candidate });
  peer.ontrack = event => {
    let audio = document.querySelector(`[data-peer="${id}"]`);
    if (!audio) {
      audio = document.createElement('audio');
      audio.dataset.peer = id;
      audio.autoplay = true;
      $('stage').append(audio);
    }
    audio.srcObject = event.streams[0] || new MediaStream([event.track]);
    audio.play().catch(() => toast('Tap Start voice call to hear the other person.'));
  };
  if (initiator) {
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    send({ type: 'offer', target: id, offer });
  }
  return peer;
}

function addLocalTracks(peer) {
  if (!stream) return;
  const existingKinds = new Set(peer.getSenders().map(sender => sender.track?.kind));
  stream.getTracks().forEach(track => {
    if (!existingKinds.has(track.kind)) peer.addTrack(track, stream);
  });
}

async function renegotiate(peer, id) {
  if (!peer || peer.signalingState !== 'stable') return;
  addLocalTracks(peer);
  const offer = await peer.createOffer();
  await peer.setLocalDescription(offer);
  send({ type: 'offer', target: id, offer });
}

async function joinRoom() {
  const room = $('roomInput').value.trim().toUpperCase();
  if (!room) return toast('Enter a room code first.');
  $('join').disabled = true;
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${protocol}://${location.host}`);
  socket.onopen = () => {
    send({ type: 'join', room });
    audioContext ||= new AudioContext();
    audioContext.resume();
    $('connection').innerHTML = '<i></i>Room connected';
    $('connection').classList.add('connected');
    $('join').textContent = 'Joined';
    $('camera').disabled = false;
    $('callButton').disabled = false;
    $('endCall').disabled = false;
    toast('Joined room.');
  };
  socket.onmessage = async event => {
    if (typeof event.data !== 'string') {
      const packet = new Uint8Array(await event.data.arrayBuffer());
      if (String.fromCharCode(...packet.slice(0, 4)) === 'AUD0') {
        receivedMicPackets += 1;
        if (receivedMicPackets === 1) console.info('INMP441 audio is reaching the website');
        playAudioPacket(packet);
        return;
      }
      if (packet[0] !== 0xff || packet[1] !== 0xd8) return;
      const frame = document.querySelector('#glassesFrame') || document.createElement('img');
      frame.id = 'glassesFrame';
      frame.alt = 'Live frame from ESP32 glasses';
      frame.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:contain';
      if (!frame.parentElement) $('stage').append(frame);
      const nextUrl = URL.createObjectURL(event.data);
      const oldUrl = frame.dataset.url;
      frame.src = nextUrl;
      frame.dataset.url = nextUrl;
      if (oldUrl) setTimeout(() => URL.revokeObjectURL(oldUrl), 1000);
      $('stage').classList.add('active');
      $('live').innerHTML = '<i></i>Glasses camera is live';
      $('live').classList.add('on');
      return;
    }
    const message = JSON.parse(event.data);
    if (message.type === 'fall_monitor') renderFallMonitor(message);
    if (message.type === 'peers') for (const id of message.peers) await makePeer(id, true);
    if (message.type === 'offer') {
      const peer = peers.get(message.sender) || await makePeer(message.sender, false);
      await peer.setRemoteDescription(message.offer);
      const answer = await peer.createAnswer();
      await peer.setLocalDescription(answer);
      send({ type: 'answer', target: message.sender, answer });
    }
    if (message.type === 'answer') await peers.get(message.sender)?.setRemoteDescription(message.answer);
    if (message.type === 'ice') await peers.get(message.sender)?.addIceCandidate(message.candidate);
    if (message.type === 'media-ready') {
      const peer = peers.get(message.sender);
      if (peer?.initiator) await renegotiate(peer, message.sender);
    }
    if (message.type === 'peer-left') {
      document.querySelector(`[data-peer="${message.peerId}"]`)?.remove();
      peers.get(message.peerId)?.close();
      peers.delete(message.peerId);
    }
    if (message.type === 'camera-offline') {
      const frame = $('glassesFrame');
      if (frame?.dataset.url) URL.revokeObjectURL(frame.dataset.url);
      frame?.remove();
      $('stage').classList.remove('active');
      $('live').innerHTML = '<i></i>Waiting for glasses';
      $('live').classList.remove('on');
    }
  };
  socket.onerror = () => toast('Could not reach the local server.');
  socket.onclose = () => {
    $('fallStatus').textContent = 'Website disconnected';
    $('fallStatus').className = 'fall-state unavailable';
  };
}

function playAudioPacket(packet) {
  if (packet.length < 10) return;
  const view = new DataView(packet.buffer, packet.byteOffset, packet.byteLength);
  const sampleRate = view.getUint32(4, true);
  const sampleCount = view.getUint16(8, true);
  if (!sampleCount) return;
  audioContext ||= new AudioContext();
  const audioBuffer = audioContext.createBuffer(1, sampleCount, sampleRate);
  const channel = audioBuffer.getChannelData(0);
  for (let index = 0; index < sampleCount; index += 1) {
    channel[index] = view.getInt16(10 + index * 2, true) / 32768;
  }
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);
  // Keep the voice live: never allow a delayed packet queue to grow into
  // noticeable latency after a slow Wi-Fi moment.
  if (nextAudioTime - audioContext.currentTime > 0.2) {
    nextAudioTime = audioContext.currentTime;
  }
  const startTime = Math.max(audioContext.currentTime, nextAudioTime);
  source.start(startTime);
  nextAudioTime = startTime + audioBuffer.duration;
}

function endCall() {
  $('fallStatus').textContent = 'Not connected';
  $('fallStatus').className = 'fall-state';
  stream?.getTracks().forEach(track => track.stop());
  stream = undefined;
  microphoneProcessor?.disconnect();
  microphoneSource?.disconnect();
  microphoneMute?.disconnect();
  microphoneFilter?.disconnect();
  microphoneCompressor?.disconnect();
  microphoneProcessor = undefined;
  microphoneSource = undefined;
  microphoneMute = undefined;
  microphoneFilter = undefined;
  microphoneCompressor = undefined;
  peers.forEach(peer => peer.close());
  peers.clear();
  document.querySelectorAll('[data-peer]').forEach(audio => audio.remove());
  const glassesFrame = $('glassesFrame');
  if (glassesFrame?.dataset.url) URL.revokeObjectURL(glassesFrame.dataset.url);
  glassesFrame?.remove();
  socket?.close();
  socket = undefined;
  audioContext?.close();
  audioContext = undefined;
  nextAudioTime = 0;
  $('local').srcObject = null;
  $('stage').classList.remove('active');
  $('live').innerHTML = '<i></i>Camera is off';
  $('live').classList.remove('on');
  $('camera').disabled = true;
  $('camera').textContent = 'Enable microphone';
  $('callButton').disabled = true;
  $('callButton').textContent = 'Start voice call ☎';
  $('endCall').disabled = true;
  $('mute').disabled = true;
  $('join').disabled = false;
  $('join').textContent = 'Join room ↗';
  $('connection').innerHTML = '<i></i>Not connected';
  $('connection').classList.remove('connected');
  $('callState').textContent = 'Call ended. Join the room to connect again.';
  toast('Camera and microphone closed.');
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
    $('camera').textContent = 'Microphone enabled';
    $('mute').disabled = false;
    $('endCall').disabled = false;
    $('callState').textContent = 'Your microphone is ready for a call.';
    startSpeakerUplink(stream);
    for (const [id, peer] of peers) {
      addLocalTracks(peer);
      if (peer.initiator) await renegotiate(peer, id);
      else send({ type: 'media-ready', target: id });
    }
    toast('Microphone is ready.');
  } catch (error) {
    toast('Microphone permission is needed for the voice call.');
  }
}

function startSpeakerUplink(mediaStream) {
  audioContext ||= new AudioContext();
  audioContext.resume();
  microphoneSource = audioContext.createMediaStreamSource(mediaStream);
  microphoneFilter = audioContext.createBiquadFilter();
  microphoneFilter.type = 'highpass';
  microphoneFilter.frequency.value = 100;
  microphoneCompressor = audioContext.createDynamicsCompressor();
  microphoneCompressor.threshold.value = -34;
  microphoneCompressor.knee.value = 18;
  microphoneCompressor.ratio.value = 4;
  microphoneCompressor.attack.value = 0.003;
  microphoneCompressor.release.value = 0.15;
  // 2048 samples keeps each 16 kHz VOX0 packet below the ESP32 speaker limit.
  microphoneProcessor = audioContext.createScriptProcessor(2048, 1, 1);
  microphoneMute = audioContext.createGain();
  microphoneMute.gain.value = 0;
  microphoneProcessor.onaudioprocess = event => {
    if (!socket || socket.readyState !== WebSocket.OPEN || muted) return;
    const input = event.inputBuffer.getChannelData(0);
    let energy = 0;
    for (let index = 0; index < input.length; index += 1) energy += input[index] * input[index];
    const rms = Math.sqrt(energy / input.length);
    const sampleCount = Math.floor(input.length * 16000 / audioContext.sampleRate);
    const packet = new ArrayBuffer(10 + sampleCount * 2);
    const view = new DataView(packet);
    new Uint8Array(packet, 0, 4).set([86, 79, 88, 48]);
    view.setUint32(4, 16000, true);
    view.setUint16(8, sampleCount, true);
    for (let index = 0; index < sampleCount; index += 1) {
      const sourceIndex = Math.min(input.length - 1, Math.floor(index * input.length / sampleCount));
      const sample = rms < 0.006 ? 0 : Math.max(-1, Math.min(1, input[sourceIndex]));
      view.setInt16(10 + index * 2, sample * 32767, true);
    }
    socket.send(packet);
    if (!startSpeakerUplink.sentPacket) {
      startSpeakerUplink.sentPacket = true;
      console.info('Website microphone uplink is sending VOX0 audio');
    }
  };
  microphoneSource.connect(microphoneFilter);
  microphoneFilter.connect(microphoneCompressor);
  microphoneCompressor.connect(microphoneProcessor);
  microphoneProcessor.connect(microphoneMute);
  microphoneMute.connect(audioContext.destination);
}

const linkedRoom = new URLSearchParams(location.search).get('room');
if (linkedRoom && /^[A-Za-z0-9_-]{1,24}$/.test(linkedRoom)) $('roomInput').value = linkedRoom.toUpperCase();

$('join').onclick = joinRoom;
$('dismissFall').onclick = () => {
  $('fallSection').classList.remove('alert');
  $('fallMessage').textContent = 'Alert dismissed. Last detection retained below.';
  $('dismissFall').disabled = true;
};
$('camera').onclick = startCamera;
$('callButton').onclick = async () => { if (!stream) await startCamera(); if (stream) { $('callState').textContent = 'Voice call is active in this room.'; $('callButton').textContent = 'Call active'; toast('Voice call started.'); } };
$('mute').onclick = event => {
  const track = stream?.getAudioTracks()[0];
  if (!track) return toast('Start the camera to enable the microphone.');
  muted = !muted;
  track.enabled = !muted;
  event.currentTarget.textContent = muted ? 'Unmute mic' : 'Mute mic';
};
$('endCall').onclick = endCall;
