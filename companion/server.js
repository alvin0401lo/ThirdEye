require('dotenv').config();
const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');
const WebSocket = require('ws');
const selfsigned = require('selfsigned');

const root = __dirname;
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css' };
const telegramToken = process.env.TELEGRAM_BOT_TOKEN;
const telegramChatId = process.env.TELEGRAM_CHAT_ID;
const publicUrl = process.env.PUBLIC_URL || `https://localhost:${process.env.PORT || 4173}`;
let lastTelegramAlert = 0;

async function sendTelegramAlert(alertText, room) {
  if (!telegramToken || !telegramChatId) {
    console.warn('Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.');
    return false;
  }
  if (Date.now() - lastTelegramAlert < 10000) return false;
  lastTelegramAlert = Date.now();
  const cameraUrl = new URL(publicUrl);
  cameraUrl.searchParams.set('room', room || 'THIRDEYE');
  const response = await fetch(`https://api.telegram.org/bot${telegramToken}/sendMessage`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      chat_id: telegramChatId,
      disable_web_page_preview: false,
      text: `${alertText || 'ThirdEye alert'}\nRoom: ${room || 'unknown'}\nOpen the link, then click Join room to view the live camera:\n${cameraUrl}`
    })
  });
  return response.ok;
}
const networkIpAddresses = [...new Set(Object.values(os.networkInterfaces()).flat()
  .filter(address => address && address.family === 'IPv4' && !address.internal)
  .map(address => address.address))];
const certificateAltNames = [
  { type: 2, value: 'thirdeye.local' },
  ...networkIpAddresses.map(ip => ({ type: 7, ip }))
];
const certificate = selfsigned.generate([{ name: 'commonName', value: 'thirdeye.local' }], {
  days: 365,
  keySize: 2048,
  algorithm: 'sha256',
  extensions: [{ name: 'subjectAltName', altNames: certificateAltNames }]
});

const publicFiles = new Set(['/companion.html', '/companion.js', '/index.html', '/script.js', '/styles.css']);

function servePage(request, response) {
  const requestPath = request.url === '/' ? '/companion.html' : request.url.split('?')[0];
  if (!publicFiles.has(requestPath)) return response.writeHead(404).end('Not found');
  const filePath = path.join(root, requestPath.slice(1));
  fs.readFile(filePath, (error, file) => {
    if (error) return response.writeHead(404).end('Not found');
    response.writeHead(200, { 'Content-Type': types[path.extname(filePath)] || 'text/plain' });
    response.end(file);
  });
}

const server = https.createServer({ key: certificate.private, cert: certificate.cert }, servePage);
const localServer = http.createServer(servePage);

const rooms = new Map();
const fallMonitors = new Map();
const telegramFallEventIds = new Map();
const wss = new WebSocket.Server({ noServer: true });

function isJpeg(data) {
  return data.length >= 4 && data.length <= 2000000 && data[0] === 0xff &&
    data[1] === 0xd8 && data[data.length - 2] === 0xff && data[data.length - 1] === 0xd9;
}

function updateCameraDemand(room) {
  const active = [...room.values()].some(peer => peer.role === 'viewer' && peer.readyState === WebSocket.OPEN);
  room.forEach(peer => {
    if (peer.role === 'camera-relay' && peer.readyState === WebSocket.OPEN) {
      peer.send(JSON.stringify({ type: 'camera_demand', active }));
    }
  });
}

function isFallMonitor(message) {
  if (typeof message.connected !== 'boolean' || typeof message.stale !== 'boolean' ||
      ![true, false, null].includes(message.sensor_ok) ||
      !Number.isSafeInteger(message.fall_count) || message.fall_count < 0) return false;
  const event = message.latest_fall;
  if (event === null) return true;
  return event && typeof event.event_id === 'string' && /^[0-9a-fA-F]{8}-[0-9]{1,10}$/.test(event.event_id) &&
    Number.isFinite(event.received_at) && event.received_at > 0 &&
    Number.isFinite(event.peak_g) && event.peak_g >= 0 && event.peak_g <= 16 &&
    Number.isFinite(event.peak_gyro) && event.peak_gyro >= 0 && event.peak_gyro <= 1800 &&
    Number.isFinite(event.tilt_deg) && event.tilt_deg >= 0 && event.tilt_deg <= 180;
}

function broadcastFallMonitor(room, monitor) {
  room.forEach(peer => {
    if (peer.role === 'viewer' && peer.readyState === WebSocket.OPEN) {
      peer.send(JSON.stringify(monitor));
    }
  });
}

function handleSocket(socket, request) {
  socket.id = Math.random().toString(36).slice(2, 10);
  socket.on('message', (raw, isBinary) => {
    if (isBinary) {
      const room = rooms.get(socket.room);
      const header = raw.subarray(0, 4).toString('ascii');
      const audioUp = socket.role === 'viewer' && header === 'VOX0';
      const cameraDown = ['glasses', 'camera-relay'].includes(socket.role) && isJpeg(raw);
      const audioDown = socket.role === 'glasses' && header === 'AUD0';
      if (!room || (!audioUp && !cameraDown && !audioDown)) return;
      const targetRole = audioUp ? 'glasses' : 'viewer';
      room?.forEach(peer => {
        if (peer !== socket && peer.role === targetRole && peer.readyState === WebSocket.OPEN) {
          if (peer.bufferedAmount < 1000000) peer.send(raw, { binary: true });
        }
      });
      return;
    }
    let message;
    try { message = JSON.parse(raw); } catch { return; }
    if (['join', 'join_glasses', 'join_camera_relay'].includes(message.type)) {
      if (socket.room || typeof message.room !== 'string' || !/^[A-Z0-9_-]{1,24}$/.test(message.room)) return;
      if (message.type === 'join_camera_relay' &&
          !['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(request.socket.remoteAddress)) return;
      socket.room = message.room;
      socket.role = message.type === 'join' ? 'viewer' : message.type === 'join_glasses' ? 'glasses' : 'camera-relay';
      const room = rooms.get(socket.room) || new Map();
      socket.send(JSON.stringify({ type: 'peers', peers: [...room.keys()].filter(peerId => room.get(peerId)?.role === 'viewer') }));
      room.set(socket.id, socket);
      rooms.set(socket.room, room);
      updateCameraDemand(room);
      if (socket.role === 'viewer' && fallMonitors.has(socket.room)) {
        socket.send(JSON.stringify(fallMonitors.get(socket.room)));
      }
      return;
    }
    if (message.type === 'fall_monitor') {
      const room = rooms.get(socket.room);
      if (socket.role !== 'camera-relay' || !room || !isFallMonitor(message)) return;
      const event = message.latest_fall;
      const monitor = {
        type: 'fall_monitor', connected: message.connected, stale: message.stale,
        sensor_ok: message.sensor_ok, fall_count: message.fall_count,
        latest_fall: event ? { event_id: event.event_id, received_at: event.received_at,
          peak_g: event.peak_g, peak_gyro: event.peak_gyro, tilt_deg: event.tilt_deg } : null
      };
      fallMonitors.set(socket.room, monitor);
      broadcastFallMonitor(room, monitor);
      if (event && telegramFallEventIds.get(socket.room) !== event.event_id) {
        telegramFallEventIds.set(socket.room, event.event_id);
        const alertText = `Suspected fall detected (${event.peak_g.toFixed(2)} g impact, ${event.tilt_deg.toFixed(1)} deg tilt).`;
        sendTelegramAlert(alertText, socket.room).catch(error => console.error('Telegram fall alert failed:', error.message));
      }
      return;
    }
    if (message.type === 'alert' && socket.role === 'glasses') {
      sendTelegramAlert(message.alert, socket.room).catch(error => console.error('Telegram alert failed:', error.message));
      return;
    }
    const target = rooms.get(socket.room)?.get(message.target);
    if (target?.readyState === WebSocket.OPEN) target.send(JSON.stringify({ ...message, sender: socket.id }));
  });
  socket.on('close', () => {
    const room = rooms.get(socket.room);
    if (!room) return;
    room.delete(socket.id);
    if (socket.role === 'camera-relay' &&
        ![...room.values()].some(peer => peer.role === 'camera-relay' && peer.readyState === WebSocket.OPEN)) {
      const monitor = fallMonitors.get(socket.room);
      if (monitor) {
        const offline = { ...monitor, connected: false, stale: true };
        fallMonitors.set(socket.room, offline);
        broadcastFallMonitor(room, offline);
      }
    }
    updateCameraDemand(room);
    if (socket.role === 'viewer') {
      room.forEach(peer => peer.send(JSON.stringify({ type: 'peer-left', peerId: socket.id })));
    } else if (![...room.values()].some(peer => ['glasses', 'camera-relay'].includes(peer.role))) {
      room.forEach(peer => peer.send(JSON.stringify({ type: 'camera-offline' })));
    }
    if (!room.size) rooms.delete(socket.room);
  });
}

wss.on('connection', handleSocket);

function upgradeToWebSocket(request, socket, head) {
  wss.handleUpgrade(request, socket, head, client => wss.emit('connection', client, request));
}

server.on('upgrade', upgradeToWebSocket);
localServer.on('upgrade', upgradeToWebSocket);

const bridgePort = process.env.BRIDGE_PORT || 4174;
const bridgeServer = http.createServer((request, response) => response.writeHead(404).end());
bridgeServer.on('upgrade', upgradeToWebSocket);

const port = process.env.PORT || 4173;
server.listen(port, '0.0.0.0', () => console.log(`ThirdEye companion running at https://localhost:${port}`));
const localPort = process.env.LOCAL_PORT || 4175;
localServer.listen(localPort, '127.0.0.1', () => console.log(`Local preview running at http://127.0.0.1:${localPort}`));
bridgeServer.listen(bridgePort, '0.0.0.0', () => console.log(`ESP32 bridge running at ws://localhost:${bridgePort}`));
