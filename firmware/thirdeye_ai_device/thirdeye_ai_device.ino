#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WebSocketsClient.h>
#include <Wire.h>
#include <driver/i2s.h>
#include <freertos/queue.h>
#include <esp_heap_caps.h>
#include <stdlib.h>
#include <esp_random.h>
#include <cmath>
#include <cstdint>
#include "esp_camera.h"

// BEGIN PORTABLE FALL DETECTOR
// Prototype thresholds, not validated human-fall classification.
struct FallDetector {
  enum State { WARMUP, MONITORING, OBSERVING, COOLDOWN };
  State state = WARMUP;
  uint32_t since = 0, previous = 0, quietSince = 0;
  bool started = false, quiet = false;
  float bx = 0, by = 0, bz = 1;
  float peakG = 0, peakGyro = 0, tiltDegrees = 0;

  void reset(uint32_t now) {
    state = WARMUP;
    since = previous = now;
    started = true;
    quiet = false;
  }

  bool update(uint32_t now, float ax, float ay, float az,
              float gx, float gy, float gz) {
    if (!started || uint32_t(now - previous) > 100) reset(now);
    previous = now;
    const float g = std::sqrt(ax * ax + ay * ay + az * az);
    const float rotation = std::sqrt(gx * gx + gy * gy + gz * gz);
    if (!std::isfinite(g) || !std::isfinite(rotation)) {
      reset(now);
      return false;
    }
    const bool stationary = std::fabs(g - 1.0f) < 0.20f && rotation < 20.0f;
    if (state == COOLDOWN) {
      if (uint32_t(now - since) >= 10000) reset(now);
      return false;
    }
    if (state == WARMUP) {
      if (!stationary) { since = now; return false; }
      bx = ax / g; by = ay / g; bz = az / g;
      if (uint32_t(now - since) >= 2000) state = MONITORING;
      return false;
    }
    if (state == MONITORING) {
      if (g >= 2.5f || rotation >= 220.0f) {
        state = OBSERVING;
        since = now;
        quiet = false;
        peakG = g; peakGyro = rotation;
      } else if (stationary) {
        bx = 0.98f * bx + 0.02f * ax / g;
        by = 0.98f * by + 0.02f * ay / g;
        bz = 0.98f * bz + 0.02f * az / g;
      }
      return false;
    }
    if (g > peakG) peakG = g;
    if (rotation > peakGyro) peakGyro = rotation;
    const float baseline = std::sqrt(bx * bx + by * by + bz * bz);
    float dot = g > 0.01f ? (bx * ax + by * ay + bz * az) / (baseline * g) : 1.0f;
    dot = dot < -1 ? -1 : dot > 1 ? 1 : dot;
    tiltDegrees = std::acos(dot) * 57.29578f;
    if (stationary && tiltDegrees >= 50.0f) {
      if (!quiet) { quiet = true; quietSince = now; }
      if (uint32_t(now - since) >= 2000 && uint32_t(now - quietSince) >= 1000) {
        state = COOLDOWN;
        since = now;
        return true;
      }
    } else quiet = false;
    if (uint32_t(now - since) >= 5000) reset(now);
    return false;
  }
};
// END PORTABLE FALL DETECTOR

// Fill these values for your trusted local PC before uploading.
#define WIFI_SSID "YOUR_2_4_GHZ_WIFI"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#define SERVER_HOST "YOUR_PC_IPV4"
#define SERVER_PORT 8081
#define DEVICE_TOKEN "YOUR_PRIVATE_TOKEN"

// Full ESP32 device; AI inference runs on the PC server.
#define MIC_ONLY_DIAGNOSTIC 0
#define PC_AUDIO_CAMERA_TEST 0
#define ENABLE_IMU 1
#define SPEAKER_BOOT_TEST 1
#if MIC_ONLY_DIAGNOSTIC && PC_AUDIO_CAMERA_TEST
#error Choose only one diagnostic mode
#endif

// OV2640 camera pins for this ESP32-S3-CAM N16R8 board.
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5
#define Y9_GPIO_NUM 16
#define Y8_GPIO_NUM 17
#define Y7_GPIO_NUM 18
#define Y6_GPIO_NUM 12
#define Y5_GPIO_NUM 10
#define Y4_GPIO_NUM 8
#define Y3_GPIO_NUM 9
#define Y2_GPIO_NUM 11
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13

// INMP441 + MAX98357A share clocks; microphone and speaker use separate data pins.
#define I2S_BCLK 2
#define I2S_WS 1
#define I2S_MIC_SD 3
#define I2S_SPK_DOUT 14

// Verify GPIO21/47 are exposed and unused on the actual board.
#define MPU6500_SDA 21
#define MPU6500_SCL 47
#define MPU6500_ADDRESS 0x68
#define AUDIO_SAMPLE_RATE 16000
#define MIC_UPLOAD_SAMPLE_RATE 8000
#define CAMERA_FRAME_INTERVAL_MS 88

WebSocketsClient cameraSocket, audioSocket, imuSocket;
volatile bool videoEnabled = false;
volatile bool cameraConnected = false, audioConnected = false, imuConnected = false;
bool imuReady = false;
unsigned long lastStatusAt = 0;
volatile unsigned long sentFrames = 0;
framesize_t streamFrameSize = FRAMESIZE_VGA;
QueueHandle_t cameraFrames = nullptr, cameraStillRequests = nullptr, cameraStillResults = nullptr;
struct CameraPacket {
  uint8_t* data = nullptr;
  size_t length = 0;
  uint32_t capturedAt = 0;
  unsigned long stillId = 0;
};
struct CameraMetrics {
  uint32_t grabs = 0, captures = 0, sends = 0, replacements = 0, expired = 0, allocationFailures = 0;
  uint64_t grabUs = 0, copyUs = 0, sendUs = 0, ageMs = 0, jpegBytes = 0;
  uint32_t grabMaxUs = 0, sendMaxUs = 0;
};
CameraMetrics cameraMetrics;
portMUX_TYPE cameraMetricsMux = portMUX_INITIALIZER_UNLOCKED;
bool publishCameraPacket(QueueHandle_t queue, CameraPacket packet);
CameraPacket copyCameraJpeg(camera_fb_t* frame, unsigned long stillId);

// BEGIN CAMERA MAILBOX
// Queue ownership transfers only on success; a replaced frame is freed once.
bool publishCameraPacket(QueueHandle_t queue, CameraPacket packet) {
  CameraPacket stale;
  if (xQueueReceive(queue, &stale, 0) == pdTRUE) {
    if (stale.data != nullptr) heap_caps_free(stale.data);
    if (queue == cameraFrames) {
      portENTER_CRITICAL(&cameraMetricsMux);
      cameraMetrics.replacements++;
      portEXIT_CRITICAL(&cameraMetricsMux);
    }
  }
  if (xQueueSend(queue, &packet, 0) == pdTRUE) return true;
  if (packet.data != nullptr) heap_caps_free(packet.data);
  return false;
}
// END CAMERA MAILBOX
volatile unsigned long sentImuPackets = 0, imuReadFailures = 0, imuSendFailures = 0;
struct FallEvent {
  char id[24];
  uint32_t at;
  float peakG, peakGyro, tilt;
};
QueueHandle_t fallQueue = nullptr;
FallEvent pendingFall = {};
bool fallPending = false;
uint32_t fallBootId = 0;
volatile unsigned long fallEvents = 0, fallQueueDrops = 0, imuSamples = 0;
volatile uint32_t lastImuSampleAt = 0;
volatile unsigned long failedFrames = 0;
volatile unsigned long sentAudioPackets = 0;
volatile unsigned long capturedAudioPackets = 0, droppedAudioPackets = 0;
volatile uint16_t microphonePeak = 0;
// BEGIN MICROPHONE LEVELS
struct MicrophoneLevels {
  uint32_t samples = 0, clipped = 0;
  uint64_t sumSquares = 0;
  uint16_t peak = 0;
  int16_t lastValue = 0;
  void add(int16_t value) {
    const int32_t magnitude = value < 0 ? -(int32_t)value : value;
    samples++;
    if (magnitude >= 32760) clipped++;
    if (magnitude > peak) peak = magnitude;
    const int64_t wide = value;
    sumSquares += (uint64_t)(wide * wide);
    lastValue = value;
  }
};
// END MICROPHONE LEVELS
MicrophoneLevels microphoneLevels;
portMUX_TYPE microphoneLevelsMux = portMUX_INITIALIZER_UNLOCKED;
uint32_t nextAudioSequence = 0;

struct AudioPacket {
  uint32_t sequence;
  uint16_t count;
  int16_t pcm[160];
};
QueueHandle_t audioQueue = nullptr;
TaskHandle_t audioTransportTask = nullptr;
volatile bool speakerPlaying = false;
volatile unsigned long playedAudioPackets = 0, droppedSpeakerPackets = 0;
volatile unsigned long speakerUnderruns = 0;

void onCameraEvent(WStype_t type, uint8_t* payload, size_t length) {
  if (type == WStype_PING || type == WStype_PONG)
    Serial.printf("Camera heartbeat: %s uptime_ms=%lu\n", type == WStype_PING ? "RX_PING" : "RX_PONG", millis());
  if (type == WStype_CONNECTED) { cameraConnected = true; Serial.println("Camera channel connected"); }
  if (type == WStype_DISCONNECTED) {
    cameraConnected = false;
    videoEnabled = false;
    Serial.printf("Camera channel disconnected: reason=%.*s\n",
      (int)(payload && length ? (length < 160 ? length : 160) : 0),
      payload ? (const char*)payload : "");
  }
  // All server profiles enable this board's existing 88 ms VGA stream.
  if (type == WStype_TEXT &&
      ((length == 8 && memcmp(payload, "VIDEO:ON", 8) == 0) ||
       (length == 13 && memcmp(payload, "VIDEO:PREVIEW", 13) == 0) ||
       (length == 10 && memcmp(payload, "VIDEO:FAST", 10) == 0))) {
    videoEnabled = true;
    Serial.println("Camera video: ON (task/viewer requested)");
  }
  if (type == WStype_TEXT && length == 9 && memcmp(payload, "VIDEO:OFF", 9) == 0) {
    videoEnabled = false;
    Serial.println("Camera video: OFF (standby)");
  }
  if (type == WStype_TEXT && length > 5 && length <= 15 && memcmp(payload, "SNAP:", 5) == 0) {
    char idText[11] = {};
    memcpy(idText, payload + 5, length - 5);
    const unsigned long id = strtoul(idText, nullptr, 10);
    if (cameraStillRequests != nullptr && id != 0) xQueueOverwrite(cameraStillRequests, &id);
  }
}

void onImuEvent(WStype_t type, uint8_t* payload, size_t length) {
  if (type == WStype_CONNECTED) { imuConnected = true; Serial.println("IMU channel connected"); }
  if (type == WStype_DISCONNECTED) { imuConnected = false; Serial.println("IMU channel disconnected"); }
  if (type == WStype_TEXT && fallPending && length == 4 + strlen(pendingFall.id) &&
      memcmp(payload, "ACK:", 4) == 0 && memcmp(payload + 4, pendingFall.id, length - 4) == 0) {
    Serial.printf("Fall event acknowledged: %s\n", pendingFall.id);
    fallPending = false;
  }
}

void onAudioEvent(WStype_t type, uint8_t* payload, size_t length) {
  if (type == WStype_PING || type == WStype_PONG)
    Serial.printf("Audio heartbeat: %s uptime_ms=%lu\n", type == WStype_PING ? "RX_PING" : "RX_PONG", millis());
  if (type == WStype_CONNECTED) {
    if (audioQueue != nullptr) xQueueReset(audioQueue);
    audioConnected = true;
    Serial.println("Audio channel connected (microphone upload only)");
    audioSocket.sendTXT("DEVICE_AUDIO:HTTP_WAV_V1");
  }
  if (type == WStype_DISCONNECTED || type == WStype_ERROR) {
    audioConnected = false;
    if (audioQueue != nullptr) xQueueReset(audioQueue);
    Serial.printf("Audio channel unavailable: reason=%.*s\n",
      (int)(payload && length ? (length < 160 ? length : 160) : 0),
      payload ? (const char*)payload : "");
  }
}

bool startAudio() {
  const i2s_config_t config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | (MIC_ONLY_DIAGNOSTIC ? 0 : I2S_MODE_TX)),
    .sample_rate = AUDIO_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 128,
    .use_apll = false,
    .tx_desc_auto_clear = !MIC_ONLY_DIAGNOSTIC,
    .fixed_mclk = 0,
  };
  const i2s_pin_config_t pins = {
    .mck_io_num = I2S_PIN_NO_CHANGE,
    .bck_io_num = I2S_BCLK,
    .ws_io_num = I2S_WS,
    .data_out_num = MIC_ONLY_DIAGNOSTIC ? I2S_PIN_NO_CHANGE : I2S_SPK_DOUT,
    .data_in_num = I2S_MIC_SD,
  };
  if (i2s_driver_install(I2S_NUM_0, &config, 0, nullptr) != ESP_OK ||
      i2s_set_pin(I2S_NUM_0, &pins) != ESP_OK) return false;
  i2s_zero_dma_buffer(I2S_NUM_0);
  return true;
}

void testSpeakerAtBoot() {
  // Bypass Wi-Fi and the playback queue using the combined RX/TX configuration.
  int32_t tone[256 * 2];
  for (size_t i = 0; i < 256; i++) {
    const int16_t sample = (int16_t)(6000 * std::sin(6.2831853 * 1000 * i / AUDIO_SAMPLE_RATE));
    tone[2 * i] = tone[2 * i + 1] = (int32_t)sample * 65536;
  }
  bool ok = true;
  for (int block = 0; block < 16; block++) {
    size_t written = 0;
    const esp_err_t result = i2s_write(I2S_NUM_0, tone, sizeof(tone), &written, pdMS_TO_TICKS(1000));
    if (result != ESP_OK || written != sizeof(tone)) {
      Serial.printf("Speaker boot I2S failed: error=%s bytes=%u/%u\n",
                    esp_err_to_name(result), (unsigned)written, (unsigned)sizeof(tone));
      ok = false;
      break;
    }
  }
  delay(80);
  i2s_zero_dma_buffer(I2S_NUM_0);
  Serial.println(ok ? "Speaker boot tone written; verify audible beep" : "Speaker boot tone failed");
}

bool startCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;
  const bool hasPsram = psramFound();
  // Allocate PSRAM frame buffers for UXGA stills, then stream at VGA.
  config.frame_size = hasPsram ? FRAMESIZE_UXGA : FRAMESIZE_QVGA;
  config.jpeg_quality = hasPsram ? 10 : 12;
  config.fb_count = hasPsram ? 2 : 1;
  config.fb_location = hasPsram ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  config.grab_mode = hasPsram ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;
  if (esp_camera_init(&config) != ESP_OK) return false;
  sensor_t* sensor = esp_camera_sensor_get();
  streamFrameSize = hasPsram ? FRAMESIZE_VGA : FRAMESIZE_QVGA;
  if (hasPsram && sensor->set_framesize(sensor, FRAMESIZE_VGA) != 0) return false;
  sensor->set_vflip(sensor, 1);
  sensor->set_hmirror(sensor, 1);
  sensor->set_brightness(sensor, 0);
  sensor->set_whitebal(sensor, 1);
  sensor->set_awb_gain(sensor, 1);
  sensor->set_exposure_ctrl(sensor, 1);
  sensor->set_aec2(sensor, 1);
  sensor->set_gain_ctrl(sensor, 1);
  sensor->set_gainceiling(sensor, GAINCEILING_16X);
  Serial.printf("Camera profile: %s, JPEG quality %d, auto exposure/white balance\n",
                hasPsram ? "VGA 640x480" : "QVGA 320x240", config.jpeg_quality);
  return true;
}

bool mpuWrite(uint8_t address, uint8_t value) {
  Wire.beginTransmission(MPU6500_ADDRESS);
  Wire.write(address); Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool startMpu6500() {
#if ARDUINO_USB_CDC_ON_BOOT && (MPU6500_SDA == 19 || MPU6500_SDA == 20 || MPU6500_SCL == 19 || MPU6500_SCL == 20)
  Serial.println("MPU6500 disabled: GPIO19/20 conflict with USB CDC; select free I2C pins");
  return false;
#endif
  if (!Wire.begin(MPU6500_SDA, MPU6500_SCL, 400000)) return false;
  Wire.setTimeOut(10);
  Wire.beginTransmission(MPU6500_ADDRESS);
  Wire.write(0x75);
  if (Wire.endTransmission(false) != 0 ||
      Wire.requestFrom((uint8_t)MPU6500_ADDRESS, (size_t)1, true) != 1 ||
      Wire.read() != 0x70) return false;
  if (!mpuWrite(0x6B, 0x01)) return false;
  delay(100);
  return mpuWrite(0x6C, 0x00) && mpuWrite(0x1A, 0x03) && mpuWrite(0x19, 9) &&
         mpuWrite(0x1B, 0x10) && mpuWrite(0x1C, 0x10) && mpuWrite(0x1D, 0x03);
}

int16_t mpuRead16() {
  const uint16_t high = Wire.read();
  const uint16_t low = Wire.read();
  return (int16_t)((high << 8) | low);
}

void sampleImu(void*) {
  FallDetector detector;
  TickType_t tick = xTaskGetTickCount();
  unsigned long lastReport = 0;
  while (true) {
    Wire.beginTransmission(MPU6500_ADDRESS);
    Wire.write(0x3B);
    if (Wire.endTransmission(false) != 0 ||
        Wire.requestFrom((uint8_t)MPU6500_ADDRESS, (size_t)14, true) != 14) {
      imuReadFailures++;
      detector.reset(millis());
      vTaskDelay(pdMS_TO_TICKS(1000));
      tick = xTaskGetTickCount();
      continue;
    }
    const float ax = mpuRead16() / 4096.0f, ay = mpuRead16() / 4096.0f, az = mpuRead16() / 4096.0f;
    mpuRead16();
    const float gx = mpuRead16() / 32.8f, gy = mpuRead16() / 32.8f, gz = mpuRead16() / 32.8f;
    imuSamples++;
    const uint32_t now = millis();
    lastImuSampleAt = now;
    if (detector.update(now, ax, ay, az, gx, gy, gz)) {
      FallEvent event = {};
      snprintf(event.id, sizeof(event.id), "%08lx-%lu", (unsigned long)fallBootId, ++fallEvents);
      event.at = now; event.peakG = detector.peakG;
      event.peakGyro = detector.peakGyro; event.tilt = detector.tiltDegrees;
      Serial.printf("FALL SUSPECTED: %s peak=%.2fg tilt=%.1fdeg\n", event.id, event.peakG, event.tilt);
      if (xQueueSend(fallQueue, &event, 0) != pdTRUE) {
        fallQueueDrops++;
        Serial.println("Fall event queue full: event lost");
      }
    }
    if (now - lastReport >= 1000) {
      lastReport = now;
      Serial.printf("Fall monitor state=%d accel=%.2f,%.2f,%.2f gyro=%.1f,%.1f,%.1f\n",
                    (int)detector.state, ax, ay, az, gx, gy, gz);
    }
    vTaskDelayUntil(&tick, pdMS_TO_TICKS(10));
  }
}

void serviceImu(void*) {
  unsigned long lastSentAt = 0, lastHeartbeat = 0;
  while (true) {
    imuSocket.loop();
    const unsigned long now = millis();
    if (!fallPending && xQueueReceive(fallQueue, &pendingFall, 0) == pdTRUE) {
      fallPending = true;
      lastSentAt = now - 2000;
    }
    if (imuConnected && fallPending && now - lastSentAt >= 2000) {
      lastSentAt = now;
      char message[256];
      snprintf(message, sizeof(message),
        "{\"type\":\"fall_suspected\",\"event_id\":\"%s\",\"uptime_ms\":%lu,\"peak_g\":%.2f,\"peak_gyro\":%.1f,\"tilt_deg\":%.1f}",
        pendingFall.id, (unsigned long)pendingFall.at, pendingFall.peakG, pendingFall.peakGyro, pendingFall.tilt);
      if (imuSocket.sendTXT(message)) sentImuPackets++;
      else imuSendFailures++;
    }
    if (imuConnected && now - lastHeartbeat >= 10000) {
      lastHeartbeat = now;
      char message[200];
      snprintf(message, sizeof(message),
        "{\"type\":\"imu_status\",\"uptime_ms\":%lu,\"samples\":%lu,\"read_failures\":%lu,\"queue_drops\":%lu,\"sensor_ok\":%s}",
        now, imuSamples, imuReadFailures, fallQueueDrops,
        imuSamples > 0 && uint32_t(now - lastImuSampleAt) < 500 ? "true" : "false");
      if (!imuSocket.sendTXT(message)) imuSendFailures++;
    }
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

void captureMicrophone(void*) {
  int32_t raw[320 * 2];
  AudioPacket packet;
  while (true) {
    size_t bytesRead = 0;
    if (i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytesRead, portMAX_DELAY) != ESP_OK || bytesRead == 0) continue;
    const size_t capturedFrames = bytesRead / (2 * sizeof(int32_t));
    packet.count = capturedFrames / (AUDIO_SAMPLE_RATE / MIC_UPLOAD_SAMPLE_RATE);
    if (packet.count == 0) continue;
    MicrophoneLevels levels;
    int32_t firstSample = 0;
    for (size_t i = 0; i < packet.count * 2; i++) {
      int32_t value = (raw[2 * i] >> 16) + (raw[2 * i + 1] >> 16);
      value = constrain(value, -32768, 32767);
      levels.add((int16_t)value);
      if ((i & 1) == 0) firstSample = value;
      else packet.pcm[i / 2] = (int16_t)((firstSample + value) / 2);
    }
    capturedAudioPackets++;
    portENTER_CRITICAL(&microphoneLevelsMux);
    microphoneLevels.samples += levels.samples;
    microphoneLevels.clipped += levels.clipped;
    microphoneLevels.sumSquares += levels.sumSquares;
    microphoneLevels.lastValue = levels.lastValue;
    if (levels.peak > microphoneLevels.peak) microphoneLevels.peak = levels.peak;
    if (levels.peak > microphonePeak) microphonePeak = levels.peak;
    portEXIT_CRITICAL(&microphoneLevelsMux);
    if (audioConnected && !speakerPlaying) {
      packet.sequence = nextAudioSequence++;
      if (xQueueSend(audioQueue, &packet, 0) != pdTRUE) {
        AudioPacket stale;
        xQueueReceive(audioQueue, &stale, 0);
        xQueueSend(audioQueue, &packet, 0);
        droppedAudioPackets++;
      }
    }
  }
}

void sendMicrophoneAudio(void*) {
  AudioPacket packet;
  uint8_t wirePacket[8 + 320];
  memcpy(wirePacket, "TEA2", 4);
  while (true) {
    const unsigned long pollStartedAt = millis();
    audioSocket.loop();
    const unsigned long pollMs = millis() - pollStartedAt;
    if (pollMs > 250) Serial.printf("Audio WebSocket poll blocked: %lu ms\n", pollMs);
    if (!audioConnected || audioQueue == nullptr || speakerPlaying) {
      vTaskDelay(pdMS_TO_TICKS(1));
      continue;
    }
    if (xQueueReceive(audioQueue, &packet, 0) == pdTRUE) {
      memcpy(wirePacket + 4, &packet.sequence, sizeof(packet.sequence));
      memcpy(wirePacket + 8, packet.pcm, packet.count * sizeof(int16_t));
      const unsigned long sendStartedAt = millis();
      if (audioSocket.sendBIN(wirePacket, 8 + packet.count * sizeof(int16_t))) {
        sentAudioPackets++;
      } else {
        droppedAudioPackets++;
      }
      const unsigned long sendMs = millis() - sendStartedAt;
      if (sendMs > 100) Serial.printf("Audio WebSocket send blocked: %lu ms queue=%u\n",
        sendMs, (unsigned)uxQueueMessagesWaiting(audioQueue));
    }
    // Let lower-priority camera work run even while draining an audio backlog.
    vTaskDelay(1);
  }
}

// Download each finite WAV before playback so Wi-Fi jitter cannot starve I2S.
uint32_t wavU32(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
         (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

void reportHttpSpeaker(const String& id, bool success, const char* reason) {
  for (int attempt = 0; attempt < 3; attempt++) {
    WiFiClient client;
    HTTPClient http;
    http.setConnectTimeout(2000);
    http.setTimeout(2000);
    const String url = String("http://") + SERVER_HOST + ":" + SERVER_PORT +
      "/speaker/status?token=" + DEVICE_TOKEN + "&id=" + id +
      "&state=" + (success ? "done" : "error") + "&reason=" + reason;
    if (http.begin(client, url)) {
      const int status = http.POST("");
      http.end();
      if (status == 204) return;
      Serial.printf("Speaker status POST: %d\n", status);
    }
    vTaskDelay(pdMS_TO_TICKS(200));
  }
  Serial.println("Speaker completion report failed");
}

void playHttpSpeaker(void*) {
  while (true) {
    if (WiFi.status() != WL_CONNECTED) {
      vTaskDelay(pdMS_TO_TICKS(500));
      continue;
    }
    String id;
    const String downloadKey = String(esp_random(), HEX) + "-" + String(esp_random(), HEX);
    uint8_t* wav = nullptr;
    int length = 0;
    size_t received = 0;
    bool ok = false;
    const char* reason = "idle";
    const uint32_t startedAt = millis();
    // Retry only before I2S starts; a partially played reply must not repeat.
    for (int attempt = 0; attempt < 3; attempt++) {
      WiFiClient client;
      HTTPClient http;
      http.setConnectTimeout(2000);
      http.setTimeout(3000);
      String url = String("http://") + SERVER_HOST + ":" + SERVER_PORT +
        "/stream.wav?token=" + DEVICE_TOKEN;
      url += "&download=" + downloadKey;
      if (!http.begin(client, url)) { reason = "http_begin_failed"; break; }
      const char* headers[] = {"X-Playback-Id"};
      http.collectHeaders(headers, 1);
      const int status = http.GET();
      if (status != 200) {
        reason = status == 204 && id.length() == 0 ? "idle" : "http_get_failed";
        if (status != 204) Serial.printf("Speaker HTTP GET: %d attempt=%d\n", status, attempt + 1);
        http.end();
        if ((status < 0 || status >= 500) && attempt < 2) {
          vTaskDelay(pdMS_TO_TICKS(200));
          continue;
        }
        break;
      }
      const String responseId = http.header("X-Playback-Id");
      if (responseId.length() == 0 || (id.length() > 0 && responseId != id)) {
        reason = "invalid_playback_id";
        http.end();
        break;
      }
      id = responseId;
      length = http.getSize();
      if (length <= 44 || length > 1920044) {
        reason = "invalid_length";
        http.end();
        break;
      }
      const uint32_t caps = psramFound() ? MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT : MALLOC_CAP_8BIT;
      wav = (uint8_t*)heap_caps_malloc(length, caps);
      if (wav == nullptr) {
        reason = "allocation_failed";
        http.end();
        break;
      }
      received = 0;
      uint32_t lastDataAt = millis();
      const uint32_t downloadStartedAt = millis();
      auto* stream = http.getStreamPtr();
      ok = stream != nullptr;
      reason = "download_timeout";
      while (ok && received < (size_t)length) {
        const int available = stream->available();
        if (available > 0) {
          const size_t wanted = ((size_t)available < (size_t)length - received) ?
            (size_t)available : (size_t)length - received;
          const int count = stream->read(wav + received, wanted);
          if (count > 0) { received += count; lastDataAt = millis(); }
        }
        // Also time out a stalled read when available() stays positive.
        if (received < (size_t)length &&
            (!http.connected() || millis() - lastDataAt > 3000 || millis() - downloadStartedAt > 15000)) {
          ok = false;
        }
        vTaskDelay(pdMS_TO_TICKS(1));
      }
      http.end();
      if (ok && received == (size_t)length) break;
      Serial.printf("Speaker HTTP download failed: id=%s attempt=%d received=%u/%d\n",
        id.c_str(), attempt + 1, (unsigned)received, length);
      heap_caps_free(wav);
      wav = nullptr;
      ok = false;
      vTaskDelay(pdMS_TO_TICKS(200));
    }
    if (ok) {
      // The server emits a standard 44-byte PCM WAV header.
      reason = "invalid_wav";
      ok = memcmp(wav, "RIFF", 4) == 0 && memcmp(wav + 8, "WAVEfmt ", 8) == 0 &&
        wavU32(wav + 16) == 16 && wav[20] == 1 && wav[21] == 0 &&
        wav[22] == 1 && wav[23] == 0 && wavU32(wav + 24) == AUDIO_SAMPLE_RATE &&
        wav[32] == 2 && wav[33] == 0 && wav[34] == 16 && wav[35] == 0 &&
        memcmp(wav + 36, "data", 4) == 0 &&
        wavU32(wav + 40) == (uint32_t)length - 44 && (length - 44) % 2 == 0;
    }
    if (ok) {
      Serial.printf("Speaker HTTP START: id=%s bytes=%d download_ms=%lu\n",
                    id.c_str(), length, (unsigned long)(millis() - startedAt));
      speakerPlaying = true;
      if (audioQueue != nullptr) xQueueReset(audioQueue);
      int32_t output[320 * 2];
      reason = "i2s_write_failed";
      for (size_t offset = 44; ok && offset < (size_t)length;) {
        const size_t remaining = ((size_t)length - offset) / 2;
        const size_t samples = remaining < 320 ? remaining : 320;
        for (size_t i = 0; i < samples; i++) {
          const int16_t sample = (int16_t)(uint16_t(wav[offset + 2*i]) |
            (uint16_t(wav[offset + 2*i + 1]) << 8));
          output[2*i] = output[2*i+1] = (int32_t)sample * 65536;
        }
        size_t written = 0;
        const size_t expected = samples * 2 * sizeof(int32_t);
        const esp_err_t result = i2s_write(I2S_NUM_0, output, expected, &written, pdMS_TO_TICKS(100));
        ok = result == ESP_OK && written == expected;
        if (ok) playedAudioPackets++;
        else Serial.printf("Speaker I2S failed: code=%d written=%u/%u\n",
          (int)result, (unsigned)written, (unsigned)expected);
        offset += samples * 2;
      }
      vTaskDelay(pdMS_TO_TICKS(80));
      i2s_zero_dma_buffer(I2S_NUM_0);
      if (audioQueue != nullptr) xQueueReset(audioQueue);
      speakerPlaying = false;
      if (ok) reason = "done";
    }
    if (wav != nullptr) heap_caps_free(wav);
    if (strcmp(reason, "idle") != 0) {
      if (!ok) droppedSpeakerPackets++;
      Serial.printf("Speaker HTTP %s: id=%s reason=%s received=%u/%d heap=%u\n",
        ok ? "DONE" : "ERROR", id.c_str(), reason, (unsigned)received, length, ESP.getFreeHeap());
      if (id.length() > 0) reportHttpSpeaker(id, ok, reason);
    }
    vTaskDelay(pdMS_TO_TICKS(250));
  }
}

CameraPacket copyCameraJpeg(camera_fb_t* frame, unsigned long stillId) {
  CameraPacket packet;
  packet.stillId = stillId;
  if (frame == nullptr || frame->format != PIXFORMAT_JPEG || frame->len < 4 || frame->len > 2000000) return packet;
  const uint32_t start = micros();
  const uint32_t caps = psramFound() ? MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT : MALLOC_CAP_8BIT;
  packet.data = (uint8_t*)heap_caps_malloc(frame->len, caps);
  if (packet.data == nullptr) {
    portENTER_CRITICAL(&cameraMetricsMux);
    cameraMetrics.allocationFailures++;
    portEXIT_CRITICAL(&cameraMetricsMux);
    return packet;
  }
  memcpy(packet.data, frame->buf, frame->len);
  packet.length = frame->len;
  packet.capturedAt = millis();
  if (stillId == 0) {
    portENTER_CRITICAL(&cameraMetricsMux);
    cameraMetrics.captures++;
    cameraMetrics.copyUs += uint32_t(micros() - start);
    portEXIT_CRITICAL(&cameraMetricsMux);
  }
  return packet;
}

void discardCameraPackets(QueueHandle_t queue) {
  CameraPacket packet;
  while (xQueueReceive(queue, &packet, 0) == pdTRUE) {
    if (packet.data != nullptr) heap_caps_free(packet.data);
  }
}

void recordCameraFailure() {
  portENTER_CRITICAL(&cameraMetricsMux);
  failedFrames++;
  portEXIT_CRITICAL(&cameraMetricsMux);
}

void captureCamera(void*) {
  unsigned long lastCaptureAt = 0;
  while (true) {
    if (!cameraConnected) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }
    unsigned long stillId = 0;
    if (xQueueReceive(cameraStillRequests, &stillId, 0) == pdTRUE) {
      discardCameraPackets(cameraFrames);
      const uint32_t startedAt = millis();
      CameraPacket result;
      result.stillId = stillId;
      sensor_t* sensor = esp_camera_sensor_get();
      if (psramFound() && sensor != nullptr && sensor->set_framesize(sensor, FRAMESIZE_UXGA) == 0) {
        for (int i = 0; i < 2; i++) {
          camera_fb_t* stale = esp_camera_fb_get();
          if (stale != nullptr) esp_camera_fb_return(stale);
        }
        for (int attempt = 0; attempt < 2 && result.data == nullptr; attempt++) {
          camera_fb_t* frame = esp_camera_fb_get();
          if (frame != nullptr) {
            if (frame->width == 1600 && frame->height == 1200) result = copyCameraJpeg(frame, stillId);
            esp_camera_fb_return(frame);
          }
        }
      }
      // This task alone owns sensor mode changes and driver frame buffers.
      if (sensor != nullptr && sensor->set_framesize(sensor, streamFrameSize) == 0) {
        for (int i = 0; i < 2; i++) {
          camera_fb_t* stale = esp_camera_fb_get();
          if (stale != nullptr) esp_camera_fb_return(stale);
        }
      }
      Serial.printf("Text capture: id=%lu bytes=%u capture_ms=%lu\n",
                    stillId, (unsigned)result.length, (unsigned long)(millis() - startedAt));
      publishCameraPacket(cameraStillResults, result);
      lastCaptureAt = millis();
      continue;
    }
    if (!videoEnabled) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }
    if (millis() - lastCaptureAt < CAMERA_FRAME_INTERVAL_MS) {
      vTaskDelay(pdMS_TO_TICKS(1));
      continue;
    }
    lastCaptureAt = millis();
    const uint32_t start = micros();
    camera_fb_t* frame = esp_camera_fb_get();
    const uint32_t grabUs = uint32_t(micros() - start);
    portENTER_CRITICAL(&cameraMetricsMux);
    cameraMetrics.grabs++;
    cameraMetrics.grabUs += grabUs;
    if (grabUs > cameraMetrics.grabMaxUs) cameraMetrics.grabMaxUs = grabUs;
    portEXIT_CRITICAL(&cameraMetricsMux);
    CameraPacket packet = copyCameraJpeg(frame, 0);
    if (frame != nullptr) esp_camera_fb_return(frame);
    if (packet.data != nullptr) {
      if (!publishCameraPacket(cameraFrames, packet)) recordCameraFailure();
    } else recordCameraFailure();
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

void serviceCamera(void*) {
  while (true) {
    cameraSocket.loop();
    if (!cameraConnected) {
      discardCameraPackets(cameraFrames);
      discardCameraPackets(cameraStillResults);
      xQueueReset(cameraStillRequests);
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }
    CameraPacket packet;
    const bool still = xQueueReceive(cameraStillResults, &packet, 0) == pdTRUE;
    if (!still && !videoEnabled) {
      discardCameraPackets(cameraFrames);
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }
    if (!still && xQueueReceive(cameraFrames, &packet, 0) != pdTRUE) {
      vTaskDelay(pdMS_TO_TICKS(1));
      continue;
    }
    const uint32_t ageMs = uint32_t(millis() - packet.capturedAt);
    if (!still && ageMs > 500) {
      heap_caps_free(packet.data);
      portENTER_CRITICAL(&cameraMetricsMux);
      cameraMetrics.expired++;
      portEXIT_CRITICAL(&cameraMetricsMux);
      continue;
    }
    const uint32_t start = micros();
    bool sent = false;
    if (still && packet.data == nullptr) {
      cameraSocket.sendTXT(String("SNAP_ERROR:") + packet.stillId);
    } else {
      const bool markerSent = !still || cameraSocket.sendTXT(String("SNAP:") + packet.stillId);
      if (markerSent) sent = cameraSocket.sendBIN(packet.data, packet.length);
    }
    const uint32_t sendUs = uint32_t(micros() - start);
    if (still) {
      Serial.printf("Text send: id=%lu bytes=%u send_ms=%.1f sent=%d\n",
                    packet.stillId, (unsigned)packet.length, sendUs / 1000.0f, sent);
      // After a marker, a failed binary send must not be followed by a live frame.
      if (packet.data != nullptr && !sent) cameraSocket.disconnect();
    } else {
      if (sent) sentFrames++;
      else { recordCameraFailure(); cameraSocket.disconnect(); }
      portENTER_CRITICAL(&cameraMetricsMux);
      cameraMetrics.sends++;
      cameraMetrics.sendUs += sendUs;
      cameraMetrics.ageMs += ageMs;
      cameraMetrics.jpegBytes += packet.length;
      if (sendUs > cameraMetrics.sendMaxUs) cameraMetrics.sendMaxUs = sendUs;
      portEXIT_CRITICAL(&cameraMetricsMux);
    }
    if (packet.data != nullptr) heap_caps_free(packet.data);
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

void connectSockets() {
  const String base = String("?token=") + DEVICE_TOKEN;
  if (!PC_AUDIO_CAMERA_TEST) {
    audioSocket.begin(SERVER_HOST, SERVER_PORT, (String("/ws/audio") + base).c_str());
    audioSocket.onEvent(onAudioEvent);
    audioSocket.setReconnectInterval(2000);
    audioSocket.enableHeartbeat(15000, 5000, 3);
  }
  if (!MIC_ONLY_DIAGNOSTIC) {
    cameraSocket.begin(SERVER_HOST, SERVER_PORT, (String("/ws/camera") + base).c_str());
    cameraSocket.onEvent(onCameraEvent);
    cameraSocket.setReconnectInterval(2000);
    cameraSocket.enableHeartbeat(15000, 5000, 3);
    if (imuReady) {
      imuSocket.begin(SERVER_HOST, SERVER_PORT, (String("/ws/imu") + base).c_str());
      imuSocket.onEvent(onImuEvent);
      imuSocket.setReconnectInterval(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("ThirdEye AI device starting");
  Serial.printf("Audio firmware: HTTP_WAV_V1; speaker_boot_test=%d\n", SPEAKER_BOOT_TEST);
  Serial.println("Microphone upload: 8 kHz PCM (TEA2); speaker: 16 kHz");
  Serial.println("Audio recovery: bounded HTTP download retries + send timing + scheduler yield");
#ifdef WEBSOCKETS_VERSION
  Serial.printf("WebSockets library: %s\n", WEBSOCKETS_VERSION);
#endif
  Serial.printf("Server target: %s:%u\n", SERVER_HOST, (unsigned)SERVER_PORT);
  if (!MIC_ONLY_DIAGNOSTIC) {
    if (!startCamera()) {
      Serial.println("Camera: failed; check camera pins and PSRAM");
      while (true) delay(1000);
    }
    Serial.println("Camera: ready");
  }
  const bool audioReady = !PC_AUDIO_CAMERA_TEST && startAudio();
  Serial.printf("Audio: %s\n", PC_AUDIO_CAMERA_TEST ? "disabled (PC audio test)" : audioReady ? "ready" : "failed");
  if (audioReady && !MIC_ONLY_DIAGNOSTIC && SPEAKER_BOOT_TEST) testSpeakerAtBoot();
  if (ENABLE_IMU && !MIC_ONLY_DIAGNOSTIC) {
    imuReady = startMpu6500();
    Serial.printf("MPU6500: %s\n", imuReady ? "ready" : "failed (camera/audio continue)");
  }
  if (imuReady) {
    fallBootId = esp_random();
    fallQueue = xQueueCreate(8, sizeof(FallEvent));
    if (fallQueue == nullptr || xTaskCreate(sampleImu, "fall-sample", 4096, nullptr, 2, nullptr) != pdPASS) {
      imuReady = false;
      Serial.println("Fall sampling task failed; camera/audio continue");
    }
  }
  if (audioReady) {
    audioQueue = xQueueCreate(48, sizeof(AudioPacket));
    if (audioQueue == nullptr || xTaskCreate(captureMicrophone, "mic-rx", 6144, nullptr, 2, nullptr) != pdPASS) {
      Serial.println("Audio task failed");
      while (true) delay(1000);
    }
    if (!MIC_ONLY_DIAGNOSTIC) {
      if (xTaskCreate(playHttpSpeaker, "speaker-http", 8192, nullptr, 2, nullptr) != pdPASS) {
        Serial.println("Speaker task failed");
        while (true) delay(1000);
      }
    }
    Serial.println(MIC_ONLY_DIAGNOSTIC ? "Microphone RX only; speaker/camera/IMU disabled" :
                   "Audio capture and speaker playback ready");
  }
  WiFi.onEvent([](WiFiEvent_t event, WiFiEventInfo_t info) {
    if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED)
      Serial.printf("Wi-Fi disconnected: reason=%u uptime_ms=%lu\n",
        (unsigned)info.wifi_sta_disconnected.reason, millis());
    else if (event == ARDUINO_EVENT_WIFI_STA_LOST_IP)
      Serial.printf("Wi-Fi lost IP: uptime_ms=%lu\n", millis());
  });
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print('.'); }
  Serial.printf("\nWi-Fi: %s\n", WiFi.localIP().toString().c_str());
  if (!MIC_ONLY_DIAGNOSTIC) {
    cameraFrames = xQueueCreate(1, sizeof(CameraPacket));
    cameraStillRequests = xQueueCreate(1, sizeof(unsigned long));
    cameraStillResults = xQueueCreate(1, sizeof(CameraPacket));
    if (cameraFrames == nullptr || cameraStillRequests == nullptr || cameraStillResults == nullptr) {
      Serial.println("Camera queues failed");
      while (true) delay(1000);
    }
  }
  connectSockets();
  if (!MIC_ONLY_DIAGNOSTIC &&
      (xTaskCreatePinnedToCore(captureCamera, "camera-capture", 6144, nullptr, 1, nullptr, 1) != pdPASS ||
       xTaskCreatePinnedToCore(serviceCamera, "camera-ws", 6144, nullptr, 2, nullptr, 0) != pdPASS)) {
    Serial.println("Camera tasks failed");
    while (true) delay(1000);
  }
  if (imuReady && xTaskCreate(serviceImu, "imu-ws", 4096, nullptr, 1, nullptr) != pdPASS) {
    imuReady = false;
    Serial.println("IMU task failed; camera/audio continue");
  }
  if (audioReady &&
      xTaskCreatePinnedToCore(sendMicrophoneAudio, "audio-ws", 8196, nullptr, 3,
                              &audioTransportTask, 0) != pdPASS) {
    Serial.println("Audio transport task failed");
    while (true) delay(1000);
  }
}

void loop() {
  static unsigned long lastMicReportAt = 0;
  const unsigned long now = millis();
  if (now - lastMicReportAt >= 1000) {
    const unsigned long elapsedMs = now - lastMicReportAt;
    lastMicReportAt = now;
    portENTER_CRITICAL(&microphoneLevelsMux);
    const MicrophoneLevels levels = microphoneLevels;
    microphoneLevels = MicrophoneLevels();
    portEXIT_CRITICAL(&microphoneLevelsMux);
    const double rms = levels.samples ? std::sqrt((double)levels.sumSquares / levels.samples) : 0;
    Serial.printf("mic_value=%d mic_peak_1s=%u mic_rms=%.1f mic_samples=%lu mic_samples/s=%.0f mic_clipped=%.1f%% audio_connected=%d speaker_playing=%d mic_active=%d\n",
      (int)levels.lastValue, (unsigned)levels.peak, rms, (unsigned long)levels.samples,
      levels.samples * 1000.0 / elapsedMs, levels.samples ? levels.clipped * 100.0 / levels.samples : 0,
      audioConnected, speakerPlaying, audioQueue != nullptr);
    Serial.printf("speaker_http=1 speaker_played=%lu speaker_drop=%lu speaker_underrun=%lu\n",
      playedAudioPackets, droppedSpeakerPackets, speakerUnderruns);
  }
  if (millis() - lastStatusAt >= 3000) {
    static unsigned long previousCaptured = 0, previousSent = 0, previousFrames = 0;
    const unsigned long elapsedMs = millis() - lastStatusAt;
    const unsigned long captureRate = (capturedAudioPackets - previousCaptured) * 1000 / elapsedMs;
    const unsigned long sendRate = (sentAudioPackets - previousSent) * 1000 / elapsedMs;
    const unsigned long fps = (sentFrames - previousFrames) * 1000 / elapsedMs;
    previousCaptured = capturedAudioPackets;
    previousSent = sentAudioPackets;
    previousFrames = sentFrames;
    lastStatusAt = millis();
    portENTER_CRITICAL(&cameraMetricsMux);
    const CameraMetrics metrics = cameraMetrics;
    cameraMetrics = CameraMetrics();
    portEXIT_CRITICAL(&cameraMetricsMux);
    portENTER_CRITICAL(&microphoneLevelsMux);
    const uint16_t peakSnapshot = microphonePeak;
    microphonePeak = 0;
    portEXIT_CRITICAL(&microphoneLevelsMux);
    Serial.printf("wifi=%d camera=%d audio=%d fps=%lu frames=%lu frame_fail=%lu mic_capture/s=%lu mic_tx/s=%lu mic_drop=%lu mic_queue=%u mic_peak=%u speaker_played=%lu speaker_drop=%lu speaker_underrun=%lu heap=%u\n",
      WiFi.status(), cameraConnected, audioConnected,
      fps, sentFrames, failedFrames, captureRate, sendRate, droppedAudioPackets,
      audioQueue == nullptr ? 0 : (unsigned)uxQueueMessagesWaiting(audioQueue),
      peakSnapshot, playedAudioPackets, droppedSpeakerPackets, speakerUnderruns, ESP.getFreeHeap());
    Serial.printf("imu=%d imu_tx=%lu imu_read_fail=%lu imu_send_fail=%lu\n",
      imuConnected, sentImuPackets, imuReadFailures, imuSendFailures);
    Serial.printf("camera_perf capture/s=%.1f grab_ms=%.1f grab_max_ms=%.1f copy_ms=%.1f send_ms=%.1f send_max_ms=%.1f queue_ms=%.1f jpeg_kb=%.1f replaced=%lu expired=%lu alloc_fail=%lu rssi=%d psram_free=%u\n",
      metrics.captures * 1000.0f / elapsedMs,
      metrics.grabs ? metrics.grabUs / (metrics.grabs * 1000.0f) : 0.0f,
      metrics.grabMaxUs / 1000.0f,
      metrics.captures ? metrics.copyUs / (metrics.captures * 1000.0f) : 0.0f,
      metrics.sends ? metrics.sendUs / (metrics.sends * 1000.0f) : 0.0f,
      metrics.sendMaxUs / 1000.0f,
      metrics.sends ? metrics.ageMs / (float)metrics.sends : 0.0f,
      metrics.sends ? metrics.jpegBytes / (metrics.sends * 1024.0f) : 0.0f,
      (unsigned long)metrics.replacements, (unsigned long)metrics.expired,
      (unsigned long)metrics.allocationFailures, WiFi.RSSI(), ESP.getFreePsram());
  }
  delay(1);
}
