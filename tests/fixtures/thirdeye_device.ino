#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <Wire.h>
#include <driver/i2s.h>
#include <freertos/queue.h>
#include <stdlib.h>
#include "esp_camera.h"

// Fill these values for your trusted local PC before uploading.
#define WIFI_SSID "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#define SERVER_HOST "YOUR_AI_HOST_IP"
#define SERVER_PORT 8081
#define DEVICE_TOKEN "YOUR_PRIVATE_TOKEN"
// Temporary isolation test. Set to 0 after confirming a clean microphone WAV.
#define MIC_ONLY_DIAGNOSTIC 0
// Use the ESP32 camera with the PC microphone and speaker during AI testing.
#define PC_AUDIO_CAMERA_TEST 1
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

#define MPU6050_SDA 19
#define MPU6050_SCL 20
#define MPU6050_ADDRESS 0x68
#define AUDIO_SAMPLE_RATE 16000
#define CAMERA_FRAME_INTERVAL_MS 70

WebSocketsClient cameraSocket, audioSocket, imuSocket;
volatile bool cameraConnected = false, audioConnected = false, imuConnected = false;
unsigned long pendingStillId = 0;
unsigned long lastCameraAt = 0, lastImuAt = 0, lastStatusAt = 0;
unsigned long sentFrames = 0, sentImuPackets = 0;
volatile unsigned long sentAudioPackets = 0;
volatile unsigned long capturedAudioPackets = 0, droppedAudioPackets = 0;
volatile uint16_t microphonePeak = 0;

struct AudioPacket {
  uint16_t count;
  int16_t pcm[320];
};
QueueHandle_t audioQueue = nullptr;
QueueHandle_t speakerQueue = nullptr;
TaskHandle_t audioTransportTask = nullptr;
volatile bool speakerPlaying = false;
volatile bool speakerEnding = false;
volatile unsigned long playedAudioPackets = 0, droppedSpeakerPackets = 0;
volatile unsigned long speakerUnderruns = 0;

void onCameraEvent(WStype_t type, uint8_t* payload, size_t length) {
  if (type == WStype_CONNECTED) { cameraConnected = true; Serial.println("Camera channel connected"); }
  if (type == WStype_DISCONNECTED) { cameraConnected = false; pendingStillId = 0; Serial.println("Camera channel disconnected"); }
  if (type == WStype_TEXT && length > 5 && length <= 15 && memcmp(payload, "SNAP:", 5) == 0) {
    char idText[11] = {};
    memcpy(idText, payload + 5, length - 5);
    pendingStillId = strtoul(idText, nullptr, 10);
  }
}

void onImuEvent(WStype_t type, uint8_t*, size_t) {
  if (type == WStype_CONNECTED) { imuConnected = true; Serial.println("IMU channel connected"); }
  if (type == WStype_DISCONNECTED) { imuConnected = false; Serial.println("IMU channel disconnected"); }
}

void onAudioEvent(WStype_t type, uint8_t* payload, size_t length) {
  if (type == WStype_CONNECTED) { audioConnected = true; Serial.println("Audio channel connected"); }
  if (type == WStype_DISCONNECTED) {
    audioConnected = false;
    speakerPlaying = false;
    speakerEnding = false;
    if (speakerQueue != nullptr) xQueueReset(speakerQueue);
    Serial.println("Audio channel disconnected");
  }
  if (speakerQueue != nullptr && type == WStype_TEXT && length == 5 && memcmp(payload, "START", 5) == 0) {
    xQueueReset(speakerQueue);
    speakerPlaying = true;
    speakerEnding = false;
  }
  if (speakerQueue != nullptr && type == WStype_TEXT && length == 4 && memcmp(payload, "STOP", 4) == 0) {
    speakerEnding = true;
  }
  if (type == WStype_BIN && speakerQueue != nullptr && length >= 2 && length <= 640 && length % 2 == 0) {
    AudioPacket packet;
    packet.count = length / 2;
    memcpy(packet.pcm, payload, length);
    if (xQueueSend(speakerQueue, &packet, 0) != pdTRUE) droppedSpeakerPackets++;
  }
}

bool startAudio() {
  const i2s_config_t config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | (MIC_ONLY_DIAGNOSTIC ? 0 : I2S_MODE_TX)),
    .sample_rate = AUDIO_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = !MIC_ONLY_DIAGNOSTIC,
    .fixed_mclk = 0,
  };
  const i2s_pin_config_t pins = {
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

void mpuWrite(uint8_t address, uint8_t value) {
  Wire.beginTransmission(MPU6050_ADDRESS);
  Wire.write(address); Wire.write(value); Wire.endTransmission();
}

bool startMpu6050() {
  Wire.begin(MPU6050_SDA, MPU6050_SCL, 400000);
  Wire.beginTransmission(MPU6050_ADDRESS);
  if (Wire.endTransmission() != 0) return false;
  mpuWrite(0x6B, 0x00);
  mpuWrite(0x1B, 0x00);
  mpuWrite(0x1C, 0x00);
  return true;
}

int16_t mpuRead16() { return (int16_t)((Wire.read() << 8) | Wire.read()); }

void sendImu() {
  Wire.beginTransmission(MPU6050_ADDRESS);
  Wire.write(0x3B); Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDRESS, 14, true);
  if (Wire.available() < 14) return;
  const float ax = mpuRead16() / 16384.0f, ay = mpuRead16() / 16384.0f, az = mpuRead16() / 16384.0f;
  mpuRead16();
  const float gx = mpuRead16() / 131.0f, gy = mpuRead16() / 131.0f, gz = mpuRead16() / 131.0f;
  char message[160];
  snprintf(message, sizeof(message), "{\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,\"gx\":%.2f,\"gy\":%.2f,\"gz\":%.2f}", ax, ay, az, gx, gy, gz);
  imuSocket.sendTXT(message);
  sentImuPackets++;
}

void captureMicrophone(void*) {
  int32_t raw[320];
  AudioPacket packet;
  while (true) {
    size_t bytesRead = 0;
    if (i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytesRead, portMAX_DELAY) != ESP_OK || bytesRead == 0) continue;
    packet.count = bytesRead / sizeof(int32_t);
    uint16_t peak = 0;
    for (size_t i = 0; i < packet.count; i++) {
      packet.pcm[i] = (int16_t)(raw[i] >> 16);
      const uint16_t magnitude = packet.pcm[i] == INT16_MIN ? INT16_MAX : abs(packet.pcm[i]);
      if (magnitude > peak) peak = magnitude;
    }
    capturedAudioPackets++;
    if (peak > microphonePeak) microphonePeak = peak;
    if (audioConnected && !speakerPlaying && xQueueSend(audioQueue, &packet, 0) != pdTRUE) {
      AudioPacket stale;
      xQueueReceive(audioQueue, &stale, 0);
      xQueueSend(audioQueue, &packet, 0);
      droppedAudioPackets++;
    }
  }
}

void sendMicrophoneAudio(void*) {
  // Keep microphone networking independent from large camera JPEG sends.
  static int16_t batch[320 * 4];
  while (true) {
    audioSocket.loop();
    if (!audioConnected || audioQueue == nullptr || speakerPlaying) {
      vTaskDelay(pdMS_TO_TICKS(1));
      continue;
    }

    AudioPacket packet;
    size_t sampleCount = 0;
    unsigned long sourcePackets = 0;
    while (sourcePackets < 4 &&
           xQueueReceive(audioQueue, &packet, 0) == pdTRUE &&
           sampleCount + packet.count <= (sizeof(batch) / sizeof(batch[0]))) {
      memcpy(batch + sampleCount, packet.pcm, packet.count * sizeof(int16_t));
      sampleCount += packet.count;
      sourcePackets++;
    }
    if (sampleCount > 0) {
      audioSocket.sendBIN((uint8_t*)batch, sampleCount * sizeof(int16_t));
      sentAudioPackets += sourcePackets;
      continue;
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

void playSpeaker(void*) {
  AudioPacket packet;
  int32_t output[320];
  bool primed = false;
  while (true) {
    if (!speakerPlaying) {
      primed = false;
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }
    if (!primed) {
      const UBaseType_t buffered = uxQueueMessagesWaiting(speakerQueue);
      if (buffered >= 6 || (speakerEnding && buffered > 0)) {
        primed = true;
      } else if (!speakerEnding) {
        vTaskDelay(pdMS_TO_TICKS(5));
        continue;
      }
    }
    if (xQueueReceive(speakerQueue, &packet, pdMS_TO_TICKS(20)) != pdTRUE) {
      if (speakerEnding) {
        vTaskDelay(pdMS_TO_TICKS(60));
        if (speakerEnding && uxQueueMessagesWaiting(speakerQueue) == 0) {
          i2s_zero_dma_buffer(I2S_NUM_0);
          speakerPlaying = false;
          speakerEnding = false;
          primed = false;
        }
      } else if (primed) {
        speakerUnderruns++;
        primed = false;
      }
      continue;
    }
    for (size_t i = 0; i < packet.count; i++) output[i] = (int32_t)packet.pcm[i] * 65536;
    size_t bytesWritten = 0;
    if (i2s_write(I2S_NUM_0, output, packet.count * sizeof(int32_t), &bytesWritten, portMAX_DELAY) == ESP_OK) {
      playedAudioPackets++;
    }
  }
}

void connectSockets() {
  const String base = String("?token=") + DEVICE_TOKEN;
  if (!PC_AUDIO_CAMERA_TEST) {
    audioSocket.begin(SERVER_HOST, SERVER_PORT, (String("/ws/audio") + base).c_str());
    audioSocket.onEvent(onAudioEvent);
    audioSocket.setReconnectInterval(2000);
  }
  if (!MIC_ONLY_DIAGNOSTIC) {
    cameraSocket.begin(SERVER_HOST, SERVER_PORT, (String("/ws/camera") + base).c_str());
    imuSocket.begin(SERVER_HOST, SERVER_PORT, (String("/ws/imu") + base).c_str());
    cameraSocket.onEvent(onCameraEvent); imuSocket.onEvent(onImuEvent);
    cameraSocket.setReconnectInterval(2000); imuSocket.setReconnectInterval(2000);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println(MIC_ONLY_DIAGNOSTIC ? "ThirdEye MIC-ONLY diagnostic" :
                 PC_AUDIO_CAMERA_TEST ? "ThirdEye camera + PC audio test" : "ThirdEye device starting");
  if (!MIC_ONLY_DIAGNOSTIC) Serial.printf("Camera: %s\n", startCamera() ? "ready" : "failed");
  const bool audioReady = !PC_AUDIO_CAMERA_TEST && startAudio();
  Serial.printf("Audio: %s\n", PC_AUDIO_CAMERA_TEST ? "disabled (PC audio test)" : audioReady ? "ready" : "failed");
  if (!MIC_ONLY_DIAGNOSTIC) Serial.printf("MPU6050: %s\n", startMpu6050() ? "ready" : "failed");
  if (audioReady) {
    audioQueue = xQueueCreate(48, sizeof(AudioPacket));
    if (audioQueue == nullptr || xTaskCreate(captureMicrophone, "mic-rx", 6144, nullptr, 2, nullptr) != pdPASS) {
      Serial.println("Audio task failed");
      while (true) delay(1000);
    }
    if (!MIC_ONLY_DIAGNOSTIC) {
      speakerQueue = xQueueCreate(16, sizeof(AudioPacket));
      if (speakerQueue == nullptr || xTaskCreate(playSpeaker, "speaker-tx", 4096, nullptr, 2, nullptr) != pdPASS) {
        Serial.println("Speaker task failed");
        while (true) delay(1000);
      }
    }
    Serial.println(MIC_ONLY_DIAGNOSTIC ? "Microphone RX only; speaker/camera/IMU disabled" :
                   "Audio capture and speaker playback ready");
  }
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print('.'); }
  Serial.printf("\nWi-Fi: %s\n", WiFi.localIP().toString().c_str());
  connectSockets();
  if (audioReady &&
      xTaskCreatePinnedToCore(sendMicrophoneAudio, "audio-ws", 4096, nullptr, 3,
                              &audioTransportTask, 0) != pdPASS) {
    Serial.println("Audio transport task failed");
    while (true) delay(1000);
  }
}

void loop() {
  if (!MIC_ONLY_DIAGNOSTIC) { cameraSocket.loop(); imuSocket.loop(); }
  if (imuConnected && millis() - lastImuAt >= 50) { lastImuAt = millis(); sendImu(); }
  if (cameraConnected && pendingStillId != 0) {
    const unsigned long stillId = pendingStillId;
    pendingStillId = 0;
    bool sent = false;
    sensor_t* sensor = esp_camera_sensor_get();
    if (psramFound() && sensor != nullptr && sensor->set_framesize(sensor, FRAMESIZE_UXGA) == 0) {
      // Two frame buffers may still contain VGA data after switching modes.
      for (int discard = 0; discard < 2; discard++) {
        camera_fb_t* stale = esp_camera_fb_get();
        if (stale != nullptr) esp_camera_fb_return(stale);
      }
      for (int attempt = 0; attempt < 2 && !sent; attempt++) {
        camera_fb_t* frame = esp_camera_fb_get();
        if (frame != nullptr) {
          if (frame->width == 1600 && frame->height == 1200 && frame->format == PIXFORMAT_JPEG) {
            cameraSocket.sendTXT(String("SNAP:") + stillId);
            sent = cameraSocket.sendBIN(frame->buf, frame->len);
            Serial.printf("Text still: %ux%u, %u bytes, sent=%d\n", frame->width, frame->height, (unsigned)frame->len, sent);
          }
          esp_camera_fb_return(frame);
        }
      }
    }
    if (sensor != nullptr && sensor->set_framesize(sensor, FRAMESIZE_VGA) == 0) {
      for (int discard = 0; discard < 2; discard++) {
        camera_fb_t* stale = esp_camera_fb_get();
        if (stale != nullptr) esp_camera_fb_return(stale);
      }
    }
    if (!sent) cameraSocket.sendTXT(String("SNAP_ERROR:") + stillId);
    lastCameraAt = millis();
  }
  if (cameraConnected && (PC_AUDIO_CAMERA_TEST ||
      // Permit a near-empty queue so continuous microphone capture does not starve video.
      (audioQueue != nullptr && uxQueueMessagesWaiting(audioQueue) < 2)) &&
      millis() - lastCameraAt >= CAMERA_FRAME_INTERVAL_MS) {
    lastCameraAt = millis();
    camera_fb_t* frame = esp_camera_fb_get();
    if (frame) { cameraSocket.sendBIN(frame->buf, frame->len); sentFrames++; esp_camera_fb_return(frame); }
  }
  if (millis() - lastStatusAt >= 3000) {
    static unsigned long previousCaptured = 0, previousSent = 0;
    const unsigned long captureRate = (capturedAudioPackets - previousCaptured) / 3;
    const unsigned long sendRate = (sentAudioPackets - previousSent) / 3;
    previousCaptured = capturedAudioPackets;
    previousSent = sentAudioPackets;
    lastStatusAt = millis();
    Serial.printf("wifi=%d camera=%d audio=%d imu=%d frames=%lu mic_capture/s=%lu mic_tx_attempt/s=%lu mic_drop=%lu mic_queue=%u mic_peak=%u imu_packets=%lu speaker_played=%lu speaker_drop=%lu speaker_underrun=%lu heap=%u\n",
      WiFi.status(), cameraConnected, audioConnected, imuConnected,
      sentFrames, captureRate, sendRate, droppedAudioPackets,
      audioQueue == nullptr ? 0 : (unsigned)uxQueueMessagesWaiting(audioQueue),
      microphonePeak, sentImuPackets, playedAudioPackets, droppedSpeakerPackets, speakerUnderruns, ESP.getFreeHeap());
    microphonePeak = 0;
  }
  delay(1);
}
