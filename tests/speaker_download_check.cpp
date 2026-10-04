#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

// Host doubles exercise the actual firmware download/playback function.
struct String : std::string {
  using std::string::string;
  String(const std::string& value) : std::string(value) {}
  String(unsigned value, int) : std::string(std::to_string(value)) {}
};
String operator+(const String& a, const char* b) { return String(std::string(a) + b); }
String operator+(const String& a, const String& b) { return String(std::string(a) + std::string(b)); }
String operator+(const char* a, const String& b) { return String(std::string(a) + std::string(b)); }
String operator+(const String& a, int b) { return String(std::string(a) + std::to_string(b)); }
constexpr int HEX = 16, WL_CONNECTED = 3, AUDIO_SAMPLE_RATE = 16000;
constexpr int MALLOC_CAP_SPIRAM = 1, MALLOC_CAP_8BIT = 2, I2S_NUM_0 = 0, ESP_OK = 0;
const char* SERVER_HOST = "test";
const char* DEVICE_TOKEN = "test";
constexpr int SERVER_PORT = 8081;
using esp_err_t = int;
struct Stop {};
struct Response { int status; std::vector<uint8_t> body; bool stalled = false; };
std::vector<Response> responses;
std::vector<std::string> requests;
unsigned tick, getCount, allocations, writes;
bool failAllocation, failI2s, shortWrite, succeeded, speakerPlaying;
unsigned long playedAudioPackets, droppedSpeakerPackets;
std::string failure;
void* audioQueue = nullptr;
unsigned long millis() { return tick; }
unsigned esp_random() { return 1234; }
unsigned pdMS_TO_TICKS(unsigned value) { return value; }
void vTaskDelay(unsigned value) { tick += value; assert(tick < 100000); }
bool psramFound() { return true; }
void* heap_caps_malloc(size_t size, unsigned) {
  if (failAllocation) return nullptr;
  ++allocations;
  return std::malloc(size);
}
void heap_caps_free(void* p) { assert(allocations); --allocations; std::free(p); }
void xQueueReset(void*) {}
void i2s_zero_dma_buffer(int) {}
int i2s_write(int, void*, size_t expected, size_t* written, unsigned) {
  ++writes;
  *written = shortWrite ? expected - 4 : expected;
  return failI2s ? -1 : ESP_OK;
}
struct { int status() { return WL_CONNECTED; } } WiFi;
struct { unsigned getFreeHeap() { return 100000; } } ESP;
struct { template<class... T> void printf(const char*, T...) {} void println(const char*) {} } Serial;
enum WStype_t { WStype_PING, WStype_PONG, WStype_CONNECTED, WStype_DISCONNECTED, WStype_TEXT };
bool cameraConnected = false, videoEnabled = false;
void* cameraStillRequests = nullptr;
void xQueueOverwrite(void*, const unsigned long*) {}
struct WiFiClient {};
struct HTTPClient {
  Response* response = nullptr;
  size_t offset = 0;
  void setConnectTimeout(int) {}
  void setTimeout(int) {}
  bool begin(WiFiClient&, const String& url) { requests.push_back(url); return true; }
  void collectHeaders(const char**, int) {}
  int GET() { assert(getCount < responses.size()); response = &responses[getCount++]; return response->status; }
  String header(const char*) { return String("42"); }
  int getSize() { return 684; }
  HTTPClient* getStreamPtr() { return this; }
  int available() { return response->stalled ? 1 : static_cast<int>(response->body.size() - offset); }
  int read(uint8_t* dest, size_t wanted) {
    if (response->stalled) return 0;
    size_t count = std::min(wanted, response->body.size() - offset);
    std::memcpy(dest, response->body.data() + offset, count);
    offset += count;
    return static_cast<int>(count);
  }
  bool connected() { return response->stalled || offset < response->body.size(); }
  void end() {}
};
void reportHttpSpeaker(const String& id, bool ok, const char* reason) {
  assert(id == "42");
  succeeded = ok;
  failure = reason;
  throw Stop{};
}

// INSERT FIRMWARE

std::vector<uint8_t> wav() {
  std::vector<uint8_t> bytes(684);
  auto u32 = [&](int at, uint32_t value) {
    for (int i = 0; i < 4; ++i) bytes[at + i] = uint8_t(value >> (8*i));
  };
  std::memcpy(bytes.data(), "RIFF", 4);
  u32(4, 676);
  std::memcpy(bytes.data() + 8, "WAVEfmt ", 8);
  u32(16, 16); bytes[20] = 1; bytes[22] = 1;
  u32(24, 16000); u32(28, 32000); bytes[32] = 2; bytes[34] = 16;
  std::memcpy(bytes.data() + 36, "data", 4); u32(40, 640);
  return bytes;
}
void run() {
  try { playHttpSpeaker(nullptr); assert(false); } catch (const Stop&) {}
  assert(allocations == 0 && !speakerPlaying);
}
void reset() {
  responses.clear(); requests.clear();
  tick = getCount = allocations = writes = 0;
  failAllocation = failI2s = shortWrite = succeeded = speakerPlaying = false;
  playedAudioPackets = droppedSpeakerPackets = 0;
  failure.clear();
}
int main() {
  for (const char* command : {"VIDEO:ON", "VIDEO:PREVIEW", "VIDEO:FAST"}) {
    videoEnabled = false;
    onCameraEvent(WStype_TEXT, (uint8_t*)command, std::strlen(command));
    assert(videoEnabled);
    onCameraEvent(WStype_TEXT, (uint8_t*)"VIDEO:OFF", 9);
    assert(!videoEnabled);
  }
  const auto complete = wav();
  const std::vector<uint8_t> truncated(complete.begin(), complete.begin() + 48);
  reset(); responses = {{200, truncated}, {200, complete}}; run();
  assert(succeeded && getCount == 2 && writes == 1 && requests[0] == requests[1]);
  assert(requests[0].find("&download=") != std::string::npos);
  reset(); responses = {{-1, {}}, {200, complete}}; run();
  assert(succeeded && getCount == 2 && writes == 1 && requests[0] == requests[1]);
  reset(); responses = {{200, truncated}, {200, truncated}, {200, truncated}}; run();
  assert(!succeeded && getCount == 3 && writes == 0 && failure == "download_timeout");
  reset(); responses = {{200, complete}}; failAllocation = true; run();
  assert(!succeeded && getCount == 1 && writes == 0 && failure == "allocation_failed");
  reset(); responses = {{200, complete}}; failI2s = true; run();
  assert(!succeeded && getCount == 1 && writes == 1 && failure == "i2s_write_failed");
  reset(); responses = {{200, complete}}; shortWrite = true; run();
  assert(!succeeded && getCount == 1 && writes == 1 && failure == "i2s_write_failed");
  reset(); auto invalid = complete; invalid[0] = 0; responses = {{200, invalid}}; run();
  assert(!succeeded && getCount == 1 && writes == 0 && failure == "invalid_wav");
  reset(); responses = {{200, complete, true}, {200, complete}}; run();
  assert(succeeded && getCount == 2 && tick > 3000 && tick < 4000);
}
