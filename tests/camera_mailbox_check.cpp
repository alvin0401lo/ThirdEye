#include <cassert>
#include <set>

struct FakeQueue {
  CameraPacket packet;
  bool occupied = false, failSend = false;
};
using QueueHandle_t = FakeQueue*;
constexpr int pdTRUE = 1;
FakeQueue liveQueue, stillQueue;
QueueHandle_t cameraFrames = &liveQueue;
CameraMetrics cameraMetrics;
int cameraMetricsMux = 0;
#define portENTER_CRITICAL(mux) (void)(mux)
#define portEXIT_CRITICAL(mux) (void)(mux)
std::set<uint8_t*> allocated;

void heap_caps_free(uint8_t* data) {
  assert(allocated.erase(data) == 1);
  delete[] data;
}
int xQueueReceive(QueueHandle_t queue, CameraPacket* packet, int) {
  if (!queue->occupied) return 0;
  *packet = queue->packet;
  queue->occupied = false;
  return pdTRUE;
}
int xQueueSend(QueueHandle_t queue, CameraPacket* packet, int) {
  if (queue->occupied || queue->failSend) return 0;
  queue->packet = *packet;
  queue->occupied = true;
  return pdTRUE;
}
CameraPacket makePacket(unsigned long stillId = 0) {
  CameraPacket packet;
  packet.data = new uint8_t[10];
  packet.length = 10;
  packet.stillId = stillId;
  allocated.insert(packet.data);
  return packet;
}

// INSERT CAMERA MAILBOX

int main() {
  CameraPacket first = makePacket();
  assert(publishCameraPacket(&liveQueue, first));
  CameraPacket sending;
  assert(xQueueReceive(&liveQueue, &sending, 0) == pdTRUE);
  CameraPacket old = makePacket(), latest = makePacket();
  assert(publishCameraPacket(&liveQueue, old));
  assert(publishCameraPacket(&liveQueue, latest));
  assert(allocated.count(old.data) == 0);
  assert(allocated.count(sending.data) == 1);
  assert(cameraMetrics.replacements == 1);
  assert(liveQueue.packet.data == latest.data);

  CameraPacket still = makePacket(42);
  assert(publishCameraPacket(&stillQueue, still));
  assert(stillQueue.packet.stillId == 42);
  assert(liveQueue.packet.data == latest.data);
  heap_caps_free(sending.data);
  assert(xQueueReceive(&liveQueue, &sending, 0) == pdTRUE);
  heap_caps_free(sending.data);
  assert(xQueueReceive(&stillQueue, &sending, 0) == pdTRUE);
  heap_caps_free(sending.data);

  liveQueue.failSend = true;
  assert(!publishCameraPacket(&liveQueue, makePacket()));
  assert(allocated.empty());
  CameraPacket failedStill;
  failedStill.stillId = 43;
  assert(publishCameraPacket(&stillQueue, failedStill));
  assert(stillQueue.packet.data == nullptr && stillQueue.packet.stillId == 43);
}
