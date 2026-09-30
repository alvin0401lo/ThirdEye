#include <cassert>
#include <limits>

static void warmup(FallDetector& detector, uint32_t start = 0) {
  for (uint32_t t = 0; t <= 2100; t += 10)
    assert(!detector.update(start + t, 0, 0, 1, 0, 0, 0));
  assert(detector.state == FallDetector::MONITORING);
}

int main() {
  FallDetector detector;
  warmup(detector);
  for (uint32_t t = 2110; t < 2500; t += 10)
    assert(!detector.update(t, 0.7f, 0, 0.7f, 0, 15, 0));
  assert(detector.state == FallDetector::MONITORING);

  // Impact followed by persistent sideways posture triggers once.
  FallDetector fall;
  warmup(fall);
  assert(!fall.update(2110, 3, 0, 0, 0, 300, 0));
  int events = 0;
  for (uint32_t t = 2120; t < 6000; t += 10)
    events += fall.update(t, 1, 0, 0, 0, 0, 0);
  assert(events == 1 && fall.state == FallDetector::COOLDOWN);

  // A quick head turn or impact returning upright must not trigger.
  FallDetector turn;
  warmup(turn);
  assert(!turn.update(2110, 0, 0, 1, 0, 300, 0));
  for (uint32_t t = 2120; t <= 7500; t += 10)
    assert(!turn.update(t, 0, 0, 1, 0, 0, 0));

  // Missing samples invalidate the candidate rather than inventing stillness.
  FallDetector gap;
  warmup(gap);
  assert(!gap.update(2110, 3, 0, 0, 0, 300, 0));
  assert(!gap.update(3000, 1, 0, 0, 0, 0, 0));
  assert(gap.state == FallDetector::WARMUP);
  assert(!gap.update(3010, std::numeric_limits<float>::quiet_NaN(), 0, 0, 0, 0, 0));

  FallDetector rollover;
  warmup(rollover, UINT32_MAX - 1000);
}
