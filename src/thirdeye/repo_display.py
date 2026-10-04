from __future__ import annotations

from typing import Any

from thirdeye.repo_object_finding import FindingMode, FindingResult


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def finding_preview(live: Any, analyzed: Any | None, age: float | None) -> Any:
    """Keep old inference overlays on their original image, beside the live view."""
    import cv2
    import numpy as np

    height, width = live.shape[:2]
    sidebar_width = max(240, width // 2)
    canvas = np.zeros((height + 36, width + sidebar_width, 3), dtype=live.dtype)
    canvas[36:, :width] = live
    cv2.putText(canvas, "LIVE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 1)
    label = "ANALYSING..." if analyzed is None else f"ANALYSIS: {max(0.0, age):.1f}s ago"
    cv2.putText(canvas, label, (width + 8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    if analyzed is not None:
        scale = min(sidebar_width / analyzed.shape[1], height / analyzed.shape[0])
        small = cv2.resize(analyzed, (max(1, round(analyzed.shape[1] * scale)), max(1, round(analyzed.shape[0] * scale))))
        canvas[36:36 + small.shape[0], width:width + small.shape[1]] = small
    return canvas


def annotate_repo_finding(frame: Any, result: FindingResult, target: str) -> Any:
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    detection = result.detection
    if detection is not None and len(detection.polygon) >= 3:
        contour = np.asarray(
            [(point.x, point.y) for point in detection.polygon],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        color = (0, 255, 127) if result.contact_ratio > 0.10 else (0, 255, 255)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [contour], color)
        alpha = result.flash_alpha if result.mode is FindingMode.FLASH else 0.25
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.polylines(frame, [contour], True, color, 5, cv2.LINE_AA)
        cv2.circle(frame, (int(detection.center_x), int(detection.center_y)), 8, (0, 255, 0), 2)

    hand = result.hand
    if hand is not None:
        points = [(int(point.x), int(point.y)) for point in hand.landmarks]
        hull = cv2.convexHull(np.asarray(points, dtype=np.int32))
        cv2.polylines(frame, [hull], True, (255, 255, 255), 1, cv2.LINE_AA)
        for first, second in HAND_CONNECTIONS:
            cv2.line(frame, points[first], points[second], (0, 255, 255), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 2, (0, 255, 255), -1)

    if detection is not None and result.mode is FindingMode.CENTER_GUIDE:
        center = (width // 2, height // 2)
        object_center = (int(detection.center_x), int(detection.center_y))
        cv2.drawMarker(frame, center, (255, 255, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.line(frame, object_center, center, (255, 255, 0), 2, cv2.LINE_AA)
    elif detection is not None and hand is not None and result.mode is FindingMode.TRACK:
        center = (
            int(sum(point.x for point in hand.landmarks) / len(hand.landmarks)),
            int(sum(point.y for point in hand.landmarks) / len(hand.landmarks)),
        )
        cv2.line(
            frame,
            center,
            (int(detection.center_x), int(detection.center_y)),
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(frame, (0, 0), (width, 66), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"{result.mode.value}: {result.guidance.value}",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        target,
        (14, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if result.mode is FindingMode.TRACK:
        ratio = "--" if result.range_ratio is None else f"{result.range_ratio:.2f}"
        cv2.putText(
            frame,
            f"contact {result.contact_ratio:.0%}  grasp {result.grasp_score:.2f}  ratio {ratio}",
            (14, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return frame
