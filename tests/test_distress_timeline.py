import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from emotion_assistant.regulation_widgets import DistressTimeline


def test_present_missing_zero_and_stale_delayed_ratings():
    app = QApplication.instance() or QApplication([])
    chart = DistressTimeline()
    chart.set_record({"status": "completed", "pre": 8, "post": 0,
                      "followup": {"status": "completed", "distress": 4}})
    assert [bar.value() for bar, _ in chart.rows] == [8, 0, 4]
    assert all(not bar.isHidden() for bar, _ in chart.rows)
    chart.set_record({"status": "completed", "pre": 5, "post": None,
                      "followup": {"status": "pending", "distress": 4}})
    assert [label.text() for _, label in chart.rows] == ["5 / 10", "未填写", "待填写"]
    assert [bar.isHidden() for bar, _ in chart.rows] == [False, True, True]
    chart.set_record({"status": "cancelled", "pre": True, "post": 12})
    assert all(bar.isHidden() for bar, _ in chart.rows)
    chart.set_record(None)
    assert chart.isHidden()
    chart.close()
    app.processEvents()
