from pathlib import Path


def test_job_detail_event_rows_carry_ids_for_live_dedupe() -> None:
    template = Path("services/templates/job_detail.html").read_text()
    app_js = Path("services/static/app.js").read_text()

    assert 'data-event-id="{{ ev.event_id }}"' in template
    assert "seedEventCursorFromDom()" in app_js
    assert "_seenEventIds.has(eventId)" in app_js
