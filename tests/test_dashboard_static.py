from pathlib import Path


def test_job_detail_event_rows_carry_ids_for_live_dedupe() -> None:
    template = Path("services/templates/job_detail.html").read_text()
    app_js = Path("services/static/app.js").read_text()

    assert 'data-event-id="{{ ev.event_id }}"' in template
    assert "seedEventCursorFromDom()" in app_js
    assert "_seenEventIds.has(eventId)" in app_js


def test_new_job_routes_wan_animate_to_dedicated_studio() -> None:
    template = Path("services/templates/job_new.html").read_text()
    assert 'value="wan_animate"' in template
    assert 'href="/animate/new"' in template
    assert 'id="nj-wan-animate-mode"' not in template
    assert "/api/uploads/wan-animate-driver" not in template


def test_animate_studio_exposes_direct_workflow_controls() -> None:
    base = Path("services/templates/base.html").read_text()
    template = Path("services/templates/animate_new.html").read_text()
    script = Path("services/static/animate.js").read_text()

    assert 'href="/animate/new"' in base
    assert 'name="animate_mode" value="animate"' in template
    assert 'name="animate_mode" value="replace"' in template
    assert 'data-driver-tab="url"' in template
    assert 'data-driver-tab="upload"' in template
    assert 'data-driver-tab="server"' in template
    assert 'name="look_source" value="auto_lora"' in template
    assert 'name="look_source" value="styled_lora"' in template
    assert 'name="look_source" value="exact_image"' in template
    assert "Design complete look" in template
    assert "What should FLUX.2 change?" in template
    assert template.count('name="style_change_target"') == 7
    assert 'name="style_change_target" value="clothing"' in template
    assert 'name="style_change_target" value="jewelry"' in template
    assert 'name="style_change_target" value="bags"' in template
    assert 'name="style_change_target" value="footwear"' in template
    assert 'name="style_change_target" value="makeup"' in template
    assert 'id="wardrobe-jewelry"' in template
    assert 'id="wardrobe-bags"' in template
    assert 'id="wardrobe-makeup"' in template
    assert 'id="wardrobe-hair"' in template
    assert 'name="audio_mode" value="driver"' in template
    assert 'name="audio_mode" value="cast_voice"' in template
    assert 'name="export_mode" value="scale_1080p"' in template
    assert 'name="export_mode" value="vertical_1080x1920"' in template
    assert 'id="animate-target-confirmed"' in template
    assert 'workflow_kind: "wan_animate_direct"' in script
    assert 'target_confirmed: true' in script
    assert 'preserve_aspect: true' in script
    assert 'options.features?.flux2_edit_enabled' in script
    assert 'options.features?.flux2_edit_max_user_references' in script
    assert 'change_targets: selectedStyleTargets()' in script
    assert 'jewelry: listFromCommaInput("wardrobe-jewelry")' in script
    assert 'bags: listFromCommaInput("wardrobe-bags")' in script
    assert 'makeup: $("#wardrobe-makeup").value.trim()' in script
    assert 'negative_constraints: $("#wardrobe-negative").value.trim()' in script
    assert "return selectedStyleTargets().length > 0;" in script
    assert "function completeLookScopeErrors()" in script
    assert "Clothing / dress references require Clothing / dress" in script
    assert "Styling detail references require Jewelry, Bags, Footwear" in script
