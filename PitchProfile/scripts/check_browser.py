"""Exercise the local UI against disposable copies of the supplied reference data.

Run with the project's virtual-environment Python. Requires Playwright and an
installed Google Chrome. Reports, screenshots and server logs go to .runtime.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_profiler import storage as S
SKILLCORNER = "skillcorner-2017461"
SOCCERNET = "soccernet-sngs-060"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check(report, name, **details):
    report["checks"].append({"name": name, "passed": True, **details})
    print(f"PASS {name}", flush=True)


def select_source(page, identifier):
    page.locator("#dataset-select").select_option(identifier)
    page.wait_for_function("id => state.manifest?.id === id && state.pid !== null && state.player?.player_id === state.pid", arg=identifier)
    # History and the independent-review form finish loading after the profile.
    page.wait_for_load_state("networkidle")
    page.wait_for_function("() => document.querySelector('#history-note').textContent.length > 0")


def open_view(page, tab):
    primary = page.locator(f'nav button[data-tab="{tab}"]')
    if primary.count():
        primary.click()
    else:
        if not page.locator('.tools-menu').evaluate('(el) => el.open'):
            page.locator('.tools-menu summary').click()
        page.locator(f'.tools-menu [data-goto="{tab}"]').click()


def run_checks(page, base_url, report, artifacts):
    page.goto(base_url, wait_until="networkidle")
    open_view(page, "overview")
    expect(page.locator("#player-name")).to_contain_text("May")
    expect(page.locator("#position-coverage")).to_have_text("38.6%")
    expect(page.locator("#source-kind")).to_have_text("Provider tracking")
    require(page.locator("#history tbody tr").count() > 0, "May's earlier-match history was not rendered")
    page.screenshot(path=str(artifacts / "profile-desktop.png"), full_page=True)
    check(report, "real_profile_and_history", player=page.locator("#player-name").inner_text(), position_coverage="38.6%")

    open_view(page, "catalogue")
    expect(page.locator('#catalogue-roles .catalogue-role')).to_have_count(42)
    expect(page.locator('#catalogue-count')).to_contain_text('42 of 42 archetypes')
    page.locator('#catalogue-group').select_option('goalkeeper')
    expect(page.locator('#catalogue-roles .catalogue-role')).to_have_count(3)
    page.locator('#catalogue-group').select_option('centre_forward')
    expect(page.locator('#catalogue-roles .catalogue-role')).to_have_count(8)
    page.locator('#catalogue-group').select_option('')
    expect(page.locator('#catalogue-roles .catalogue-role')).to_have_count(42)
    require(page.locator('nav button[data-tab="reviews"]').count() == 0, 'Legacy three-role screen is still exposed')
    check(report, 'all_42_archetypes_visible_and_position_filtered')

    open_view(page, "experiments")
    expect(page.locator("#train-button")).to_be_disabled()
    expect(page.locator("#training-results")).to_contain_text("No archetype classifier has been trained")
    expect(page.locator("#reconstruction")).to_contain_text("does not contain the reconstruction experiment")
    check(report, "experiments_and_training_gate")

    open_view(page, "intervals")
    page.locator('#case-group').select_option('centre_forward')
    page.locator('#case-start').fill('251')
    page.locator('#case-end').fill('1451')
    page.locator('#case-create-button').click()
    expect(page.locator('#case-content')).to_be_visible()
    expect(page.locator('#case-labels input[type=range]')).to_have_count(8)
    expect(page.locator('#case-status')).to_contain_text('20-minute duration requirement met')
    for reviewer, rating in (('interval-browser-one', '75'), ('interval-browser-two', '85')):
        page.locator('#case-reviewer').fill(reviewer)
        with page.expect_response(lambda response: '/reviews?reviewer=' in response.url):
            page.locator('#case-reviewer').press('Tab')
        expect(page.locator('#case-unknown-target_man')).to_be_checked()
        expect(page.locator('#case-label-target_man')).to_be_disabled()
        page.locator('#case-unknown-target_man').uncheck()
        page.locator('#case-label-target_man').evaluate(
            "(el, v) => { el.value = v; el.dispatchEvent(new Event('input', {bubbles: true})); }", rating)
        expect(page.locator('#case-value-target_man')).to_have_text(rating + '%')
        page.locator('#case-evidence').fill('Disposable automated test only; no real football labels.')
        for i, row in enumerate(page.locator('.sequence-row').all()):
            row.locator('.seq-start').fill(str(260 + i * 20))
            row.locator('.seq-end').fill(str(265 + i * 20))
            row.locator('.seq-note').fill('Artificial sequence for UI verification only')
        with page.expect_response(lambda response: response.url.endswith('/review') and response.request.method == 'POST') as saved:
            page.locator('#case-save').click()
        require(saved.value.ok, 'Interval review failed')
        expect(page.locator('#case-assessment')).to_be_hidden()
    # Only target_man is rated, so its normalised mixture share is 100% even though the raw rating was 85%.
    expect(page.locator('#case-mixture')).to_contain_text('Target man')
    expect(page.locator('#case-mixture')).to_contain_text('100.0%')
    page.locator('#case-assess').click()
    expect(page.locator('#case-assessment')).to_contain_text('agreed')
    expect(page.locator('#case-assessment')).to_contain_text('Primary: Target man')
    page.screenshot(path=str(artifacts / 'interval-review.png'), full_page=True)
    check(report, 'all_position_interval_review_and_independence')
    page.set_viewport_size({'width': 390, 'height': 844})
    interval_width = page.evaluate('document.documentElement.scrollWidth')
    require(interval_width <= 391, f'Interval review overflows mobile viewport: {interval_width}')
    page.screenshot(path=str(artifacts / 'interval-mobile.png'), full_page=True)
    page.set_viewport_size({'width': 1440, 'height': 1000})

    open_view(page, "overview")
    page.locator('#players button[data-player="29075"]').click()
    expect(page.locator("#player-name")).to_contain_text("Vergos")
    page.locator("#identity-panel summary").click()
    before_prediction = page.evaluate("() => state.player.prediction.status")
    require(before_prediction == "position_unconfirmed", f"Expected position selection prompt before setting a group, got {before_prediction}")
    page.locator("#identity-group").select_option("wide_attacker")
    with page.expect_response(lambda response: response.url.endswith("/players/29075") and response.request.method == "GET"), \
         page.expect_response(lambda response: response.url.endswith("/identity") and response.request.method == "POST") as saved:
        page.locator("#identity-form button").click()
    require(saved.value.ok, "Identity save failed")
    page.wait_for_function("() => state.pid === '29075' && state.player?.player_id === '29075'")
    expect(page.locator("#player-name")).to_contain_text("Vergos")
    expect(page.locator("#identity-group")).to_have_value("wide_attacker")
    # No v2 model is trained in this disposable environment, so the visible "Awaiting team
    # reference labels" message is identical either way; check state directly instead to
    # confirm setting a position group (no interval case involved) reached the v2 predictor
    # rather than resurfacing the interval-only 20-minute pilot message (insufficient_evidence).
    after = page.evaluate("() => ({taxonomy_version: state.player.taxonomy_version, status: state.player.prediction.status})")
    require(after["taxonomy_version"] and after["status"] == "not_trained",
            f"Setting an archetype position group did not reach the v2 predictor: {after}")
    expect(page.locator("#prediction")).to_contain_text("Awaiting team reference labels")
    check(report, "identity_save_preserves_selected_player_and_sets_archetype_group")

    select_source(page, SOCCERNET)
    expect(page.locator("#source-kind")).to_have_text("Reference annotations")
    page.wait_for_function("() => {const image = document.querySelector('#media-container img'); return image?.complete && image.naturalWidth > 0;}")
    expect(page.locator("#prediction")).to_contain_text("Choose an archetype position group")
    check(report, "reference_image_and_prediction_distinction")

    page.locator("#upload-open").click()
    page.locator('#upload-form input[name="file"]').set_input_files(str(S.EVIDENCE / "publisher_preview.mp4"))
    page.locator('#upload-form input[name="title"]').fill("Disposable GPU browser check")
    page.locator('#upload-form input[name="max_seconds"]').fill("4")
    with page.expect_response(lambda response: response.url.endswith("/api/upload") and response.request.method == "POST") as uploaded:
        page.locator("#upload-form button.primary").click()
    require(uploaded.value.ok, "Video upload failed")
    upload = uploaded.value.json()
    # Inspect the same job polled by the UI; report the actual error promptly.
    deadline = time.monotonic() + 180
    job = None
    while time.monotonic() < deadline:
        response = page.request.get(base_url + "/api/jobs/" + upload["job_id"])
        require(response.ok, "Video-job polling failed")
        job = response.json()
        if job["status"] in ("failed", "complete"):
            break
        page.wait_for_timeout(600)
    require(job and job["status"] == "complete", f"Video processing did not complete: {job}")
    identifier = upload["dataset_id"]
    page.wait_for_function("id => state.manifest?.id === id && state.player !== null", arg=identifier, timeout=15000)
    manifest = page.request.get(base_url + "/api/datasets/" + identifier).json()["manifest"]
    processing = manifest["processing"]
    require(processing["device"] == "cuda:0", f"Expected cuda:0 processing, got {processing}")
    require(processing["overlay_encoder"] == "h264_nvenc", f"Expected NVENC overlay, got {processing}")
    require(bool(processing.get("device_name")), "GPU device name was not recorded")
    page.locator("#media-container video").evaluate("video => {video.muted = true; video.play().catch(() => {});}")
    page.wait_for_function("() => {const video = document.querySelector('#media-container video'); return video && video.readyState >= 2 && !video.error;}", timeout=30000)
    page.locator("#media-container video").evaluate("video => video.pause()")
    video_state = page.locator("#media-container video").evaluate("video => ({readyState:video.readyState,width:video.videoWidth,height:video.videoHeight,duration:video.duration})")
    require(video_state["width"] > 0 and video_state["duration"] > 0, "Browser could not decode the overlay")
    report["processing"] = processing
    report["video"] = video_state
    check(report, "gpu_upload_and_browser_overlay", dataset_id=identifier, tracks=len(manifest["players"]), device=processing["device"], encoder=processing["overlay_encoder"])

    page.locator("#calibration-panel summary").click()
    page.locator("#load-cal-frame").click()
    page.wait_for_function("() => state.cal.image !== null && state.cal.image.naturalWidth > 0")
    page.locator("#cal-image").click(position={"x": 30, "y": 30})
    page.locator("#cal-pitch").click(position={"x": 50, "y": 50})
    page.wait_for_function("() => state.cal.imagePoints.length === 1")
    select_source(page, SKILLCORNER)
    reset = page.evaluate("() => ({image:state.cal.image,images:state.cal.imagePoints.length,pitches:state.cal.pitchPoints.length,pending:state.cal.pending,pixel:[...document.querySelector('#cal-image').getContext('2d').getImageData(0,0,1,1).data]})")
    require(reset["image"] is None and reset["images"] == 0 and reset["pitches"] == 0 and reset["pending"] is None, f"Stale calibration state: {reset}")
    require(reset["pixel"] == [0, 0, 0, 0], "Previous source frame remained on calibration canvas")
    check(report, "calibration_frame_and_source_reset")

    empty_id = "browser-empty-source"
    empty = {"manifest": {"id": empty_id, "title": "Disposable empty source", "source": "Browser fixture",
                           "source_kind": "model_predictions", "note": "No tracked people", "players": [],
                           "duration_seconds": 1, "video": None}, "profiles": []}
    pattern = "**/api/datasets/" + empty_id
    page.route(pattern, lambda route: route.fulfill(json=empty))
    try:
        page.evaluate("id => selectDataset(id)", empty_id)
        expect(page.locator("#player-name")).to_have_text("No tracked people found")
        expect(page.locator("#position-coverage")).to_have_text("—")
        for element in ("history", "event-summary", "manual-events", "player-subtitle", "prediction"):
            require(page.locator("#" + element).inner_text() == "", f"Stale content in {element}")
        for form in ("identity-form", "event-form"):
            require(page.locator("#" + form + " button:enabled").count() == 0, f"Empty source retained an enabled {form}")
        expect(page.locator('#case-content')).to_be_hidden()
        expect(page.locator('#case-create-button')).to_be_disabled()
        select_source(page, SKILLCORNER)
        expect(page.locator("#identity-form button")).to_be_enabled()
    finally:
        page.unroute(pattern)
    check(report, "empty_source_clears_player_and_disables_forms")

    page.set_viewport_size({"width": 390, "height": 844})
    open_view(page, "overview")
    dimensions = page.evaluate("() => ({viewport:innerWidth,document:document.documentElement.scrollWidth,body:document.body.scrollWidth})")
    require(max(dimensions["document"], dimensions["body"]) <= dimensions["viewport"] + 1, f"Mobile page overflows: {dimensions}")
    page.screenshot(path=str(artifacts / "profile-mobile.png"), full_page=True)
    check(report, "mobile_390px_without_overflow", **dimensions)


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    artifacts = ROOT / ".runtime" / ("browser-" + stamp)
    artifacts.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "started": datetime.now(timezone.utc).isoformat(),
              "checks": [], "page_errors": [], "http_500_errors": [],
              "artifacts": str(artifacts), "data_policy": "All mutations use disposable copies; supplied data is unchanged."}
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="pitchprofile-browser-") as temporary:
            disposable = Path(temporary)
            for dataset in (SKILLCORNER, SOCCERNET):
                shutil.copytree(S.DATA / "datasets" / dataset, disposable / "datasets" / dataset)
            shutil.copy2(S.DATA / "player_history.csv", disposable / "player_history.csv")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            base_url = f"http://127.0.0.1:{port}"
            environment = os.environ.copy()
            environment.update(PITCHPROFILE_DATA=str(disposable), PITCHPROFILE_DEVICE="cuda:0", PYTHONUNBUFFERED="1")
            with (artifacts / "server.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, "run.py", "--host", "127.0.0.1", "--port", str(port)],
                    cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                try:
                    deadline = time.monotonic() + 90
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError("The disposable server stopped during startup; inspect server.log")
                        try:
                            with urlopen(base_url + "/api/status", timeout=1) as response:
                                report["initial_status"] = json.load(response)
                            break
                        except (URLError, TimeoutError, OSError):
                            time.sleep(.2)
                    else:
                        raise TimeoutError("The disposable server was not ready within 90 seconds")
                    with sync_playwright() as playwright:
                        browser = playwright.chromium.launch(channel="chrome", headless=True)
                        try:
                            page = browser.new_page(viewport={"width": 1440, "height": 1000})
                            page.set_default_timeout(15000)
                            page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                            page.on("response", lambda response: report["http_500_errors"].append({"url": response.url, "status": response.status}) if response.status >= 500 else None)
                            try:
                                run_checks(page, base_url, report, artifacts)
                                require(not report["page_errors"], f"JavaScript errors: {report['page_errors']}")
                                require(not report["http_500_errors"], f"Server errors: {report['http_500_errors']}")
                                check(report, "no_javascript_or_http_server_errors")
                            except Exception:
                                page.screenshot(path=str(artifacts / "failure.png"), full_page=True)
                                raise
                        finally:
                            browser.close()
                finally:
                    if process.poll() is None:
                        if os.name == "nt":
                            # Windows venv python.exe can be a launcher whose child
                            # owns SQLite/video handles. Stop only this owned tree.
                            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                           capture_output=True, timeout=15, check=False)
                        else:
                            process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
            server_log = (artifacts / "server.log").read_text(encoding="utf-8")
            require("Traceback (most recent call last)" not in server_log, "Server logged a traceback; inspect server.log")
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        report["finished"] = datetime.now(timezone.utc).isoformat()
        (artifacts / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report: {artifacts / 'report.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
