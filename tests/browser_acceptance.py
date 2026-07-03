import base64
import json
import os
import socket
import struct
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CDP_HTTP = os.environ.get("MLHIGH_CDP_HTTP", "http://127.0.0.1:9228")


class CDP:
    def __init__(self, websocket_url: str):
        parsed = urllib.parse.urlparse(websocket_url)
        self.sock = socket.create_connection((parsed.hostname, parsed.port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET {parsed.path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Origin: http://127.0.0.1\r\n\r\n"
        )
        self.sock.sendall(request.encode())
        response = b""
        while b"\r\n\r\n" not in response:
            response += self.sock.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(response.decode(errors="replace"))
        self.next_id = 0
        self.events = []

    def _read_exact(self, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Chrome DevTools socket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _send_frame(self, payload: bytes, opcode: int = 1) -> None:
        mask = os.urandom(4)
        length = len(payload)
        header = bytearray([0x80 | opcode])
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        header.extend(mask)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def _recv_frame(self) -> tuple[int, bytes]:
        first, second = self._read_exact(2)
        opcode = first & 0x0F
        length = second & 0x7F
        masked = bool(second & 0x80)
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        mask = self._read_exact(4) if masked else b""
        payload = self._read_exact(length)
        if masked:
            payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        if opcode == 9:
            self._send_frame(payload, opcode=10)
            return self._recv_frame()
        if opcode == 8:
            raise ConnectionError("Chrome DevTools socket closed")
        return opcode, payload

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        message_id = self.next_id
        self._send_frame(json.dumps({
            "id": message_id,
            "method": method,
            "params": params or {},
        }).encode())
        while True:
            opcode, payload = self._recv_frame()
            if opcode != 1:
                continue
            message = json.loads(payload)
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})
            self.events.append(message)

    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", {
            "expression": expression,
            "awaitPromise": True,
            "returnByValue": True,
            "userGesture": True,
        })
        if result.get("exceptionDetails"):
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")


def connect() -> CDP:
    targets = json.load(urllib.request.urlopen(f"{CDP_HTTP}/json", timeout=10))
    page = next(item for item in targets if item.get("type") == "page")
    cdp = CDP(page["webSocketDebuggerUrl"])
    cdp.call("Runtime.enable")
    cdp.call("Page.enable")
    cdp.call("Network.enable")
    cdp.call("Log.enable")
    cdp.call("Emulation.setDeviceMetricsOverride", {
        "width": 1600,
        "height": 1000,
        "deviceScaleFactor": 1,
        "mobile": False,
    })
    cdp.call("Page.reload", {"ignoreCache": True})
    wait_for(cdp, "document.readyState === 'complete' && typeof STATE === 'object'", 20, "page load")
    return cdp


def wait_for(cdp: CDP, expression: str, timeout: float, label: str) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            last = cdp.evaluate(f"Boolean({expression})")
            if last:
                return
        except Exception as exc:
            last = str(exc)
        time.sleep(0.15)
    raise AssertionError(f"Timed out waiting for {label}: {last}")


def screenshot(cdp: CDP, filename: str) -> None:
    result = cdp.call("Page.captureScreenshot", {"format": "png", "fromSurface": True})
    (ROOT / filename).write_bytes(base64.b64decode(result["data"]))


def dispatch_control(cdp: CDP, element_id: str, value) -> None:
    encoded = json.dumps(str(value))
    cdp.evaluate(
        f"""(() => {{
          const input = document.getElementById({json.dumps(element_id)});
          if (!input) throw new Error('Missing control: {element_id}');
          input.value = {encoded};
          input.dispatchEvent(new Event('input', {{ bubbles: true }}));
          input.dispatchEvent(new Event('change', {{ bubbles: true }}));
          return true;
        }})()"""
    )


def set_synthetic_chart(cdp: CDP, title: str, traces: list[dict]) -> None:
    payload = json.dumps({"data": traces, "layout": {"title": title}}, ensure_ascii=False)
    cdp.evaluate(
        f"""(() => {{
          STATE.chartTitle = '';
          STATE.activeChartIndex = 0;
          STATE.currentChartBundle = [{{
            title: {json.dumps(title, ensure_ascii=False)},
            plotly: {json.dumps(payload, ensure_ascii=False)}
          }}];
          renderAllCharts(STATE.currentChartBundle);
          renderAppearanceControls();
          return true;
        }})()"""
    )
    wait_for(cdp, "document.querySelector('.chart-plot')?._fullLayout", 10, title)


def test_novice_workflow(cdp: CDP) -> None:
    cdp.evaluate("document.getElementById('loadExampleBtn').click()")
    wait_for(cdp, "STATE.columns.length === 57 && document.querySelectorAll('#visual_pool .var-chip-item').length === 57", 20, "example data")
    cdp.evaluate("document.querySelector('.nav-step[data-tab=\"variables\"]').click()")

    layout = cdp.evaluate(
        """(() => {
          const list = document.getElementById('visual_pool');
          const items = [...list.querySelectorAll('.var-chip-item')].slice(0, 8);
          const rects = items.map(item => item.getBoundingClientRect());
          const card = list.closest('.var-transfer-card').getBoundingClientRect();
          const controls = document.getElementById('roleControls').getBoundingClientRect();
          const grid = document.querySelector('.ml-role-grid').getBoundingClientRect();
          return {
            count: items.length,
            flexDirection: getComputedStyle(list).flexDirection,
            overflowY: getComputedStyle(list).overflowY,
            scrollable: list.scrollHeight > list.clientHeight,
            sameX: rects.every(rect => Math.abs(rect.x - rects[0].x) < 1),
            orderedY: rects.every((rect, index) => index === 0 || rect.y > rects[index - 1].y),
            cardHeight: card.height,
            controlsHeight: controls.height,
            gridHeight: grid.height,
            bottomGap: Math.abs(card.bottom - grid.bottom),
            firstVariable: items[0]?.dataset.var,
            lastVariable: document.querySelector('#visual_pool .var-chip-item:last-child')?.dataset.var,
          };
        })()"""
    )
    assert layout["flexDirection"] == "column", layout
    assert layout["overflowY"] == "auto", layout
    assert layout["scrollable"] and layout["sameX"] and layout["orderedY"], layout
    assert layout["cardHeight"] > 650, layout
    assert layout["gridHeight"] > 650 and layout["bottomGap"] < 3, layout
    assert layout["cardHeight"] >= layout["controlsHeight"] - 90, layout
    assert layout["firstVariable"] == "subject_id" and layout["lastVariable"] == "x6", layout
    assert cdp.evaluate("document.querySelector('.nav-step.active')?.dataset.tab") == "variables"
    screenshot(cdp, "qa_variable_layout.png")

    cdp.evaluate(
        """(() => {
          ['x1', 'x2', 'x3', 'x4'].forEach(name =>
            document.querySelector(`#visual_pool .var-chip-item[data-var="${name}"]`).click()
          );
          document.querySelector('[data-action="add"][data-target="research_vars"]').click();
          document.querySelector('#visual_pool .var-chip-item[data-var="label"]').click();
          document.querySelector('[data-action="add"][data-target="outcome_vars"]').click();
          document.querySelector('.nav-step[data-tab="method"]').click();
          return true;
        })()"""
    )
    wait_for(cdp, "STATE.methodAvailabilityServer?.ml_lr && STATE.validationKey", 20, "method validation")
    availability = cdp.evaluate(
        """(() => ({
          lr: STATE.methodAvailabilityServer.ml_lr,
          psm: STATE.methodAvailabilityServer.propensity_score,
          psmCardDisabled: document.querySelector('[data-method="propensity_score"]').classList.contains('method-disabled')
        }))()"""
    )
    assert not availability["psm"]["available"] and availability["psmCardDisabled"], availability
    cdp.evaluate("document.querySelector('#methodCatTabs [data-cat=\"ml_models\"]').click()")
    wait_for(cdp, "document.querySelector('[data-method=\"ml_lr\"]')", 5, "machine learning method cards")
    availability["lrCardDisabled"] = cdp.evaluate(
        "document.querySelector('[data-method=\"ml_lr\"]').classList.contains('method-disabled')"
    )
    assert availability["lr"]["available"] and not availability["lrCardDisabled"], availability

    event_start = len(cdp.events)
    cdp.evaluate(
        """(() => {
          document.querySelector('[data-method="ml_lr"]').click();
          document.querySelector('.nav-step[data-tab="run"]').click();
          document.getElementById('generateBtn').click();
          return true;
        })()"""
    )
    wait_for(cdp, "STATE.currentChartBundle?.length > 0 && document.querySelector('#result-view-chart.active .chart-plot')?._fullLayout", 90, "analysis result")
    products = cdp.evaluate(
        """(() => ({
          charts: STATE.currentChartBundle.length,
          tables: document.querySelectorAll('#resultTablesContainer table').length,
          discussion: document.getElementById('discussionContainer').innerText.length,
          resultActive: document.getElementById('tab-result').classList.contains('active')
        }))()"""
    )
    assert products["charts"] > 0 and products["tables"] > 0, products
    assert products["discussion"] > 200 and products["resultActive"], products
    analyze_requests = [
        event for event in cdp.events[event_start:]
        if event.get("method") == "Network.requestWillBeSent"
        and event.get("params", {}).get("request", {}).get("method") == "POST"
        and event.get("params", {}).get("request", {}).get("url", "").endswith("/api/analyze")
    ]
    assert len(analyze_requests) == 1, f"Expected one analysis request, got {len(analyze_requests)}"
    errors = cdp.evaluate("[...document.querySelectorAll('.toast.error')].map(node => node.innerText)")
    assert not errors, errors


def test_ldsc_workflow(cdp: CDP) -> None:
    cdp.evaluate(
        """(async () => {
          const assigned = collectRoleVars();
          for (const role of ['research_vars', 'covar_vars', 'outcome_vars']) {
            if (assigned[role]?.length) moveRoleVariables([...assigned[role]], role, 'pool');
          }
          moveRoleVariables(['trait'], 'pool', 'research_vars');
          moveRoleVariables(['h2_se'], 'pool', 'covar_vars');
          moveRoleVariables(['h2'], 'pool', 'outcome_vars');
          STATE.validationKey = null;
          await validateMethodsWithBackendV10();
          document.querySelector('.nav-step[data-tab="method"]').click();
          document.querySelector('#methodCatTabs [data-cat="advanced_stats"]').click();
          return STATE.methodAvailabilityServer?.ldsc;
        })()"""
    )
    wait_for(cdp, "STATE.methodAvailabilityServer?.ldsc?.available && document.querySelector('[data-method=\"ldsc\"]')", 20, "LDSC availability")
    cdp.evaluate(
        """(() => {
          STATE.currentChartBundle = [];
          document.querySelector('[data-method="ldsc"]').click();
          document.querySelector('.nav-step[data-tab="run"]').click();
          document.getElementById('generateBtn').click();
          return true;
        })()"""
    )
    wait_for(
        cdp,
        "STATE.currentChartBundle?.[0]?.title === '性状间遗传相关性热图' && document.querySelector('.chart-plot')?._fullLayout",
        90,
        "LDSC heatmap",
    )
    dispatch_control(cdp, "chartWidthInput", 640)
    dispatch_control(cdp, "chartHeightInput", 520)
    wait_for(cdp, "document.querySelector('.chart-plot')?.dataset.chartWidth === '640'", 10, "LDSC chart dimensions")
    output = cdp.evaluate(
        """(() => {
          const trace = STATE.currentPlotlyData[0];
          const mount = document.querySelector('.chart-plot').getBoundingClientRect();
          const preview = document.getElementById('chartPreviewContainer').getBoundingClientRect();
          activateResultTabV11('discussion');
          const discussion = document.getElementById('discussionContainer');
          const content = discussion.querySelector('.discussion-content');
          const computed = getComputedStyle(discussion);
          return {
            titles: STATE.currentChartBundle.map(chart => chart.title),
            type: trace.type,
            x: trace.x,
            y: trace.y,
            z: trace.z,
            yRange: document.querySelector('.chart-plot')._fullLayout.yaxis.range,
            centered: Math.abs((mount.left + mount.width / 2) - (preview.left + preview.width / 2)),
            discussionLength: discussion.innerText.length,
            discussionText: discussion.innerText,
            scrollbarWidth: computed.scrollbarWidth,
            scrollbarGap: discussion.offsetWidth - discussion.clientWidth,
            contentWidth: content?.getBoundingClientRect().width,
            containerWidth: discussion.getBoundingClientRect().width,
            toastContainerDisplay: getComputedStyle(document.getElementById('toastContainer')).display,
          };
        })()"""
    )
    assert output["titles"] == [
        "性状间遗传相关性热图",
        "各性状遗传力森林图",
        "重点遗传相关性状组合",
        "遗传力与平均共病遗传关联",
    ], output
    assert output["type"] == "heatmap" and output["x"] == output["y"], output
    assert len(output["z"]) == len(output["x"]) >= 2, output
    assert all(len(row) == len(output["x"]) for row in output["z"]), output
    assert output["yRange"][0] > output["yRange"][1], output
    assert output["centered"] < 3, output
    assert output["discussionLength"] > 1500, output
    assert "分析概要" in output["discussionText"] and "图表阅读顺序" in output["discussionText"], output
    assert output["scrollbarWidth"] == "none" and output["scrollbarGap"] <= 3, output
    assert output["contentWidth"] <= output["containerWidth"], output
    assert output["toastContainerDisplay"] == "none", output
    activate = cdp.evaluate("activateResultTabV11('chart'); true")
    assert activate
    screenshot(cdp, "qa_ldsc_heatmap.png")


def test_independent_dimensions(cdp: CDP) -> None:
    set_synthetic_chart(cdp, "尺寸独立性验收", [{
        "type": "scatter",
        "mode": "lines+markers",
        "name": "系列 A",
        "x": [1, 2, 3, 4],
        "y": [2, 4, 3, 6],
    }])
    dispatch_control(cdp, "chartWidthInput", 600)
    dispatch_control(cdp, "chartHeightInput", 500)
    wait_for(cdp, "document.querySelector('.chart-plot')?.dataset.chartWidth === '600' && document.querySelector('.chart-plot')?.dataset.chartHeight === '500'", 10, "baseline dimensions")

    def metrics():
        return cdp.evaluate(
            """(() => {
              const mount = document.querySelector('.chart-plot');
              const svg = mount.querySelector('.main-svg');
              const title = mount.querySelector('.gtitle');
              const mr = mount.getBoundingClientRect();
              const sr = svg.getBoundingClientRect();
              const pr = document.getElementById('chartPreviewContainer').getBoundingClientRect();
              return {
                mount: {x: mr.x, y: mr.y, width: mr.width, height: mr.height},
                preview: {x: pr.x, y: pr.y, width: pr.width, height: pr.height},
                svg: {width: sr.width, height: sr.height},
                titleX: title?.getBoundingClientRect().x,
                titleOffsetX: title?.getBoundingClientRect().x - mr.x,
                titleFont: title ? getComputedStyle(title).fontSize : null,
                layoutWidth: mount._fullLayout.width,
                layoutHeight: mount._fullLayout.height,
              };
            })()"""
        )

    baseline = metrics()
    assert abs((baseline["mount"]["x"] + baseline["mount"]["width"] / 2) - (baseline["preview"]["x"] + baseline["preview"]["width"] / 2)) < 3, baseline
    dispatch_control(cdp, "chartWidthInput", 700)
    wait_for(cdp, "document.querySelector('.chart-plot')?.dataset.chartWidth === '700'", 10, "width change")
    wider = metrics()
    assert wider["mount"]["width"] == 700 and wider["layoutWidth"] == 700, wider
    assert wider["mount"]["height"] == baseline["mount"]["height"], (baseline, wider)
    assert wider["svg"]["height"] == baseline["svg"]["height"], (baseline, wider)
    assert abs((wider["mount"]["x"] + wider["mount"]["width"] / 2) - (wider["preview"]["x"] + wider["preview"]["width"] / 2)) < 3, wider
    assert abs(wider["titleOffsetX"] - baseline["titleOffsetX"]) < 1, (baseline, wider)
    assert wider["titleFont"] == baseline["titleFont"], (baseline, wider)

    dispatch_control(cdp, "chartHeightInput", 620)
    wait_for(cdp, "document.querySelector('.chart-plot')?.dataset.chartHeight === '620'", 10, "height change")
    taller = metrics()
    assert taller["mount"]["height"] == 620 and taller["layoutHeight"] == 620, taller
    assert taller["mount"]["width"] == wider["mount"]["width"], (wider, taller)
    assert taller["svg"]["width"] == wider["svg"]["width"], (wider, taller)
    assert abs(taller["mount"]["x"] - wider["mount"]["x"]) < 1, (wider, taller)
    assert abs(taller["titleOffsetX"] - wider["titleOffsetX"]) < 1, (wider, taller)
    assert taller["titleFont"] == wider["titleFont"], (wider, taller)


def test_type_specific_controls(cdp: CDP) -> None:
    set_synthetic_chart(cdp, "柱状图验收", [{
        "type": "bar",
        "name": "发生率",
        "x": ["A", "B", "C"],
        "y": [12, 18, 9],
    }])
    bar = cdp.evaluate(
        """(() => ({
          controls: ['barWidthInput', 'barStyleSelect'].every(id => !!document.getElementById(id)),
          picker: [...document.querySelectorAll('.category-color-input')].map(x => x.value),
          rendered: STATE.currentPlotlyData[0].marker.color
        }))()"""
    )
    assert bar["controls"], bar
    assert [color.lower() for color in bar["picker"]] == [color.lower() for color in bar["rendered"]], bar
    bar_axes = cdp.evaluate(
        """(() => {
          const layout = document.querySelector('.chart-plot')._fullLayout;
          return {
            xSide: layout.xaxis.side,
            ySide: layout.yaxis.side,
            namedArrows: (layout.annotations || []).filter(item => String(item.name || '').startsWith('v27_axis_')).length
          };
        })()"""
    )
    assert bar_axes == {"xSide": "bottom", "ySide": "left", "namedArrows": 0}, bar_axes
    dispatch_control(cdp, "barWidthInput", 0.8)
    dispatch_control(cdp, "barStyleSelect", "outline")
    wait_for(cdp, "STATE.currentPlotlyData?.[0]?.width === 0.8", 10, "bar controls")
    bar_after = cdp.evaluate("({width: STATE.currentPlotlyData[0].width, fill: STATE.currentPlotlyData[0].marker.color, edge: STATE.currentPlotlyData[0].marker.line.width})")
    assert bar_after["width"] == 0.8 and bar_after["edge"] == 2
    assert all(color == "rgba(255,255,255,0)" for color in bar_after["fill"])

    set_synthetic_chart(cdp, "折线图验收", [{
        "type": "scatter",
        "mode": "lines+markers",
        "name": "趋势",
        "x": [1, 2, 3],
        "y": [2, 5, 4],
    }])
    for control in ["lineWidthInput", "lineDashSelect", "lineShapeSelect", "markerSizeInput", "markerShapeSelect"]:
        assert cdp.evaluate(f"Boolean(document.getElementById({json.dumps(control)}))"), control
    dispatch_control(cdp, "lineWidthInput", 5)
    dispatch_control(cdp, "lineDashSelect", "dash")
    dispatch_control(cdp, "lineShapeSelect", "spline")
    dispatch_control(cdp, "markerSizeInput", 12)
    wait_for(cdp, "STATE.currentPlotlyData?.[0]?.line?.width === 5 && STATE.currentPlotlyData?.[0]?.marker?.size === 12", 10, "line controls")
    line = cdp.evaluate("({line: STATE.currentPlotlyData[0].line, marker: STATE.currentPlotlyData[0].marker})")
    assert line["line"]["dash"] == "dash" and line["line"]["shape"] == "spline", line

    set_synthetic_chart(cdp, "横向柱状图验收", [{
        "type": "bar",
        "orientation": "h",
        "name": "效应量",
        "x": [0.4, 0.8, 1.2],
        "y": ["变量一", "变量二", "变量三"],
    }])
    horizontal = cdp.evaluate(
        """(() => ({
          labels: STATE.currentPlotlyData[0].y,
          yRange: document.querySelector('.chart-plot')._fullLayout.yaxis.range,
          xSide: document.querySelector('.chart-plot')._fullLayout.xaxis.side,
          ySide: document.querySelector('.chart-plot')._fullLayout.yaxis.side,
        }))()"""
    )
    assert horizontal["labels"] == ["变量一", "变量二", "变量三"], horizontal
    assert horizontal["yRange"][0] > horizontal["yRange"][1], horizontal
    assert horizontal["xSide"] == "bottom" and horizontal["ySide"] == "left", horizontal

    cases = [
        ("直方图验收", [{"type": "histogram", "x": [1, 1, 2, 2, 3, 4]}], ["histogramBinsInput", "barStyleSelect"]),
        ("箱线图验收", [{"type": "box", "y": [1, 2, 3, 4, 8]}], ["boxPointsSelect", "lineWidthInput"]),
        ("小提琴图验收", [{"type": "violin", "y": [1, 2, 2, 3, 4]}], ["violinPointsSelect", "lineWidthInput"]),
        ("饼图验收", [{"type": "pie", "labels": ["A", "B"], "values": [4, 6]}], ["pieHoleInput"]),
        ("热图验收", [{"type": "heatmap", "x": ["A", "B"], "y": ["A", "B"], "z": [[1, -0.4], [-0.4, 1]]}], ["heatmapOpacityInput"]),
        ("桑基图验收", [{"type": "sankey", "node": {"label": ["A", "B"]}, "link": {"source": [0], "target": [1], "value": [5]}}], ["sankeyNodePadInput", "sankeyNodeThicknessInput"]),
    ]
    for title, traces, controls in cases:
        set_synthetic_chart(cdp, title, traces)
        for control in controls:
            assert cdp.evaluate(f"Boolean(document.getElementById({json.dumps(control)}))"), f"{title}: {control}"
        if traces[0]["type"] == "heatmap":
            heatmap_axis = cdp.evaluate(
                """(() => ({
                  yRange: document.querySelector('.chart-plot')._fullLayout.yaxis.range,
                  xSide: document.querySelector('.chart-plot')._fullLayout.xaxis.side,
                  ySide: document.querySelector('.chart-plot')._fullLayout.yaxis.side,
                }))()"""
            )
            assert heatmap_axis["yRange"][0] > heatmap_axis["yRange"][1], heatmap_axis
            assert heatmap_axis["xSide"] == "bottom" and heatmap_axis["ySide"] == "left", heatmap_axis
            assert not cdp.evaluate("Boolean(document.querySelector('[data-chart-state=\"heatmapColorscale\"], [data-heatmap-var-color]'))")
            heatmap_theme = cdp.evaluate(
                """(() => ({
                  colorscale: STATE.currentPlotlyData[0].colorscale,
                  expected: getActiveTheme().divergentScale,
                  zmid: STATE.currentPlotlyData[0].zmid,
                }))()"""
            )
            assert heatmap_theme["colorscale"] == heatmap_theme["expected"], heatmap_theme
            assert heatmap_theme["zmid"] == 0, heatmap_theme
            dispatch_control(cdp, "heatmapOpacityInput", 0.45)
            wait_for(cdp, "STATE.currentPlotlyData?.[0]?.opacity === 0.45", 10, "heatmap opacity")

    dispatch_control(cdp, "sankeyNodePadInput", 24)
    dispatch_control(cdp, "sankeyNodeThicknessInput", 26)
    wait_for(cdp, "STATE.currentPlotlyData?.[0]?.node?.pad === 24", 10, "sankey controls")
    sankey = cdp.evaluate("STATE.currentPlotlyData[0].node")
    assert sankey["pad"] == 24 and sankey["thickness"] == 26, sankey

    screenshot(cdp, "qa_chart_controls.png")


def test_no_runtime_errors(cdp: CDP) -> None:
    fatal = []
    for event in cdp.events:
        if event.get("method") == "Runtime.exceptionThrown":
            fatal.append(event)
        if event.get("method") == "Log.entryAdded":
            entry = event.get("params", {}).get("entry", {})
            if entry.get("level") == "error":
                fatal.append(event)
    assert not fatal, json.dumps(fatal[:5], ensure_ascii=False)


if __name__ == "__main__":
    browser = connect()
    test_novice_workflow(browser)
    print("  [PASS] Novice example → variables → availability → analysis workflow")
    test_ldsc_workflow(browser)
    print("  [PASS] LDSC workflow, centered heatmap, and friendly discussion")
    test_independent_dimensions(browser)
    print("  [PASS] Centered chart width and height are pixel-independent")
    test_type_specific_controls(browser)
    print("  [PASS] Type-specific controls match rendered chart types and colors")
    test_no_runtime_errors(browser)
    print("  [PASS] No browser runtime errors")
    print("Browser acceptance passed!")
