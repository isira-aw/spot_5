"""The HTTP surface, including the parts that must refuse."""
import pytest

TOKEN = {"X-Admin-Token": "test-token"}


@pytest.fixture
def client(env, fake_market):
    import main
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c


def test_health_reports_every_dependency(client):
    h = client.get("/health").json()
    assert h["ok"] is True and h["mode"] == "PAPER"
    assert h["knowledge_base"]["sections"] >= 10
    assert h["risk_model"]["kind"] in ("heuristic", "logistic", "gbm")
    assert h["kill_switch"] is False


def test_config_never_leaks_a_secret(client):
    cfg = client.get("/config").json()
    assert cfg["admin_token"] is True                 # a boolean, not the token
    assert cfg["llm"]["groq_api_key"] in (True, False)
    assert "***" in cfg["db"]["url"] or "@" not in cfg["db"]["url"]


def test_state_shows_the_book_and_the_rules_as_briefed(client):
    body = client.get("/state").json()
    assert body["mode"] == "PAPER"
    assert body["portfolio"]["cash"] == 10000.0
    assert body["restrictions"]["max_position_pct"] == 25.0
    rules = client.get("/admin/rules").json()
    assert any("Never risk more than" in line for line in rules["as_briefed_to_the_agent"])


def test_admin_endpoints_require_the_token(client):
    assert client.put("/admin/rules", json={"max_position_pct": 10}).status_code == 401
    assert client.post("/admin/kill-switch", json={"enabled": True}).status_code == 401
    bad = client.put("/admin/rules", json={"max_position_pct": 10},
                     headers={"X-Admin-Token": "wrong"})
    assert bad.status_code == 401


def test_rules_can_be_tightened_through_the_api(client):
    r = client.put("/admin/rules", json={"max_position_pct": 8, "min_confidence": 0.7,
                                         "notes": ["No CPI days."], "note": "tighten"},
                   headers=TOKEN)
    assert r.status_code == 200
    effective = r.json()["effective"]
    assert effective["max_position_pct"] == 8.0 and effective["min_confidence"] == 0.7
    assert "No CPI days." in effective["notes"]


def test_an_unknown_action_in_the_rules_is_rejected(client):
    r = client.put("/admin/rules", json={"allowed_actions": ["BUY", "SHORT"]}, headers=TOKEN)
    assert r.status_code == 422


def test_the_kill_switch_is_visible_everywhere_once_engaged(client):
    client.post("/admin/kill-switch", json={"enabled": True, "reason": "drill"},
                headers=TOKEN)
    assert client.get("/health").json()["kill_switch"] is True
    assert client.get("/state").json()["restrictions"]["kill_switch"] is True
    events = client.get("/events", params={"category": "risk"}).json()
    assert any("ENGAGED" in e["message"] for e in events)


def test_publishing_a_knowledge_base_takes_effect_immediately(client):
    before = client.get("/knowledge-base").json()["version"]
    body = {"content": "# KB\n\n## One Rule\n**Tags:** process\n\nDo not lose money.\n",
            "label": "minimal"}
    published = client.post("/admin/knowledge-base", json=body, headers=TOKEN).json()
    assert published["published"]["sections"] == 1
    after = client.get("/knowledge-base").json()
    assert after["version"] != before and after["sections"] == 1


def test_an_unparseable_knowledge_base_is_refused_before_it_is_written(client):
    before = client.get("/knowledge-base").json()["version"]
    r = client.post("/admin/knowledge-base",
                    json={"content": "no headings here at all, just a long line of prose "
                                     "that goes on and on without ever using a heading"},
                    headers=TOKEN)
    assert r.status_code == 422
    assert client.get("/knowledge-base").json()["version"] == before


def test_switching_to_real_is_refused_until_preflight_passes(client):
    r = client.post("/admin/mode", json={"mode": "REAL", "confirm": True}, headers=TOKEN)
    assert r.status_code == 412
    problems = r.json()["detail"]["problems"]
    assert any("LIVE_TRADING_CONFIRMED" in p for p in problems)
    assert client.post("/admin/mode", json={"mode": "REAL"},
                       headers=TOKEN).status_code == 400        # no confirm


def test_a_cycle_can_be_run_on_demand(client):
    result = client.post("/admin/cycle/run", json={"autotrade": False}, headers=TOKEN).json()
    assert result["cycle_id"] and result["status"] in ("ok", "protective_exit")
    assert result["decision"]["rationale"]
    assert client.get("/decisions").json()[0]["cycle_id"] == result["cycle_id"]


def test_resetting_the_paper_book_needs_confirmation_and_leaves_real_alone(client):
    assert client.post("/admin/paper/reset", headers=TOKEN).status_code == 400
    r = client.post("/admin/paper/reset", params={"confirm": True, "starting_cash": 5000},
                    headers=TOKEN)
    assert r.status_code == 200 and r.json()["cash"] == 5000.0
    assert client.get("/stats", params={"mode": "REAL"}).json()["trades"] == 0
