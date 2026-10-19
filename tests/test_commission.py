"""Weekly commission: a percent of each client's realized weekly profit, one
invoice per finished Monday-Sunday week, paid by UPI and marked by the client.

Pinned here: the money arithmetic, that a loss owes nothing, that a rate never
reprices or back-bills the past, that a paid invoice is frozen, that one client
cannot touch another's, and that a server which cannot see the real trades
never rewrites an amount due.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

import commission
import config_store
import mongo_client
import user_manager

# Monday 2026-10-12. Last finished week: Mon 5 Oct - Sun 11 Oct.
TODAY = date(2026, 10, 12)
LAST, PREV, THIS = "2026-10-05", "2026-09-28", "2026-10-12"


def _trade(user, day, pnl, status="CLOSED"):
    return {"trade_id": f"{user}-{day}-{pnl}-{status}",
            "timestamp": f"{day}T05:00:00", "user_id": user, "status": status,
            "realized_pnl": pnl if status == "CLOSED" else None}


@pytest.fixture
def world(monkeypatch, tmp_path):
    """No Mongo, an in-memory config store, a fake user table and a trade
    ledger the test fills in."""
    store: dict = {}
    monkeypatch.setattr(config_store, "load",
                        lambda name, default=None: json.loads(
                            json.dumps(store.get(name, dict(default or {})))))
    monkeypatch.setattr(config_store, "save",
                        lambda name, data: store.__setitem__(name, data))
    monkeypatch.setattr(mongo_client, "get_db", lambda: None)
    monkeypatch.setattr(commission, "_LOCAL_PATH",
                        str(tmp_path / "invoices.json"))
    monkeypatch.setattr(commission, "_today", lambda: TODAY)
    monkeypatch.setattr(commission, "_ledger_is_shared", lambda: True)

    users = {n: {"user_id": f"id-{n}", "username": n, "role": "client",
                 "display_name": n.upper(), "status": "active"}
             for n in ("c1", "c2")}
    monkeypatch.setattr(user_manager, "get_user", lambda n: users.get(n))
    monkeypatch.setattr(user_manager, "list_users",
                        lambda role=None: list(users.values()))

    def set_commission(user_id, pct, start):
        user = next(u for u in users.values() if u["user_id"] == user_id)
        user.update(commission_pct=pct, commission_from=start)
        return user
    monkeypatch.setattr(user_manager, "set_commission", set_commission)

    trades: dict[str, list] = {"Live": [], "Paper": []}
    monkeypatch.setattr(commission, "_read_trades",
                        lambda env: pd.DataFrame(trades[env]))
    return trades, users


def _one(username, week):
    return next(i for i in commission.sync(username) if i["week_start"] == week)


def test_twenty_percent_of_a_thousand_is_two_hundred(world):
    trades, _ = world
    trades["Live"] += [_trade("c1", "2026-10-06", 700.0),
                       _trade("c1", "2026-10-09", 300.0)]
    commission.set_rate("c1", 20, effective_from=LAST)
    inv = _one("c1", LAST)
    assert (inv["profit"], inv["pct"], inv["amount"], inv["status"]) == (
        1000.0, 20.0, 200.0, commission.DUE)
    assert inv["week_end"] == "2026-10-11"

    view = commission.client_view("c1")
    assert view["total_due"] == 200.0
    link = view["invoices"][0]["upi_uri"]
    assert link.startswith("upi://pay?pa=9458603249@ybl&")
    assert "am=200.00" in link and "cu=INR" in link


def test_a_losing_week_owes_nothing_and_is_not_carried_forward(world):
    trades, _ = world
    trades["Live"] += [_trade("c1", "2026-09-29", -5000.0),
                       _trade("c1", "2026-10-06", 1000.0)]
    commission.set_rate("c1", 20, effective_from=PREV)
    assert _one("c1", PREV)["status"] == commission.NIL
    assert _one("c1", PREV)["amount"] == 0.0
    assert _one("c1", LAST)["amount"] == 200.0
    assert commission.client_view("c1")["invoices"][1]["upi_uri"] == ""


def test_the_running_week_accrues_but_is_not_invoiced(world):
    trades, _ = world
    trades["Live"].append(_trade("c1", THIS, 500.0))
    commission.set_rate("c1", 10)
    view = commission.client_view("c1")
    assert view["invoices"] == []
    assert view["current_week"]["accrued"] == 50.0
    assert view["current_week"]["payable_on"] == "2026-10-19"


def test_switching_commission_on_never_back_bills_history(world):
    trades, _ = world
    trades["Live"].append(_trade("c1", "2026-10-06", 9000.0))
    out = commission.set_rate("c1", 20)
    assert out["commission_from"] == THIS
    assert commission.sync("c1") == []


def test_open_trades_other_clients_and_paper_are_not_charged(world):
    trades, _ = world
    trades["Live"] += [_trade("c1", "2026-10-06", 400.0),
                       _trade("c1", "2026-10-07", 999.0, status="OPEN"),
                       _trade("c2", "2026-10-06", 5000.0)]
    trades["Paper"].append(_trade("c1", "2026-10-06", 7000.0))
    commission.set_rate("c1", 25, effective_from=LAST)
    assert _one("c1", LAST)["amount"] == 100.0
    assert commission.sync("c2") == []


def test_a_new_rate_does_not_reprice_a_finished_week(world):
    trades, _ = world
    trades["Live"].append(_trade("c1", "2026-10-06", 1000.0))
    commission.set_rate("c1", 20, effective_from=LAST)
    commission.set_rate("c1", 50)
    inv = _one("c1", LAST)
    assert (inv["pct"], inv["amount"]) == (20.0, 200.0)


def test_rate_change_freezes_weeks_nobody_had_looked_at_yet(world):
    trades, users = world
    trades["Live"].append(_trade("c1", "2026-10-06", 1000.0))
    # Rate set directly: no invoice exists yet for the finished week.
    users["c1"].update(commission_pct=20.0, commission_from=LAST)
    commission.set_rate("c1", 50)
    assert _one("c1", LAST)["amount"] == 200.0


def test_unpaid_invoice_follows_the_ledger_paid_one_is_frozen(world):
    trades, _ = world
    trades["Live"].append(_trade("c1", "2026-10-06", 1000.0))
    commission.set_rate("c1", 20, effective_from=LAST)
    assert _one("c1", LAST)["amount"] == 200.0

    trades["Live"].append(_trade("c1", "2026-10-08", 500.0))
    assert _one("c1", LAST)["amount"] == 300.0

    paid = commission.mark_paid("c1", LAST, "UTR 4321!!<script>")
    assert paid["status"] == commission.PAID
    assert paid["paid_ref"] == "UTR 4321script"

    trades["Live"].append(_trade("c1", "2026-10-09", 9000.0))
    inv = _one("c1", LAST)
    assert (inv["amount"], inv["status"]) == (300.0, commission.PAID)


def test_only_a_due_invoice_can_be_marked_paid(world):
    trades, _ = world
    trades["Live"] += [_trade("c1", "2026-09-29", -10.0),
                       _trade("c1", "2026-10-06", 1000.0)]
    commission.set_rate("c1", 20, effective_from=PREV)
    commission.sync("c1")
    with pytest.raises(ValueError):
        commission.mark_paid("c1", PREV)            # nil
    commission.mark_paid("c1", LAST)
    with pytest.raises(ValueError):
        commission.mark_paid("c1", LAST)            # already paid
    with pytest.raises(ValueError):
        commission.mark_paid("c2", LAST)            # not theirs / none


def test_admin_verifies_and_reopens(world):
    trades, _ = world
    trades["Live"].append(_trade("c1", "2026-10-06", 1000.0))
    commission.set_rate("c1", 20, effective_from=LAST)
    commission.sync("c1")
    commission.mark_paid("c1", LAST, "REF1")
    assert commission.admin_set_status("c1", LAST, "verified")["status"] == \
        commission.VERIFIED
    reopened = commission.admin_set_status("c1", LAST, "due")
    assert (reopened["status"], reopened["paid_ref"]) == (commission.DUE, "")
    with pytest.raises(ValueError):
        commission.admin_set_status("c1", LAST, "due")
    with pytest.raises(ValueError):
        commission.admin_set_status("c1", LAST, "paid")


def test_a_server_that_cannot_see_the_trades_leaves_invoices_alone(world, monkeypatch):
    trades, _ = world
    trades["Live"].append(_trade("c1", "2026-10-06", 1000.0))
    commission.set_rate("c1", 20, effective_from=LAST)
    assert _one("c1", LAST)["amount"] == 200.0
    # This server's trade store fell back to an (empty) local file.
    monkeypatch.setattr(commission, "_ledger_is_shared", lambda: False)
    trades["Live"].clear()
    inv = _one("c1", LAST)
    assert (inv["amount"], inv["status"]) == (200.0, commission.DUE)


def test_paper_can_be_charged_for_a_dry_run(world):
    trades, _ = world
    trades["Paper"].append(_trade("c1", "2026-10-06", 2000.0))
    commission.update_settings("9458603249@ybl", "WelthWest", "Paper")
    commission.set_rate("c1", 10, effective_from=LAST)
    inv = _one("c1", LAST)
    assert (inv["environment"], inv["amount"]) == ("Paper", 200.0)


def test_settings_and_rate_are_validated(world):
    with pytest.raises(ValueError):
        commission.update_settings("not a upi id", "", "Live")
    with pytest.raises(ValueError):
        commission.set_rate("c1", 101)
    with pytest.raises(KeyError):
        commission.set_rate("nobody", 10)
    assert commission.settings()["upi_id"] == "9458603249@ybl"
    saved = commission.update_settings("shop.name@okaxis", "W W", "Live")
    assert saved["upi_id"] == "shop.name@okaxis"
    assert "pn=W%20W" in commission.upi_uri(saved["upi_id"], "W W", 12.5, "x")


def test_admin_overview_totals(world):
    trades, _ = world
    trades["Live"] += [_trade("c1", "2026-10-06", 1000.0),
                       _trade("c2", "2026-10-06", 2000.0)]
    commission.set_rate("c1", 20, effective_from=LAST)
    commission.set_rate("c2", 10, effective_from=LAST)
    commission.sync("c2")
    commission.mark_paid("c2", LAST)
    out = commission.admin_overview()
    assert out["totals"] == {"total_due": 200.0, "total_paid": 200.0,
                             "total_unverified": 200.0}
    assert [c["username"] for c in out["clients"]] == ["c1", "c2"]


def test_routes_scope_a_client_to_their_own_invoices(world):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.auth import CurrentUser, get_current_user
    from api.routers import commission as routes

    trades, _ = world
    trades["Live"].append(_trade("c1", "2026-10-06", 1000.0))
    commission.set_rate("c1", 20, effective_from=LAST)

    app = FastAPI()
    app.include_router(routes.router)
    who = {"name": "c2"}
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        user_id=f"id-{who['name']}", username=who["name"], role="client")
    http = TestClient(app)

    assert http.get("/commission/me").json()["invoices"] == []
    assert http.post(f"/commission/me/invoices/{LAST}/paid",
                     json={}).status_code == 400
    assert http.get("/commission/admin/overview").status_code == 403
    assert http.put("/commission/admin/clients/c2/rate",
                    json={"pct": 0}).status_code == 403

    who["name"] = "c1"
    assert http.get("/commission/me").json()["total_due"] == 200.0
    res = http.post(f"/commission/me/invoices/{LAST}/paid",
                    json={"reference": "UTR9"})
    assert res.status_code == 200 and res.json()["status"] == "paid"


# -- payment emails + history ------------------------------------------------ #
def test_marking_paid_emails_the_client_and_the_admin(world, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import commission_notify
    import mailer
    from api.auth import CurrentUser, get_current_user
    from api.routers import commission as routes

    trades, _ = world
    trades["Live"].append(_trade("c1", "2026-10-06", 1000.0))
    commission.set_rate("c1", 20, effective_from=LAST)

    real_get = user_manager.get_user
    monkeypatch.setattr(user_manager, "get_user", lambda n: (
        {**(real_get(n) or {}), "email": "Client@Example.com",
         "display_name": "Client One"} if n == "c1" else real_get(n)))
    monkeypatch.setenv("COMMISSION_ADMIN_EMAIL", "boss@example.com")
    monkeypatch.setattr(mailer, "is_configured", lambda: True)
    sent = []
    monkeypatch.setattr(mailer, "send",
                        lambda to, subj, body: sent.append((to, subj, body)) or (True, ""))
    commission.sync("c1")
    real_notify = commission_notify.notify_paid
    monkeypatch.setattr(commission_notify, "notify_paid",
                        lambda inv: real_notify(inv, background=False))

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        user_id="id-c1", username="c1", role="client")
    res = TestClient(app).post(f"/commission/me/invoices/{LAST}/paid",
                               json={"reference": "UTR77"})
    assert res.status_code == 200

    to = {s[0]: s for s in sent}
    assert set(to) == {"client@example.com", "boss@example.com"}
    client_subj, client_body = to["client@example.com"][1:]
    assert "20% commission to WelthWest Algo" in client_subj
    assert "Rs 200.00" in client_body and "UTR77" in client_body
    assert "Client One (c1)" in to["boss@example.com"][1]


def test_a_mail_failure_never_fails_mark_paid(world, monkeypatch):
    import commission_notify
    import mailer

    monkeypatch.setattr(mailer, "is_configured", lambda: True)

    def boom(*a):
        raise RuntimeError("smtp down")
    monkeypatch.setattr(mailer, "send", boom)
    commission_notify.notify_paid(
        {"invoice_id": "c1:x", "username": "c1", "week_start": LAST,
         "week_end": LAST, "amount": 1, "pct": 1}, background=False)


def test_admin_recipients_fall_back_to_admin_accounts(world, monkeypatch):
    import commission_notify

    monkeypatch.delenv("COMMISSION_ADMIN_EMAIL", raising=False)
    monkeypatch.setattr(user_manager, "list_users", lambda role=None: [
        {"username": "admin", "role": "admin", "email": "Owner@Example.com"}])
    assert commission_notify.admin_recipients() == ["owner@example.com"]


def test_admin_overview_lists_payment_history(world):
    trades, _ = world
    trades["Live"].append(_trade("c1", "2026-10-06", 1000.0))
    trades["Live"].append(_trade("c2", "2026-10-06", 500.0))
    commission.set_rate("c1", 20, effective_from=LAST)
    commission.set_rate("c2", 10, effective_from=LAST)
    commission.sync("c1")
    commission.mark_paid("c1", LAST, "R1")
    payments = commission.admin_overview()["payments"]
    assert [(p["username"], p["amount"], p["paid_ref"]) for p in payments] \
        == [("c1", 200.0, "R1")]
    assert payments[0]["display_name"]
