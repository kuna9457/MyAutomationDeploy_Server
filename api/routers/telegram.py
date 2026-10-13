"""
api/routers/telegram.py
Admin-only surface for the Telegram signal channel (telegram_signals.py):
connect the bot, see every signal posted and its outcome, and correct or
withdraw one. Nothing here reaches the trading path — editing a published
stop or target changes the CHANNEL's levels, never an account's position.
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import credentials_vault
import telegram_signals as tg
from api.auth import CurrentUser, require_admin

router = APIRouter(prefix="/telegram", tags=["telegram"])


class TelegramConfigRequest(BaseModel):
    enabled: bool = False
    chat_id: str = ""
    post_exits: bool = True
    header: str = Field("", max_length=120)
    footer: str = Field("", max_length=300)
    #: Blank keeps the saved token. Never echoed back.
    bot_token: str = ""
    clear_token: bool = False


class ManualSignalRequest(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    entry: float
    stop: float
    target: float


class EditLevelsRequest(BaseModel):
    stop: Optional[float] = None
    target: Optional[float] = None
    #: Also post a reply in the channel so followers are notified.
    notify: bool = True


class CloseRequest(BaseModel):
    price: Optional[float] = None
    cancel: bool = False


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError:
        raise HTTPException(404, "No such signal.")
    except (ValueError, tg.TelegramError) as exc:
        raise HTTPException(400, str(exc))
    except credentials_vault.VaultUnavailable:
        raise HTTPException(400, "CREDENTIALS_KEY is not set on this server, so the "
                                 "bot token can't be stored encrypted. Set it in .env, "
                                 "or put TELEGRAM_BOT_TOKEN in .env instead.")


@router.get("/config")
def get_config(_: CurrentUser = Depends(require_admin)):
    return tg.public_config()


@router.put("/config")
def put_config(req: TelegramConfigRequest, _: CurrentUser = Depends(require_admin)):
    return _run(tg.update_config, **req.model_dump())


@router.post("/test")
async def test(_: CurrentUser = Depends(require_admin)):
    return await run_in_threadpool(_run, tg.send_test)


@router.get("/signals")
def signals(_: CurrentUser = Depends(require_admin)):
    return tg.list_signals()


@router.post("/signals")
def post_signal(req: ManualSignalRequest, _: CurrentUser = Depends(require_admin)):
    return _run(tg.post_manual, req.symbol, req.side, req.entry, req.stop, req.target)


@router.put("/signals/{signal_id}")
def edit_signal(signal_id: str, req: EditLevelsRequest,
                _: CurrentUser = Depends(require_admin)):
    return _run(tg.edit_levels, signal_id, req.stop, req.target, req.notify)


@router.post("/signals/{signal_id}/close")
def close_signal(signal_id: str, req: CloseRequest,
                 _: CurrentUser = Depends(require_admin)):
    return _run(tg.close_signal, signal_id, req.price, req.cancel)


@router.post("/signals/{signal_id}/resend")
def resend_signal(signal_id: str, _: CurrentUser = Depends(require_admin)):
    return _run(tg.resend, signal_id)


@router.delete("/signals/{signal_id}")
def delete_signal(signal_id: str, from_channel: bool = True,
                  _: CurrentUser = Depends(require_admin)):
    _run(tg.delete_signal, signal_id, from_channel)
    return {"ok": True}
