import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI()
START_TS = time.time()


Scope = Literal["category", "merchant", "customer", "trigger"]
SendAs = Literal["vera", "merchant_on_behalf"]


class ContextPush(BaseModel):
    scope: Scope
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


class StoredContext(BaseModel):
    version: int
    payload: Dict[str, Any]


contexts: Dict[Tuple[Scope, str], StoredContext] = {}
seen_suppression_keys: set[str] = set()
conversation_turns: Dict[str, List[Dict[str, Any]]] = {}
conversation_last_bot_body: Dict[str, str] = {}
auto_reply_counts: Dict[Tuple[str, str], int] = {}


AUTO_REPLY_PATTERNS = [
    r"\bthank you for contacting\b",
    r"\bour team will respond\b",
    r"\bwe will get back\b",
    r"\bthis is an automated\b",
    r"\bfor any queries\b",
]

HOSTILE_PATTERNS = [
    r"\bstop\b.*\bmessage",
    r"\bunsubscribe\b",
    r"\bdon'?t message\b",
    r"\buseless\b",
    r"\bspam\b",
]

COMMITMENT_PATTERNS = [
    r"\blet'?s do it\b",
    r"\blets do it\b",
    r"\bgo ahead\b",
    r"\bok\b.*\bwhat'?s next\b",
    r"\bwhat'?s next\b",
    r"\byes\b.*\bdo it\b",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get(scope: Scope, context_id: str) -> Optional[Dict[str, Any]]:
    entry = contexts.get((scope, context_id))
    return entry.payload if entry else None


def _merchant_display_name(merchant: Dict[str, Any]) -> str:
    ident = merchant.get("identity", {}) if merchant else {}
    return ident.get("name") or "there"


def _merchant_first_name(merchant: Dict[str, Any]) -> Optional[str]:
    ident = merchant.get("identity", {}) if merchant else {}
    return ident.get("owner_first_name")


def _prefers_hinglish(merchant: Dict[str, Any]) -> bool:
    langs = (merchant.get("identity", {}) or {}).get("languages", []) or []
    return "hi" in langs


def _safe_pct(x: Any) -> Optional[int]:
    try:
        return int(round(float(x) * 100))
    except Exception:
        return None


def _is_auto_reply(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(re.search(p, t) for p in AUTO_REPLY_PATTERNS)


def _is_hostile_or_stop(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(re.search(p, t) for p in HOSTILE_PATTERNS)


def _is_commitment(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(re.search(p, t) for p in COMMITMENT_PATTERNS)


def _shorten(s: str, max_len: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_len else (s[: max_len - 1].rstrip() + "…")


def _pick_active_offer_title(merchant: Dict[str, Any], category: Dict[str, Any]) -> Optional[str]:
    offers = merchant.get("offers", []) if merchant else []
    for o in offers:
        if o.get("status") == "active" and o.get("title"):
            return o["title"]
    catalog = category.get("offer_catalog", []) if category else []
    if catalog:
        title = catalog[0].get("title")
        return title
    return None


def _find_digest_item(category: Dict[str, Any], item_id: str) -> Optional[Dict[str, Any]]:
    for item in category.get("digest", []) if category else []:
        if item.get("id") == item_id:
            return item
    return None


def _template_for_first_touch(trigger_kind: str, send_as: SendAs) -> str:
    if send_as == "merchant_on_behalf":
        if trigger_kind in {"recall_due", "chronic_refill_due", "trial_followup", "wedding_package_followup"}:
            return "merchant_customer_outreach_v1"
        return "merchant_customer_generic_v1"

    if trigger_kind in {"research_digest", "regulation_change", "supply_alert", "cde_opportunity"}:
        return "vera_knowledge_nudge_v1"
    if trigger_kind in {"renewal_due"}:
        return "vera_subscription_nudge_v1"
    return "vera_generic_v1"


def compose_action(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kind = trigger.get("kind", "unknown")
    suppression_key = trigger.get("suppression_key") or f"{kind}:{trigger.get('id', '')}"
    send_as: SendAs = "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera"

    mname = _merchant_display_name(merchant)
    first = _merchant_first_name(merchant)
    hinglish = _prefers_hinglish(merchant)

    body = ""
    cta = "open_ended"
    rationale = ""

    if kind == "research_digest":
        top_id = (trigger.get("payload") or {}).get("top_item_id")
        item = _find_digest_item(category, top_id) if top_id else None
        title = item.get("title") if item else "this week's digest item"
        source = item.get("source") if item else None
        trial_n = item.get("trial_n") if item else None
        seg = item.get("patient_segment") if item else None
        ctr = (merchant.get("performance") or {}).get("ctr")
        peer_ctr = (category.get("peer_stats") or {}).get("avg_ctr")
        ctr_pct = _safe_pct(ctr)
        peer_pct = _safe_pct(peer_ctr)

        who = f"Dr. {first}" if (first and category.get("slug") == "dentists") else (first or mname)
        line1 = f"{who}, {title}"
        details = []
        if trial_n:
            details.append(f"trial n={trial_n}")
        if seg:
            details.append(seg.replace("_", " "))
        if ctr_pct is not None and peer_pct is not None and ctr_pct < peer_pct:
            details.append(f"your CTR {ctr_pct}% vs peer ~{peer_pct}%")
        tail = f"Want me to draft a 4-line WhatsApp you can share with patients?" if hinglish else "Want me to draft a 4-line WhatsApp you can share with customers?"
        cite = f"— {source}" if source else ""
        body = _shorten(f"{line1}. ({', '.join(details)}) Worth a look. {tail} {cite}".replace("()", "").strip(), 700)
        cta = "open_ended"
        rationale = "Using category digest item + merchant performance anchor; open-ended CTA invites a low-friction next step."

    elif kind == "regulation_change":
        top_id = (trigger.get("payload") or {}).get("top_item_id")
        deadline = (trigger.get("payload") or {}).get("deadline_iso")
        item = _find_digest_item(category, top_id) if top_id else None
        title = item.get("title") if item else "Regulation update"
        source = item.get("source") if item else None
        actionable = item.get("actionable") if item else None
        who = f"Dr. {first}" if (first and category.get("slug") == "dentists") else (first or mname)
        body = f"{who}, compliance heads-up: {title}."
        if deadline:
            body += f" Effective by {deadline}."
        if actionable:
            body += f" Action: {actionable}."
        if source:
            body += f" — {source}"
        cta = "binary_yes_no"
        body += " Reply YES if you want a 30-sec checklist to audit your setup."
        rationale = "Regulation trigger with deadline + actionable next step; binary CTA reduces friction."

    elif kind in {"perf_dip", "perf_spike", "seasonal_perf_dip"}:
        metric = (trigger.get("payload") or {}).get("metric") or "performance"
        delta = (trigger.get("payload") or {}).get("delta_pct")
        window = (trigger.get("payload") or {}).get("window") or "7d"
        pct = _safe_pct(delta)
        who = first or mname
        sign = "up" if (pct is not None and pct > 0) else "down"
        if pct is None:
            body = f"{who}, quick update: your {metric} changed in the last {window}. Want me to break down what likely drove it and what to do next?"
        else:
            body = f"{who}, your {metric} is {sign} {abs(pct)}% over {window}."
            driver = (trigger.get("payload") or {}).get("likely_driver")
            if driver:
                body += f" Likely driver: {driver.replace('_', ' ')}."
            if kind == "seasonal_perf_dip" and (trigger.get("payload") or {}).get("is_expected_seasonal"):
                body += " This dip is seasonal—so I'd focus on retention vs spending on ads right now."
            body += " Want me to draft a 1-post + 1-offer plan for the next 7 days?"
        cta = "open_ended"
        rationale = "Performance trigger grounded in delta% and window; offers a concrete draftable artifact to prompt reply."

    elif kind == "renewal_due":
        days = (trigger.get("payload") or {}).get("days_remaining")
        plan = (trigger.get("payload") or {}).get("plan")
        amount = (trigger.get("payload") or {}).get("renewal_amount")
        who = first or mname
        body = f"{who}, your {plan or ''} plan renews in {days} days.".replace("  ", " ").strip()
        if amount:
            body += f" Renewal amount ₹{amount}."
        body += " Reply YES if you want me to share the renewal link + what you keep/lose if it lapses."
        cta = "binary_yes_no"
        rationale = "Renewal trigger: specific days remaining + amount; single binary CTA to move to action quickly."

    elif kind == "festival_upcoming":
        festival = (trigger.get("payload") or {}).get("festival", "festival")
        days = (trigger.get("payload") or {}).get("days_until")
        offer = _pick_active_offer_title(merchant, category)
        who = first or mname
        body = f"{who}, {festival} is coming up"
        if isinstance(days, int):
            body += f" in {days} days"
        body += "."
        if offer:
            body += f" You already have “{offer}” — want me to draft a {festival} variant for your Google post + WhatsApp reply?"
        else:
            body += f" Want a {festival} offer in service+price format (not % off) for your category? Reply YES."
            cta = "binary_yes_no"
        rationale = "Festival trigger anchored by days-until + existing offer presence; asks for a small commitment to draft creatives."

    elif kind == "ipl_match_today":
        match = (trigger.get("payload") or {}).get("match")
        venue = (trigger.get("payload") or {}).get("venue")
        tiso = (trigger.get("payload") or {}).get("match_time_iso")
        is_weeknight = (trigger.get("payload") or {}).get("is_weeknight")
        who = first or mname
        offer = _pick_active_offer_title(merchant, category)
        body = f"{who}, {match} today"
        if venue:
            body += f" at {venue}"
        if tiso:
            body += f" ({tiso[-14:-6]})."
        else:
            body += "."
        if is_weeknight is False:
            body += " Weekend matches usually shift dine-in demand to delivery."
        if offer:
            body += f" Your offer “{offer}” can be framed as delivery-only for match night. Want me to draft 2 lines for Swiggy/Zomato + a WhatsApp snippet?"
        else:
            body += " Want me to draft a match-night delivery combo in service+price style? Reply YES."
            cta = "binary_yes_no"
        rationale = "IPL trigger: uses match details + weekend nuance; proposes concrete draft artifacts for engagement."

    elif kind == "review_theme_emerged":
        theme = (trigger.get("payload") or {}).get("theme")
        occ = (trigger.get("payload") or {}).get("occurrences_30d")
        quote = (trigger.get("payload") or {}).get("common_quote")
        who = first or mname
        body = f"{who}, a quick pattern in recent reviews: {theme.replace('_', ' ') if theme else 'a repeated theme'}"
        if occ:
            body += f" ({occ} mentions in 30d)."
        else:
            body += "."
        if quote:
            body += f" Example: “{_shorten(str(quote), 80)}”."
        body += " Want me to draft a 2-line public reply + one operational fix to reduce repeats?"
        cta = "open_ended"
        rationale = "Review-theme trigger grounded in count + example quote; asks for permission to draft replies and fixes."

    elif kind == "milestone_reached":
        metric = (trigger.get("payload") or {}).get("metric")
        value_now = (trigger.get("payload") or {}).get("value_now")
        milestone = (trigger.get("payload") or {}).get("milestone_value")
        who = first or mname
        body = f"{who}, you’re close to a milestone: {metric.replace('_', ' ') if metric else 'milestone'}"
        if value_now is not None and milestone is not None:
            body += f" ({value_now}/{milestone})."
        else:
            body += "."
        body += " Want a short message you can send customers to nudge 1 more review this week?"
        cta = "binary_yes_no"
        rationale = "Milestone trigger with concrete progress; proposes a simple next step with binary CTA."

    elif kind in {"active_planning_intent"}:
        topic = (trigger.get("payload") or {}).get("intent_topic", "a package")
        who = first or mname
        body = f"{who}, got it — I’ll draft a starter version for {topic.replace('_', ' ')} (tiers + what’s included)."
        body += " Reply YES and tell me your target price band (e.g., ₹99/₹149/₹199) and I’ll finalize."
        cta = "binary_yes_no"
        rationale = "Intent trigger: switches to action mode immediately by offering a draft artifact and only one needed input."

    elif kind in {"dormant_with_vera", "curious_ask_due"}:
        who = first or mname
        if hinglish:
            body = f"Hi {who}! Quick check — is week mein sabse zyada kis service ka demand aa raha hai? Main usko Google post + pricing WhatsApp reply mein draft kar dungi."
        else:
            body = f"Hi {who}! Quick check — what service has been most asked-for this week? I’ll turn it into a Google post + a pricing WhatsApp reply draft."
        cta = "open_ended"
        rationale = "Low-friction curious ask to restart engagement; offers immediate value (drafts) to prompt reply."

    elif kind == "supply_alert":
        molecule = (trigger.get("payload") or {}).get("molecule")
        batches = (trigger.get("payload") or {}).get("affected_batches") or []
        mfr = (trigger.get("payload") or {}).get("manufacturer")
        who = first or mname
        body = f"{who}, urgent stock/compliance alert: {molecule} recall"
        if batches:
            body += f" (batches: {', '.join(batches[:3])})."
        else:
            body += "."
        if mfr:
            body += f" Manufacturer: {mfr}."
        body += " Reply YES if you want a customer notification draft + replacement workflow."
        cta = "binary_yes_no"
        rationale = "Supply alert uses batch numbers and molecule; binary CTA for a ready-to-send workflow draft."

    elif kind == "category_seasonal":
        season = (trigger.get("payload") or {}).get("season", "this season")
        trends = (trigger.get("payload") or {}).get("trends") or []
        who = first or mname
        body = f"{who}, {season.replace('_', ' ')} shelf check: "
        if trends:
            body += _shorten(", ".join(trends[:4]).replace("_", " "), 120) + ". "
        body += "Want me to draft a 3-line WhatsApp broadcast + one counter display suggestion?"
        cta = "open_ended"
        rationale = "Seasonal trend trigger grounded in listed demand shifts; offers concrete drafts."

    # Customer-facing kinds
    elif kind == "recall_due" and customer:
        cname = (customer.get("identity", {}) or {}).get("name") or "there"
        slots = ((trigger.get("payload") or {}).get("available_slots") or [])[:2]
        slot_labels = [s.get("label") for s in slots if s.get("label")]
        offer = _pick_active_offer_title(merchant, category) or "a cleaning slot"
        clinic = mname
        lang_pref = (customer.get("identity", {}) or {}).get("language_pref", "")
        mix = "mix" in str(lang_pref).lower() or "hi" in str(lang_pref).lower()
        due_date = (trigger.get("payload") or {}).get("due_date")

        if mix:
            body = f"Hi {cname}, {clinic} here. Aapka recall due hai"
            if due_date:
                body += f" (due: {due_date})."
            else:
                body += "."
            if slot_labels:
                body += f" 2 slots ready hain: {slot_labels[0]} ya {slot_labels[1] if len(slot_labels) > 1 else ''}."
            body += f" Offer: {offer}. Reply 1/2 ya apna time bata dein."
        else:
            body = f"Hi {cname}, {clinic} here. Your recall is due"
            if due_date:
                body += f" (due: {due_date})."
            else:
                body += "."
            if slot_labels:
                body += f" 2 slots ready: {slot_labels[0]} or {slot_labels[1] if len(slot_labels) > 1 else ''}."
            body += f" Offer: {offer}. Reply 1/2 or tell us a time that works."
        body = _shorten(re.sub(r"\s+", " ", body).strip(), 700)
        cta = "multi_choice_slot"
        rationale = "Customer-scoped recall message uses customer name + offered slots + real offer; CTA is slot selection."

    elif kind in {"customer_lapsed_hard", "trial_followup", "chronic_refill_due", "wedding_package_followup"} and customer:
        cname = (customer.get("identity", {}) or {}).get("name") or "there"
        clinic = mname
        lang_pref = (customer.get("identity", {}) or {}).get("language_pref", "")
        mix = "mix" in str(lang_pref).lower() or "hi" in str(lang_pref).lower()

        if kind == "chronic_refill_due":
            payload = trigger.get("payload") or {}
            molecules = payload.get("molecule_list") or []
            runs_out = payload.get("stock_runs_out_iso")
            mol_txt = ", ".join(molecules[:4]) if molecules else "your medicines"
            if mix:
                body = f"Namaste {cname} — {clinic} here. Aapki {mol_txt} {runs_out[:10] if runs_out else 'soon'} ko khatam ho rahi hain. Reply CONFIRM to prepare the pack."
            else:
                body = f"Namaste {cname} — {clinic} here. Your {mol_txt} runs out on {runs_out[:10] if runs_out else 'soon'}. Reply CONFIRM to prepare the pack."
            cta = "binary_confirm_cancel"
            rationale = "Refill reminder uses molecule list + run-out date; confirm CTA to proceed."

        elif kind == "customer_lapsed_hard":
            days = (trigger.get("payload") or {}).get("days_since_last_visit")
            focus = (trigger.get("payload") or {}).get("previous_focus")
            if mix:
                body = f"Hi {cname} 👋 {clinic} here. ~{days} days ho gaye — no stress. {focus.replace('_',' ') if focus else 'A quick session'} ke liye ek trial slot hold karu? Reply YES."
            else:
                body = f"Hi {cname} 👋 {clinic} here. It’s been ~{days} days — no judgment. Want me to hold a free trial slot for your {focus.replace('_',' ') if focus else 'goal'}? Reply YES."
            cta = "binary_yes_no"
            rationale = "Winback message is warm/no-shame + grounded in days-since-last-visit; low-friction YES CTA."

        elif kind == "trial_followup":
            opts = (trigger.get("payload") or {}).get("next_session_options") or []
            label = (opts[0] or {}).get("label") if opts else None
            if mix:
                body = f"Hi {cname}, {clinic} here — trial ke baad next session book kar dein? {label or ''} slot available hai. Reply YES to confirm."
            else:
                body = f"Hi {cname}, {clinic} here — want to lock your next session after the trial? {label or ''} is available. Reply YES to confirm."
            cta = "binary_yes_no"
            rationale = "Trial follow-up uses the next session option label; YES CTA to confirm."

        elif kind == "wedding_package_followup":
            days = (trigger.get("payload") or {}).get("days_to_wedding")
            if mix:
                body = f"Hi {cname} 💍 {clinic} here. Wedding in {days} days — skin-prep window open hai. Want me to share a 30-day plan + price? Reply YES."
            else:
                body = f"Hi {cname} 💍 {clinic} here. Wedding in {days} days — perfect window to start prep. Want a 30-day plan + price options? Reply YES."
            cta = "binary_yes_no"
            rationale = "Bridal follow-up grounded in days-to-wedding; asks permission to send plan."

    else:
        who = first or mname
        body = f"{who}, I noticed something worth a quick look. Want me to draft the next best step for you?"
        cta = "open_ended"
        rationale = "Fallback nudge that asks a single open-ended question without inventing facts."

    template_name = _template_for_first_touch(kind, send_as)
    template_params = [
        _shorten(first or mname, 40),
        _shorten(body, 120),
        _shorten((trigger.get("id") or kind), 60),
    ]

    return {
        "send_as": send_as,
        "template_name": template_name,
        "template_params": template_params,
        "body": body,
        "cta": cta,
        "suppression_key": suppression_key,
        "rationale": rationale,
    }


@app.get("/v1/healthz")
async def healthz():
    counts: Dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in contexts.keys():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START_TS), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Local Submission",
        "team_members": ["ravi5"],
        "model": "heuristic-v1 (no external LLM)",
        "approach": "deterministic composer with trigger-kind dispatch + safety checks (auto-reply/hostile/commitment)",
        "contact_email": "local@example.com",
        "version": "0.1.0",
        "submitted_at": _utc_now_iso(),
    }


@app.post("/v1/context")
async def push_context(body: ContextPush):
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur.version >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur.version}
    contexts[key] = StoredContext(version=body.version, payload=body.payload)
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": _utc_now_iso()}


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions: List[Dict[str, Any]] = []

    for trg_id in body.available_triggers[:20]:
        trigger = _get("trigger", trg_id)
        if not trigger:
            continue

        suppression_key = trigger.get("suppression_key")
        if suppression_key and suppression_key in seen_suppression_keys:
            continue

        merchant_id = trigger.get("merchant_id")
        merchant = _get("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue

        category_slug = merchant.get("category_slug")
        category = _get("category", category_slug) if category_slug else None
        if not category:
            continue

        customer = None
        customer_id = trigger.get("customer_id")
        if trigger.get("scope") == "customer" and customer_id:
            customer = _get("customer", customer_id)
            if not customer:
                continue

        composed = compose_action(category, merchant, trigger, customer)
        conv_id = f"conv_{merchant_id}_{trg_id}"
        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed["send_as"],
            "trigger_id": trg_id,
            "template_name": composed["template_name"],
            "template_params": composed["template_params"],
            "body": composed["body"],
            "cta": composed["cta"],
            "suppression_key": composed["suppression_key"],
            "rationale": composed["rationale"],
        }
        actions.append(action)
        if suppression_key:
            seen_suppression_keys.add(suppression_key)

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = conversation_turns.setdefault(body.conversation_id, [])
    conv.append({"from": body.from_role, "message": body.message, "received_at": body.received_at})

    msg = (body.message or "").strip()
    msg_norm = msg.lower()
    merchant_key = (body.merchant_id or "unknown").strip()

    # Hard opt-out / hostile
    if _is_hostile_or_stop(msg):
        return {"action": "end", "rationale": "Merchant/customer explicitly asked to stop or expressed hostility; closing conversation."}

    # Auto-reply handling (escalate: nudge once, then wait, then end)
    if _is_auto_reply(msg):
        k = (merchant_key, msg_norm)
        auto_reply_counts[k] = auto_reply_counts.get(k, 0) + 1
        repeats = auto_reply_counts[k]

        if repeats >= 4:
            return {"action": "end", "rationale": "Detected the same canned auto-reply 4x for this merchant; ending to avoid wasting turns."}
        if repeats >= 2:
            return {"action": "wait", "wait_seconds": 86400, "rationale": "Repeated auto-reply from merchant; waiting 24h for a real owner reply."}
        return {"action": "wait", "wait_seconds": 14400, "rationale": "Detected likely WhatsApp auto-reply; backing off 4h for the owner/manager."}

    # Commitment / "let's do it" — switch to action mode
    if _is_commitment(msg):
        return {
            "action": "send",
            "body": "Done - I'm drafting it now. Share your 1 goal for this week (more calls / more directions / more bookings) and I'll send back a ready-to-post draft you can approve.",
            "cta": "open_ended",
            "rationale": "Detected explicit commitment; switching from qualifying to action with a concrete next step.",
        }

    # Lightweight intent detection + context awareness
    yes_re = re.compile(r"\b(yes|yep|y|sure|confirm|ok|1|reply\s*yes|please do)\b")
    no_re = re.compile(r"\b(no|nah|not now|cancel|2|don't|dont)\b")
    slot_choice_re = re.compile(r"\b([12])\b")
    draft_re = re.compile(r"\b(draft|post|whatsapp|google post|draft it|send.*draft)\b")

    is_yes = bool(yes_re.search(msg_norm))
    is_no = bool(no_re.search(msg_norm))
    slot_choice_m = slot_choice_re.search(msg_norm)
    slot_choice = int(slot_choice_m.group(1)) if slot_choice_m else None
    is_draft = bool(draft_re.search(msg_norm))

    # Try to infer trigger id from conversation id (tick uses conv_{merchant_id}_{trigger_id})
    trigger = None
    try:
        if body.conversation_id.startswith("conv_") and "_" in body.conversation_id:
            trg_id = body.conversation_id.rsplit("_", 1)[-1]
            trigger = _get("trigger", trg_id)
    except Exception:
        trigger = None

    merchant = _get("merchant", body.merchant_id) if body.merchant_id else None
    category = _get("category", (merchant.get("category_slug") if merchant else None)) if merchant else None
    customer = _get("customer", body.customer_id) if body.customer_id else None

    # If the last outbound action invited a YES/NO and user said YES, produce the concrete draft
    if is_yes and trigger:
        composed = compose_action(category or {}, merchant or {}, trigger or {}, customer)
        # If CTA was binary or open-ended where a draft makes sense, send the draft
        return {
            "action": "send",
            "body": f"Okay — here’s a draft you can use:\n\n{composed.get('body')}",
            "cta": "open_ended",
            "rationale": "User affirmed (YES) and trigger context exists; returning a draft based on trigger + merchant context.",
        }

    # Slot selection for customer-scoped triggers (e.g., recall_due)
    if slot_choice and trigger and trigger.get("scope") == "customer":
        payload = trigger.get("payload") or {}
        slots = (payload.get("available_slots") or [])[:3]
        idx = slot_choice - 1
        if 0 <= idx < len(slots):
            chosen = slots[idx]
            label = chosen.get("label") or str(chosen)
            who = (customer.get("identity", {}) or {}).get("name") if customer else "there"
            clinic = _merchant_display_name(merchant) if merchant else "the clinic"
            return {
                "action": "send",
                "body": f"Confirmed — booking {label} for {who}. I’ll notify {clinic} and hold this slot.",
                "cta": "open_ended",
                "rationale": "User selected a slot from the customer-scoped trigger; booking acknowledged.",
            }
        else:
            # invalid slot index
            return {"action": "send", "body": "I couldn't find that slot — reply 1 or 2 to pick a slot.", "cta": "multi_choice_slot", "rationale": "Invalid slot choice."}

    # If user asked for a draft explicitly, attempt to build one from trigger context
    if is_draft and trigger:
        composed = compose_action(category or {}, merchant or {}, trigger or {}, customer)
        return {"action": "send", "body": composed.get("body"), "cta": "open_ended", "rationale": "User asked for a draft; returning a context-aware draft."}

    # If user said NO explicitly, acknowledge and close or ask for next preference
    if is_no:
        return {"action": "send", "body": "No problem — what would you prefer instead? More calls, more walk-ins, or more repeat customers?", "cta": "open_ended", "rationale": "User declined; asking for alternative priority."}

    # Quick food/order intent shortcut
    if "food" in msg_norm or "order" in msg_norm:
        return {
            "action": "send",
            "body": "Nice! What would you like to order — pizza, burgers, or something else?",
            "cta": "suggest_options",
            "rationale": "Detected food ordering intent",
        }

    # Fallback: use merchant/customer context to make the acknowledgement less generic
    who = None
    if customer:
        who = (customer.get("identity", {}) or {}).get("name")
    if not who and merchant:
        who = _merchant_first_name(merchant) or _merchant_display_name(merchant)
    who = who or "there"

    last_bot = conversation_last_bot_body.get(body.conversation_id, "")
    response_body = f"Got it, {who}. What should I prioritize right now — more calls, more walk-ins, or more repeat customers?"
    if response_body.strip() == last_bot.strip():
        response_body = f"Understood, {who}. Tell me your top priority (calls / walk-ins / repeat) and I’ll draft the next message accordingly."
    conversation_last_bot_body[body.conversation_id] = response_body

    return {
        "action": "send",
        "body": response_body,
        "cta": "open_ended",
        "rationale": "Acknowledged the reply and asked one low-friction question to route the next action (context-aware).",
    }

