# state_manager.py  (phiên bản data-driven: đọc flows.json)
import json, re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# ---------- Extractors ----------
YES = {"có","ok","oke","okay","đồng ý","vâng","phải","ừ","ừm"}
NO  = {"không","khong","ko","thôi","không cần","không có","no"}

def extract_yesno(text: str) -> Optional[bool]:
    t = text.lower()
    if any(w in t for w in YES): return True
    if any(w in t for w in NO):  return False
    return None

DATE_PATTERNS = [r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"]
def extract_date(text: str) -> Optional[datetime]:
    t = text.strip()
    m = re.search(DATE_PATTERNS[0], t)
    if m:
        d, mth, y = map(int, m.groups())
        try: return datetime(y, mth, d)
        except ValueError: pass
    m = re.search(DATE_PATTERNS[1], t)
    if m:
        y, mth, d = map(int, m.groups())
        try: return datetime(y, mth, d)
        except ValueError: pass
    return None

EXTRACTORS = {
    "yesno": extract_yesno,
    "date":  extract_date,
    # có thể bổ sung: "color", "number", "book_title"...
}

# ---------- Computations ----------
def compute_borrow_due_date(ctx: Dict[str, Any]) -> str:
    d: datetime = ctx["borrow_date"]
    due = d + timedelta(days=14)
    return (f"Thời hạn mượn là 14 ngày từ ngày {d.strftime('%d/%m/%Y')}."
            f" Hạn trả dự kiến: {due.strftime('%d/%m/%Y')}.")

COMPUTE_FNS = {
    "borrow_due_date": compute_borrow_due_date,
}

# ---------- State Manager ----------
class StateManager:
    def __init__(self, flows_path: str = "flows.json"):
        with open(flows_path, "r", encoding="utf-8") as f:
            self.flows = json.load(f)
        self.reset()

    def reset(self):
        self.active_flow: Optional[str] = None
        self.step_idx: int = 0
        self.ctx: Dict[str, Any] = {}
        self.pending_followup: Optional[str] = None

    def start_flow(self, intent: str) -> Optional[str]:
        spec = self.flows.get(intent)
        if not spec: return None

        steps = spec.get("steps")
        if steps:
            self.active_flow = intent
            self.step_idx = 0
            self.pending_followup = None
            return steps[0]["ask"]

        # followup-only
        if spec.get("followup"):
            self.pending_followup = intent
            return None
        return None

    def handle(self, predicted_intent: str, user_text: str) -> Optional[str]:
        # 1) nếu đang ở flow nhiều bước -> điền slot
        if self.active_flow:
            spec  = self.flows[self.active_flow]
            steps = spec["steps"]
            step  = steps[self.step_idx]

            slot  = step["slot"]
            typ   = step["type"]
            ext   = EXTRACTORS.get(typ)

            value = ext(user_text) if ext else None
            if value is None:
                return step["ask"]  # nhắc lại

            # lưu slot
            self.ctx[slot] = value

            # on_filled
            filled_reply = None
            of = step.get("on_filled") or {}
            if "reply_true" in of or "reply_false" in of:
                # yes/no
                if isinstance(value, bool):
                    filled_reply = of["reply_true"] if value else of["reply_false"]
            if "compute" in of:
                fn = COMPUTE_FNS.get(of["compute"])
                if fn: filled_reply = fn(self.ctx)

            # sang bước kế
            self.step_idx += 1
            if self.step_idx >= len(steps):
                self.reset()
            return filled_reply or "Đã ghi nhận."

        # 2) nếu không ở flow: thử khởi động flow theo intent hiện tại
        boot = self.start_flow(predicted_intent)
        if boot is not None:
            return boot

        # 3) theo dõi câu hỏi đuôi (followup) cho intent vừa trả lời
        if self.pending_followup:
            fw = self.flows[self.pending_followup].get("followup", {})
            kws: List[str] = fw.get("keywords", [])
            if any(k in user_text.lower() for k in kws):
                ans = fw.get("reply")
                self.pending_followup = None
                return ans
            self.pending_followup = None

        # 4) nhường cho responses mặc định theo intent
        return None
