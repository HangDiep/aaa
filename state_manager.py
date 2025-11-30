# state_manager.py  (FSM phiên bản khớp flows.json của bạn)
import re
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

YES = {"có","co","ok","oke","okay","yes","y","yeah","đồng ý","dong y","vâng","vang","phải","phai","ừ","ừm","um","uh","dạ","da","rồi","roi","được","duoc","dc"}
NO  = {"không","khong","ko","thôi","thoi","không cần","khong can","không có","khong co","no","chưa","chua","không đâu","khong dau","không được","khong duoc","k"}

def extract_yesno(text: str) -> Optional[bool]:
    t = text.lower().strip()
    if any(w in t for w in YES): return True
    if any(w in t for w in NO):  return False
    # mau "khong + dong tu"
    if re.search(r"\bkh(o|ô)ng\s+\w+", t): return False
    return None

def parse_vn_date_ddmmyyyy(text: str) -> Optional[datetime]:
    t = text.strip()
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", t)
    if m:
        d, mth, y = map(int, m.groups())
        try:
            return datetime(y, mth, d)
        except ValueError:
            return None
    return None

class StateManager:
    class StateManager:
        def __init__(self, flows_path=None, flows_dict=None):
            if flows_dict is not None:
                self.flows = flows_dict
            elif flows_path:
                with open(flows_path, "r", encoding="utf-8") as f:
                    self.flows = json.load(f)
            else:
                self.flows = {}
            self.active_flow = None
            self.current_state = None


    def handle(self, predicted_intent: str, user_text: str, user_id: str = "default") -> Optional[str]:
        if self.active_flow:
            return self._step(user_text)

        if predicted_intent in self.flows:
            self._enter_flow(predicted_intent)
            sdef = self._state_def()
            if sdef:
                fb = sdef.get("fallback", {})
                if "reply" in fb:
                    return fb["reply"]
            return None

        return None

    def bootstrap_by_text(self, user_text: str, user_id: str = "default") -> Optional[str]:
        if self.active_flow:
            return self._step(user_text)
        return None

    def _enter_flow(self, flow_name: str):
        self.active_flow = flow_name
        self.current_state = self.flows[flow_name]["start"]
        self.ctx = {}

    def _exit_flow(self):
        self.active_flow = None
        self.current_state = None
        self.ctx = {}

    def _flow_def(self) -> Optional[Dict[str, Any]]:
        if not self.active_flow:
            return None
        return self.flows.get(self.active_flow)

    def _state_def(self) -> Optional[Dict[str, Any]]:
        f = self._flow_def()
        if not f: return None
        return f["states"].get(self.current_state)

    def _goto(self, next_state: Optional[str]):
        if next_state == "END" or next_state is None:
            self._exit_flow()
        else:
            self.current_state = next_state

    def _step(self, user_text: str) -> Optional[str]:
        sdef = self._state_def()
        if not sdef:
            self._exit_flow()
            return None

        # YES/NO branch
        if "yes" in sdef or "no" in sdef:
            yn = extract_yesno(user_text)
            if yn is True and "yes" in sdef:
                reply = sdef["yes"].get("reply")
                self._goto(sdef["yes"].get("next"))
                return reply
            if yn is False and "no" in sdef:
                reply = sdef["no"].get("reply")
                self._goto(sdef["no"].get("next"))
                return reply
            # Không hiểu yes/no -> không trả fallback ở đây; nhường cho intents.json trả lời
            return None
# Date branch
        if "expect_date" in sdef:
            d = parse_vn_date_ddmmyyyy(user_text)
            if d:
                self.ctx["borrow_date"] = d
                due = d + timedelta(days=14)
                rep_tmpl = sdef["expect_date"].get("reply_template", "")
                reply = rep_tmpl.format(
                    borrow_date=d.strftime("%d/%m/%Y"),
                    due_date=due.strftime("%d/%m/%Y")
                )
                self._goto(sdef["expect_date"].get("next"))
                return reply
            # Không parse được ngày -> nhường cho intents.json; giữ nguyên state
            return None
        fb = sdef.get("fallback", {})
        if "reply" in fb:
            return fb["reply"]

        self._exit_flow()
        return None
    def exit_flow(self):
        self._exit_flow()
        self._fallback_count = 0