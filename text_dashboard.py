import json
import requests
import reflex as rx

# use VM's external IP so the browser can access it
API_BASE = "http://35.184.194.100:8000"

class State(rx.State):
    text: str = ""
    top_k: int = 3
    result: dict = {}
    history: list[dict] = []
    error: str = ""

    # explicit setters (avoid auto-setter deprecation)
    def set_text(self, v):
        # v may be a string or an event value; assign directly
        self.text = v

    def set_top_k(self, v):
        # v may come as string; coerce & clamp to [1, 10]
        try:
            n = int(str(v))
            self.top_k = max(1, min(10, n))
        except Exception:
            # ignore bad input
            pass

    def send(self):
        self.error = ""
        try:
            r = requests.post(
                f"{API_BASE}/predict/",
                json={"text": self.text, "top_k": self.top_k},
                timeout=10,
            )
            r.raise_for_status()
            self.result = r.json()
        except Exception as e:
            self.error = f"Predict failed: {e}"

    def refresh(self):
        self.error = ""
        try:
            r = requests.get(f"{API_BASE}/predictions/", timeout=10)
            r.raise_for_status()
            self.history = r.json()
        except Exception as e:
            self.error = f"Fetch history failed: {e}"


def index() -> rx.Component:
    return rx.vstack(
        rx.heading("Text Classifier Dashboard"),
        rx.hstack(
            rx.input(
                value=State.text,
                on_change=State.set_text,
                placeholder="Enter text",
            ),
            # numeric input using core input (no plugin)
            rx.input(
                type_="number",
                value=State.top_k,
                on_change=State.set_top_k,
                placeholder="top_k (1–10)",
            ),
            rx.button("Predict", on_click=State.send),
            spacing="3",
            width="100%",
        ),
        # error message (if any)
        rx.cond(
            State.error != "",
            rx.text(lambda: State.error),
        ),
        # prediction result (pretty-printed JSON as text)
        rx.cond(
            State.result != {},
            rx.vstack(
                rx.text("Prediction result:"),
                rx.text(lambda: json.dumps(State.result, indent=2)),
                align_items="start",
            ),
        ),
        # history controls + simple listing
        rx.hstack(
            rx.button("Refresh History", on_click=State.refresh),
        ),
        rx.cond(
            (State.history != []) & (State.history != None),
            rx.vstack(
                rx.text("History:"),
                rx.foreach(
                    State.history,
                    # render each item as a single JSON line of text
                    lambda row: rx.text(lambda row=row: json.dumps(row)),
                ),
                align_items="start",
                max_height="400px",
                overflow="auto",
            ),
        ),
        spacing="4",
        width="100%",
        max_width="900px",
        align_items="stretch",
    )

app = rx.App()
app.add_page(index, title="Classifier Dashboard")
