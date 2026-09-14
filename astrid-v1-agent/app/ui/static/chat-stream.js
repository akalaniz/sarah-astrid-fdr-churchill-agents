(() => {
  function safePrefix(text) {
    let open = null;
    for (let i = 0; i < text.length;) {
      const lineStart = i === 0 || text[i - 1] === "\n";
      const fence = lineStart ? text.slice(i).match(/^ {0,3}(`{3,}|~{3,})/) : null;
      if (open && open.kind === "fence") {
        if (fence && fence[1][0] === open.char && fence[1].length >= open.length &&
            /^[ \t]*(?:\r?\n|$)/.test(text.slice(i + fence[0].length))) {
          open = null;
          i += fence[0].length;
        } else {
          i++;
        }
        continue;
      }
      if (!open && fence) {
        open = { kind: "fence", char: fence[1][0], length: fence[1].length, start: i };
        i += fence[0].length;
        continue;
      }
      if (text[i] === "`" && (!open || open.kind === "code")) {
        const ticks = text.slice(i).match(/^`+/)[0].length;
        if (!open) open = { kind: "code", length: ticks, start: i };
        else if (open.length === ticks) open = null;
        i += ticks;
        continue;
      }
      if (open && open.kind === "code") {
        i++;
        continue;
      }
      if (open && text.startsWith(open.close, i)) {
        i += open.close.length;
        open = null;
        continue;
      }
      if (text[i] === "\\") {
        if (!open && (text[i + 1] === "[" || text[i + 1] === "(")) {
          open = { kind: "math", close: text[i + 1] === "[" ? "\\]" : "\\)", start: i };
        } else if (!open && i === text.length - 1) {
          return text.slice(0, i);
        }
        i += 2;
        continue;
      }
      if (!open && text[i] === "$") {
        const close = text[i + 1] === "$" ? "$$" : "$";
        open = { kind: "math", close, start: i };
        i += close.length;
        continue;
      }
      i++;
    }
    // Balance code only in the preview; keep unfinished equations out of MathJax.
    if (open && open.kind === "fence") return text + "\n" + open.char.repeat(open.length) + "\n";
    if (open && open.kind === "code") return text + String.fromCharCode(96).repeat(open.length);
    return open ? text.slice(0, open.start) : text;
  }

  async function readResponse(response, onText) {
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "Chat request failed.");
    }
    if (!(response.headers.get("content-type") || "").includes("text/event-stream")) {
      return response.json();
    }
    if (!response.body) throw new Error("Streaming is unavailable in this browser.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        const { value, done } = await reader.read();
        buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
        let boundary;
        while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
          const frame = buffer.slice(0, boundary.index);
          buffer = buffer.slice(boundary.index + boundary[0].length);
          let event = "";
          const data = [];
          for (const line of frame.split(/\r?\n/)) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
          }
          if (!data.length) continue;
          const payload = JSON.parse(data.join("\n"));
          if (event === "text") onText(payload.text || "");
          if (event === "done") return payload;
          if (event === "error") throw new Error(payload.detail || "The response was interrupted.");
        }
        if (done) throw new Error("The connection closed before the answer finished. Please try again.");
      }
    } finally {
      await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  }

  function createRenderer(render) {
    let latest = "";
    let timer = null;
    let running = null;
    let finished = false;
    let failure = null;
    function schedule() {
      if (timer !== null || running || finished) return;
      timer = window.setTimeout(() => {
        timer = null;
        const snapshot = latest;
        running = Promise.resolve().then(() => render(safePrefix(snapshot)))
          .catch((error) => { failure = error; })
          .finally(() => {
            running = null;
            if (latest !== snapshot) schedule();
          });
      }, 100);
    }
    return {
      update(text) {
        if (finished) return;
        latest = text;
        schedule();
      },
      async finish(text) {
        finished = true;
        window.clearTimeout(timer);
        timer = null;
        if (running) await running;
        await render(text);
        if (failure) console.warn("A streaming preview could not be rendered.", failure);
      },
    };
  }

  window.ChatStreaming = { safePrefix, readResponse, createRenderer };
})();
