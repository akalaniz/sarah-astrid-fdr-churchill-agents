const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");
const context = { window: { setTimeout, clearTimeout }, TextDecoder, console };
vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../app/ui/static/chat-stream.js"), "utf8"), context);
const { safePrefix, readResponse, createRenderer } = context.window.ChatStreaming;
const tick = String.fromCharCode(96);
const fence = tick.repeat(3);

test("incomplete equations wait for their closing delimiter", () => {
  assert.equal(safePrefix("Start \\[E=mc"), "Start ");
  assert.equal(safePrefix("Start \\[E=mc^2\\]"), "Start \\[E=mc^2\\]");
  assert.equal(safePrefix("Start $$\\frac{a}{b}"), "Start ");
  assert.equal(safePrefix("Start $x_i$ end"), "Start $x_i$ end");
  assert.equal(safePrefix("Start \\(x_i"), "Start ");
});

test("code previews close synthetically without sending code to MathJax", () => {
  const code = fence + "python\nformula = '\\[E=mc^2\\]'";
  assert.equal(safePrefix(code), code + "\n" + fence + "\n");
  assert.equal(safePrefix(tick + "$x_i"), tick + "$x_i" + tick);
  const closed = code + "\n" + fence + "\nDone";
  assert.equal(safePrefix(closed), closed);
});

function responseFor(text, width = 1) {
  const bytes = new TextEncoder().encode(text);
  return new Response(new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += width) controller.enqueue(bytes.slice(i, i + width));
      controller.close();
    },
  }), { headers: { "Content-Type": "text/event-stream" } });
}

test("SSE handles byte-split UTF-8, CRLF, keepalives and multiple events", async () => {
  const snapshots = [];
  const stream = ": connected\r\n\r\nevent: text\r\ndata: {\"text\":\"\u03bc\"}\r\n\r\n" +
    "event: text\r\ndata: {\"text\":\"\u03bc = 1\"}\r\n\r\n" +
    "event: done\r\ndata: {\"text\":\"\u03bc = 1\",\"sources\":[]}\r\n\r\n";
  const result = await readResponse(responseFor(stream), text => snapshots.push(text));
  assert.deepEqual(snapshots, ["\u03bc", "\u03bc = 1"]);
  assert.equal(result.text, "\u03bc = 1");
});

test("an interrupted stream cannot look like a complete response", async () => {
  await assert.rejects(readResponse(responseFor('event: text\ndata: {"text":"partial"}\n\n'), () => {}),
    /before the answer finished/);
});

test("explicit stream errors surface without retrying", async () => {
  await assert.rejects(readResponse(responseFor('event: error\ndata: {"detail":"Interrupted"}\n\n'), () => {}),
    /Interrupted/);
});

test("legacy JSON command responses remain supported", async () => {
  const result = await readResponse(new Response('{"text":"command","sources":[]}',
    { headers: { "Content-Type": "application/json" } }), () => { throw new Error("Must not stream JSON"); });
  assert.equal(result.text, "command");
});

test("HTTP errors retain a useful message", async () => {
  await assert.rejects(readResponse(new Response('{"detail":"Empty message"}', { status: 400 }), () => {}),
    /Empty message/);
});

test("final rendering waits for the preview and is never overwritten later", async () => {
  const renders = [];
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const renderer = createRenderer(async text => {
    renders.push(text);
    if (text === "first") await gate;
  });
  renderer.update("first");
  await new Promise(resolve => setTimeout(resolve, 130));
  renderer.update("second");
  const finished = renderer.finish("final");
  release();
  await finished;
  await new Promise(resolve => setTimeout(resolve, 150));
  assert.deepEqual(renders, ["first", "final"]);
});

