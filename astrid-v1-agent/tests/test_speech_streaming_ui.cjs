const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

function harness(withSegmenter = true) {
  const context = { window: {}, Intl: withSegmenter ? Intl : {}, console };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../app/ui/static/speech-stream.js"), "utf8"), context);
  const spoken = [];
  const states = [];
  let current = null;
  let cancels = 0;
  const queue = context.window.SpeechStreaming.createQueue({
    synthesis: {
      speak(utterance) {
        assert.equal(current, null, "speech must never overlap");
        current = utterance;
        spoken.push(utterance.text);
      },
      cancel() { cancels++; current = null; },
    },
    createUtterance: text => ({ text }),
    onState: active => states.push(active),
  });
  function end() {
    assert(current, "there must be a current utterance");
    const utterance = current;
    current = null;
    utterance.onend?.();
    return utterance;
  }
  function drain() {
    for (let count = 0; current; count++) {
      assert(count < 100);
      end();
    }
  }
  return { queue, spoken, states, end, drain,
    current: () => current, cancels: () => cancels };
}

test("the first sentence starts before the final response", () => {
  const h = harness();
  h.queue.update("The first sentence is ready. The second is still arriving");
  assert.deepEqual(h.spoken, ["The first sentence is ready."]);
  assert.equal(h.states.at(-1), true);
});

test("partial sentences and a possible decimal wait for lookahead", () => {
  const h = harness();
  h.queue.update("The value is 3.");
  assert.deepEqual(h.spoken, []);
  h.queue.update("The value is 3.14 meters. Next");
  assert.deepEqual(h.spoken, ["The value is 3.14 meters."]);
});

test("common titles and initials stay with the sentence", () => {
  const h = harness();
  h.queue.update("Dr. A. Smith measured 3.14 meters. Next");
  assert.deepEqual(h.spoken, ["Dr. A. Smith measured 3.14 meters."]);
});

test("an ordered list marker is not a sentence", () => {
  const h = harness();
  h.queue.update("1. Start here. Next");
  assert.deepEqual(h.spoken, ["1. Start here."]);
});

test("snapshots queue in order, never cancel or interrupt normal playback", () => {
  const h = harness();
  h.queue.update("First sentence. Next");
  h.queue.update("First sentence. Second sentence! A third");
  assert.deepEqual(h.spoken, ["First sentence."]);
  assert.equal(h.cancels(), 0);
  h.end();
  assert.deepEqual(h.spoken, ["First sentence.", "Second sentence!"]);
  h.queue.finish("First sentence. Second sentence! A third, unfinished fragment");
  h.drain();
  assert.deepEqual(h.spoken, ["First sentence.", "Second sentence!", "A third, unfinished fragment"]);
  assert.equal(h.states.at(-1), false);
});

test("repeated snapshots and repeated finalization never repeat words", () => {
  const h = harness();
  const text = "First sentence. Second sentence. Final fragment";
  h.queue.update(text);
  h.queue.update(text);
  h.queue.finish(text);
  h.queue.finish(text);
  h.queue.update(text + " Ignored");
  h.drain();
  assert.deepEqual(h.spoken, ["First sentence.", "Second sentence.", "Final fragment"]);
});

test("every character arrival and normalized trailing whitespace retain exactly one copy", () => {
  const h = harness();
  const text = "Hello Alex. Dr. Smith measured 3.14 meters.\nNext comes the result! A final fragment";
  for (let i = 1; i <= text.length; i++) h.queue.update(text.slice(0, i).trim());
  h.queue.finish(text);
  h.drain();
  assert.equal(h.spoken.join(" "), text.replace(/\s+/g, " "));
});

test("validation-only replies stay silent until finalization", () => {
  const h = harness();
  h.queue.update("");
  assert.deepEqual(h.spoken, []);
  h.queue.finish("Checked response. The mechanism fails.");
  h.drain();
  assert.deepEqual(h.spoken, ["Checked response.", "The mechanism fails."]);
});

test("legacy JSON replies are spoken once without any previews", () => {
  const h = harness();
  h.queue.finish("Legacy answer. Its second sentence.");
  h.drain();
  assert.deepEqual(h.spoken, ["Legacy answer.", "Its second sentence."]);
});

test("stop cancels current and queued speech and ignores future stream events", () => {
  const h = harness();
  h.queue.update("First sentence. Second sentence. More");
  const lateEnd = h.current().onend;
  h.queue.cancel();
  lateEnd();
  h.queue.update("First sentence. Second sentence. More arrives.");
  h.queue.finish("First sentence. Second sentence. More arrives.");
  assert.deepEqual(h.spoken, ["First sentence."]);
  assert.equal(h.current(), null);
  assert.equal(h.states.at(-1), false);
  assert.equal(h.cancels(), 1);
});

test("stop remains available during a gap between sentences", () => {
  const h = harness();
  h.queue.update("First sentence. More");
  h.end();
  assert.equal(h.states.at(-1), true);
  h.queue.cancel();
  h.queue.finish("First sentence. More arrives.");
  assert.deepEqual(h.spoken, ["First sentence."]);
});

test("speaker errors cancel remaining speech instead of replaying it", () => {
  const h = harness();
  h.queue.finish("First sentence. Second sentence.");
  h.current().onerror({ error: "not-allowed" });
  h.drain();
  assert.deepEqual(h.spoken, ["First sentence."]);
  assert.equal(h.states.at(-1), false);
});

test("duplicate stale end callbacks cannot advance the current sentence", () => {
  const h = harness();
  h.queue.finish("First sentence. Second sentence. Third sentence.");
  const oldEnd = h.current().onend;
  h.end();
  oldEnd();
  assert.deepEqual(h.spoken, ["First sentence.", "Second sentence."]);
  h.drain();
  assert.deepEqual(h.spoken, ["First sentence.", "Second sentence.", "Third sentence."]);
});

test("a rewritten committed prefix cancels safely without reading the answer twice", () => {
  const h = harness();
  h.queue.update("First sentence. Draft");
  h.queue.finish("A revised first sentence. Final answer.");
  h.drain();
  assert.deepEqual(h.spoken, ["First sentence."]);
  assert.equal(h.cancels(), 1);
});

test("an uncommitted tail may change before finalization", () => {
  const h = harness();
  h.queue.update("First sentence. The unfinished draft");
  h.queue.finish("First sentence. The corrected conclusion.");
  h.drain();
  assert.deepEqual(h.spoken, ["First sentence.", "The corrected conclusion."]);
});

test("empty replies do not speak or enable Stop Voice", () => {
  const h = harness();
  h.queue.update("");
  h.queue.finish("");
  assert.deepEqual(h.spoken, []);
  assert.equal(h.states.at(-1), false);
});

test("browsers without Intl.Segmenter safely use whole-answer speech", () => {
  const h = harness(false);
  h.queue.update("First sentence. Next");
  assert.deepEqual(h.spoken, []);
  h.queue.finish("First sentence. Final fragment");
  h.drain();
  assert.deepEqual(h.spoken, ["First sentence. Final fragment"]);
});

test("utterance construction failures do not break response rendering", () => {
  const context = { window: {}, Intl, console: { warn() {} } };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../app/ui/static/speech-stream.js"), "utf8"), context);
  let active = true;
  const queue = context.window.SpeechStreaming.createQueue({
    synthesis: { cancel() {} },
    createUtterance() { throw new Error("No voice device"); },
    onState(value) { active = value; },
  });
  assert.doesNotThrow(() => queue.update("First sentence. Next"));
  assert.equal(active, false);
});

