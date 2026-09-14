(() => {
  const segmenter = typeof Intl.Segmenter === "function"
    ? new Intl.Segmenter("en-US", { granularity: "sentence" }) : null;

  function sentenceEnds(text, final) {
    const ends = [];
    if (segmenter) {
      for (const { segment, index } of segmenter.segment(text)) {
        const end = index + segment.length;
        // Wait for lookahead: a period at the end of a snapshot might become 3.14.
        if (end >= text.length || !text.slice(end).trim()) continue;
        if (!/[.!?\u3002\uff01\uff1f]["'\u2019\u201d)\]]*\s*$/.test(segment)) continue;
        if (/\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\.\s*$/i.test(segment) ||
            /(?:^|\s)[A-Z]\.\s*$/.test(segment) ||
            /^\s*\d+\.\s*$/.test(segment)) continue;
        ends.push(end);
      }
    }
    // Older browsers without a sentence segmenter retain whole-answer speech.
    if (final && ends[ends.length - 1] !== text.length) ends.push(text.length);
    return ends;
  }

  function createQueue({ synthesis, createUtterance, onState = () => {} }) {
    const pending = [];
    let current = null;
    let committed = "";
    let finished = false;
    let cancelled = false;
    let started = false;

    function state() {
      onState(!cancelled && Boolean(current || pending.length || (started && !finished)));
    }

    function cancel() {
      if (cancelled) return;
      cancelled = true;
      pending.length = 0;
      const speaking = current;
      current = null;
      if (speaking) {
        speaking.onend = null;
        speaking.onerror = null;
      }
      synthesis.cancel();
      state();
    }

    function pump() {
      if (cancelled || current || !pending.length) {
        state();
        return;
      }
      try {
        const utterance = createUtterance(pending.shift());
        current = utterance; // Retain it until onend; submit only one utterance at a time.
        utterance.onend = () => {
          if (cancelled || current !== utterance) return;
          current = null;
          pump();
        };
        utterance.onerror = () => {
          if (current === utterance) cancel();
        };
        state();
        synthesis.speak(utterance);
      } catch (error) {
        cancel();
        console.warn("Speech playback is unavailable.", error);
      }
    }

    function accept(snapshot, final) {
      if (cancelled || finished) return;
      const text = String(snapshot || "");
      // Never replay a rewritten prefix that has already been queued or spoken.
      if (!text.startsWith(committed)) {
        cancel();
        return;
      }
      const remaining = text.slice(committed.length);
      let consumed = 0;
      for (const end of sentenceEnds(remaining, final)) {
        const sentence = remaining.slice(consumed, end).trim();
        if (sentence) pending.push(sentence);
        consumed = end;
      }
      committed += remaining.slice(0, consumed);
      finished = final;
      if (pending.length) started = true;
      pump();
    }

    return {
      update(text) { accept(text, false); },
      finish(text) { accept(text, true); },
      cancel,
    };
  }

  window.SpeechStreaming = { createQueue };
})();

