const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chatForm");
const inputEl = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const micButton = document.getElementById("micButton");
const resetButton = document.getElementById("resetButton");
const speechToggleButton = document.getElementById("speechToggleButton");
const stopSpeechButton = document.getElementById("stopSpeechButton");
const voiceSelect = document.getElementById("voiceSelect");
const refreshMemoryButton = document.getElementById("refreshMemoryButton");
const refreshAgentInboxButton = document.getElementById("refreshAgentInboxButton");
const showAllAgentInboxButton = document.getElementById("showAllAgentInboxButton");
const archiveAgentInboxButton = document.getElementById("archiveAgentInboxButton");
const clearAgentInboxButton = document.getElementById("clearAgentInboxButton");
const sendAgentMessageButton = document.getElementById("sendAgentMessageButton");
const agentMessageInput = document.getElementById("agentMessageInput");
const sourcesPanel = document.getElementById("sourcesPanel");
const sourceCount = document.getElementById("sourceCount");
const memoryPanel = document.getElementById("memoryPanel");
const agentInboxPanel = document.getElementById("agentInboxPanel");
const statusLine = document.getElementById("statusLine");
const transcriptLine = document.getElementById("transcriptLine");

let recognition = null;
let isListening = false;
let speechEnabled = true;
let selectedVoice = null;
const VOICE_STORAGE_KEY = "sarahSelectedVoiceURI_v2_ava";
const PREFERRED_SARAH_VOICES = [
  /microsoft ava/i,
  /\bava\b/i,
  /microsoft jenny/i,
  /microsoft aria/i,
  /microsoft zira/i,
  /google us english female/i,
  /\bjenny\b/i,
  /\baria\b/i,
  /\bzira\b/i,
  /\bsamantha\b/i,
  /\bsusan\b/i,
  /\bjoanna\b/i,
  /\bsalli\b/i,
];

function addMessage(role, text) {
  const item = document.createElement("article");
  item.className = `message ${role}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "FDR";
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  item.append(label, body);
  messagesEl.appendChild(item);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  inputEl.disabled = isBusy;
  sendButton.textContent = isBusy ? "Sending" : "Send";
}

async function sendMessage(message, options = {}) {
  const text = message.trim();
  if (!text) return;
  addMessage("user", text);
  if (options.fromVoice) {
    transcriptLine.textContent = `You said: ${text}`;
  }
  inputEl.value = "";
  setBusy(true);
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Chat request failed.");
    const answerText = payload.text || payload.answer || "";
    addMessage("sarah", answerText);
    renderSources(payload.sources || [], []);
    speakFDR(answerText);
  } catch (error) {
    addMessage("sarah", `Request failed: ${error.message}`);
  } finally {
    setBusy(false);
    inputEl.focus();
  }
}

function renderSources(localSources, webSources) {
  const total = localSources.length + webSources.length;
  sourceCount.textContent = String(total);
  sourcesPanel.replaceChildren();
  if (total === 0) {
    sourcesPanel.textContent = "No sources for the last answer.";
    return;
  }

  localSources.forEach((source, index) => {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <strong>${index + 1}. ${escapeHtml(source.source_filename || "local source")}</strong>
      <span>${escapeHtml(source.location || "")}</span>
      <span>score ${Number(source.similarity_score || 0).toFixed(4)}</span>
      <p>${escapeHtml((source.text || "").slice(0, 420))}</p>
    `;
    sourcesPanel.appendChild(item);
  });

  webSources.forEach((source, index) => {
    const item = document.createElement("div");
    item.className = "source-item web";
    const url = escapeHtml(source.url || "#");
    item.innerHTML = `
      <strong>Web ${index + 1}. ${escapeHtml(source.title || "web source")}</strong>
      <span>${escapeHtml(source.publisher || "")} | ${escapeHtml(source.date || "")}</span>
      <a href="${url}" target="_blank" rel="noreferrer">Open source</a>
    `;
    sourcesPanel.appendChild(item);
  });
}

async function loadMemory() {
  try {
    const response = await fetch("/api/memory");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Memory request failed.");
    memoryPanel.replaceChildren();
    if (!payload.memories.length) {
      memoryPanel.textContent = "No active memories stored.";
      return;
    }
    payload.memories.forEach((memory) => {
      const item = document.createElement("div");
      item.className = "memory-item";
      item.innerHTML = `
        <strong>${escapeHtml(memory.memory_type)}</strong>
        <p>${escapeHtml(memory.text)}</p>
      `;
      memoryPanel.appendChild(item);
    });
  } catch (error) {
    memoryPanel.textContent = `Memory unavailable: ${error.message}`;
  }
}

async function loadAgentInbox(showAll = false) {
  try {
    const response = await fetch(showAll ? "/agent/inbox?all=true" : "/agent/inbox");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || payload.detail || "Agent inbox request failed.");
    agentInboxPanel.replaceChildren();
    if (!payload.messages.length) {
      agentInboxPanel.textContent = showAll ? "No inter-agent messages for FDR." : "No unread direct/reply inter-agent messages.";
      return;
    }
    payload.messages.forEach((message) => {
      const item = document.createElement("div");
      item.className = "agent-item";
      item.innerHTML = `
        <strong>${escapeHtml(message.from_agent || "agent")} | ${escapeHtml(message.subject || "Message")}</strong>
        <span>${escapeHtml(message.category || "direct")} | ${escapeHtml(message.status || "unread")} | ${escapeHtml(message.id || "")}</span>
        <p>${escapeHtml(message.body || "")}</p>
      `;
      agentInboxPanel.appendChild(item);
    });
  } catch (error) {
    agentInboxPanel.textContent = `Agent inbox unavailable: ${error.message}`;
  }
}

async function archiveAgentInbox() {
  const response = await fetch("/agent/archive_inbox", { method: "POST" });
  const payload = await response.json();
  if (!response.ok) {
    agentInboxPanel.textContent = `Archive failed: ${payload.error || payload.detail || "Agent bus error."}`;
    return;
  }
  addMessage("sarah", `Archived ${payload.archived} messages addressed to FDR.`);
  loadAgentInbox();
}

async function clearAgentInbox() {
  const confirmed = window.confirm("Clear only messages addressed to FDR from the shared bus?");
  if (!confirmed) return;
  const response = await fetch("/agent/clear_inbox_confirm", { method: "POST" });
  const payload = await response.json();
  if (!response.ok) {
    agentInboxPanel.textContent = `Clear failed: ${payload.error || payload.detail || "Agent bus error."}`;
    return;
  }
  addMessage("sarah", `Cleared ${payload.cleared} messages addressed to FDR.`);
  loadAgentInbox();
}

async function sendAgentMessage() {
  const body = agentMessageInput.value.trim();
  if (!body) return;
  sendAgentMessageButton.disabled = true;
  try {
    const response = await fetch("/agent/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ to_agent: "Churchill", subject: "Message from FDR", body }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Agent send failed.");
    agentMessageInput.value = "";
    addMessage("sarah", `Sent to Churchill as ${payload.message.id}.`);
  } catch (error) {
    addMessage("sarah", `Agent send failed: ${error.message}`);
  } finally {
    sendAgentMessageButton.disabled = false;
  }
}

async function resetConversation() {
  stopSpeaking();
  await fetch("/api/reset", { method: "POST" });
  messagesEl.replaceChildren();
  renderSources([], []);
  transcriptLine.textContent = "Microphone idle.";
  addMessage("sarah", "Active conversation reset.");
}

function setupMicrophone() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    micButton.disabled = true;
    micButton.title = "Browser speech recognition is unavailable.";
    transcriptLine.textContent = "Speech recognition is unavailable in this browser. Use Edge or Chrome.";
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onstart = () => {
    isListening = true;
    stopSpeaking();
    micButton.textContent = "Listening";
    micButton.classList.add("listening");
    transcriptLine.textContent = "Listening...";
  };
  recognition.onerror = (event) => {
    addMessage("sarah", `Microphone transcription failed: ${event.error}`);
    transcriptLine.textContent = `Microphone failed: ${event.error}`;
  };
  recognition.onend = () => {
    isListening = false;
    micButton.textContent = "Speak";
    micButton.classList.remove("listening");
    if (!inputEl.value.trim()) {
      transcriptLine.textContent = "Microphone idle.";
    }
  };
  recognition.onresult = (event) => {
    let transcript = "";
    let isFinal = false;
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      transcript += event.results[i][0].transcript;
      isFinal = isFinal || event.results[i].isFinal;
    }
    const cleanTranscript = transcript.trim();
    inputEl.value = cleanTranscript;
    transcriptLine.textContent = cleanTranscript ? `Hearing: ${cleanTranscript}` : "Listening...";
    if (isFinal && cleanTranscript) {
      sendMessage(cleanTranscript, { fromVoice: true });
    }
  };
}

function setupSpeechSynthesis() {
  if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {
    speechEnabled = false;
    speechToggleButton.disabled = true;
    stopSpeechButton.disabled = true;
    voiceSelect.disabled = true;
    voiceSelect.replaceChildren(new Option("Voice unavailable", ""));
    speechToggleButton.textContent = "Voice unavailable";
    return;
  }

  const refreshVoices = () => {
    const voices = window.speechSynthesis.getVoices();
    selectedVoice = chooseFDRVoice(voices);
    renderVoiceOptions(voices);
  };

  refreshVoices();
  window.speechSynthesis.onvoiceschanged = refreshVoices;
  voiceSelect.addEventListener("change", () => {
    const voices = window.speechSynthesis.getVoices();
    selectedVoice = voices.find((voice) => voice.voiceURI === voiceSelect.value) || selectedVoice;
    if (selectedVoice) {
      localStorage.setItem(VOICE_STORAGE_KEY, selectedVoice.voiceURI);
    }
  });
}

function chooseFDRVoice(voices) {
  if (!voices.length) return null;

  const savedVoiceURI = localStorage.getItem(VOICE_STORAGE_KEY);
  const savedVoice = voices.find((voice) => voice.voiceURI === savedVoiceURI);
  if (savedVoice) return savedVoice;

  const usEnglishVoices = voices.filter((voice) => normalizeLang(voice.lang) === "en-us");
  const englishVoices = voices.filter((voice) => normalizeLang(voice.lang).startsWith("en"));
  return (
    findPreferredVoice(usEnglishVoices) ||
    findPreferredVoice(englishVoices) ||
    usEnglishVoices.find((voice) => !looksLikeMaleVoice(voice)) ||
    englishVoices.find((voice) => !looksLikeMaleVoice(voice)) ||
    usEnglishVoices[0] ||
    englishVoices[0] ||
    voices[0]
  );
}

function findPreferredVoice(voices) {
  return voices.find((voice) => {
    const label = `${voice.name} ${voice.voiceURI}`;
    return PREFERRED_SARAH_VOICES.some((pattern) => pattern.test(label));
  });
}

function renderVoiceOptions(voices) {
  voiceSelect.replaceChildren();
  if (!voices.length) {
    voiceSelect.appendChild(new Option("Loading voices...", ""));
    return;
  }

  const sortedVoices = [...voices].sort((a, b) => {
    const aScore = voicePreferenceScore(a);
    const bScore = voicePreferenceScore(b);
    if (aScore !== bScore) return bScore - aScore;
    return a.name.localeCompare(b.name);
  });

  sortedVoices.forEach((voice) => {
    const option = new Option(`${voice.name} (${voice.lang || "unknown"})`, voice.voiceURI);
    option.selected = selectedVoice && voice.voiceURI === selectedVoice.voiceURI;
    voiceSelect.appendChild(option);
  });
}

function voicePreferenceScore(voice) {
  const label = `${voice.name} ${voice.voiceURI}`;
  let score = 0;
  if (normalizeLang(voice.lang) === "en-us") score += 100;
  if (normalizeLang(voice.lang).startsWith("en")) score += 30;
  if (PREFERRED_SARAH_VOICES.some((pattern) => pattern.test(label))) score += 80;
  if (looksLikeMaleVoice(voice)) score -= 40;
  return score;
}

function looksLikeMaleVoice(voice) {
  return /\b(guy|david|mark|george|james|daniel|ryan|thomas|male)\b/i.test(`${voice.name} ${voice.voiceURI}`);
}

function normalizeLang(lang) {
  return String(lang || "").toLowerCase();
}

function speakFDR(text) {
  if (!speechEnabled || !("speechSynthesis" in window)) return;

  stopSpeaking();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-US";
  utterance.rate = 0.96;
  utterance.pitch = 0.92;
  utterance.volume = 1;
  if (selectedVoice) {
    utterance.voice = selectedVoice;
  }
  utterance.onstart = () => {
    stopSpeechButton.disabled = false;
  };
  utterance.onend = () => {
    stopSpeechButton.disabled = true;
  };
  utterance.onerror = () => {
    stopSpeechButton.disabled = true;
  };
  window.speechSynthesis.speak(utterance);
}

function stopSpeaking() {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  stopSpeechButton.disabled = true;
}

function toggleSpeech() {
  speechEnabled = !speechEnabled;
  speechToggleButton.textContent = speechEnabled ? "Speaker on" : "Speaker off";
  speechToggleButton.setAttribute("aria-pressed", String(speechEnabled));
  if (!speechEnabled) {
    stopSpeaking();
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(inputEl.value);
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage(inputEl.value);
  }
});

micButton.addEventListener("click", () => {
  if (!recognition) return;
  if (isListening) {
    recognition.stop();
  } else {
    recognition.start();
  }
});

resetButton.addEventListener("click", resetConversation);
speechToggleButton.addEventListener("click", toggleSpeech);
stopSpeechButton.addEventListener("click", stopSpeaking);
refreshMemoryButton.addEventListener("click", loadMemory);
refreshAgentInboxButton.addEventListener("click", loadAgentInbox);
showAllAgentInboxButton.addEventListener("click", () => loadAgentInbox(true));
archiveAgentInboxButton.addEventListener("click", archiveAgentInbox);
clearAgentInboxButton.addEventListener("click", clearAgentInbox);
sendAgentMessageButton.addEventListener("click", sendAgentMessage);

fetch("/api/status")
  .then((response) => response.json())
  .then((payload) => {
    statusLine.textContent = `Localhost only | ${payload.model}`;
  })
  .catch(() => {
    statusLine.textContent = "Localhost only";
  });

setupMicrophone();
setupSpeechSynthesis();
loadMemory();
loadAgentInbox();
addMessage("sarah", "Ready.");
stopSpeechButton.disabled = true;
