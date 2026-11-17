"use strict";

const canvas = document.querySelector("#world");
const context = canvas.getContext("2d", { alpha: false });
const generation = document.querySelector("#generation");
const population = document.querySelector("#population");
const backendBadge = document.querySelector("#backendBadge");
const parsedCommand = document.querySelector("#parsedCommand");
const statusMessage = document.querySelector("#statusMessage");
const conversation = document.querySelector("#conversation");
const speed = document.querySelector("#speed");
const activity = document.querySelector("#activity");
const activityTitle = document.querySelector("#activityTitle");
const activityDetail = document.querySelector("#activityDetail");
const activityElapsed = document.querySelector("#activityElapsed");

let latestState = null;
let runTimer = null;
let runActive = false;
let pendingCommand = null;
let busySince = null;
let connected = false;
let feedbackError = null;
let pausedWhileWorking = false;
let conversationEpoch = null;
const renderedConversationIds = new Set();
const pendingUserBubbles = [];

function isBusy() {
  return pendingCommand !== null || Boolean(latestState?.queue_pending);
}

function updateActivity() {
  const busy = isBusy();
  if (busy && busySince === null) busySince = performance.now();
  if (!busy) busySince = null;
  const elapsed = busySince === null ? 0 : (performance.now() - busySince) / 1000;
  let title = runActive ? "Running" : "Ready";
  let detail = runActive ? "The next step starts after the selected pause." : "Choose an action or send a message.";
  let mode = "idle";
  if (!connected) {
    mode = "error";
    title = latestState ? "Connection interrupted" : "Connecting to the world…";
    detail = "Checking the connection. The last board stays visible.";
  } else if (busy) {
    mode = "busy";
    const kind = pendingCommand?.command.type;
    title = pendingCommand?.sending ? "Sending request…"
      : kind === "tick" ? "Advancing the world…"
      : kind === "say" ? "Qwen is responding…"
      : kind ? "Qwen is updating the world…" : "Qwen is working…";
    detail = pausedWhileWorking ? "Paused after this request. The current work will finish."
      : elapsed >= 15 ? "Still waiting for Qwen. Showing the last completed board."
      : "This takes a few seconds. Showing the last completed board.";
  } else if (feedbackError) {
    mode = "error";
    title = "Request not completed";
    detail = feedbackError;
  }
  activity.dataset.state = mode;
  // Avoid re-announcing unchanged text on every elapsed-time update.
  if (activityTitle.textContent !== title) activityTitle.textContent = title;
  if (activityDetail.textContent !== detail) activityDetail.textContent = detail;
  activityElapsed.textContent = busy && connected ? `${elapsed.toFixed(1)}s` : "";
  document.querySelector(".canvas-frame").setAttribute("aria-busy", String(busy));
  canvas.setAttribute("aria-disabled", String(busy || !connected));
  for (const id of ["stepButton", "clearButton", "placeButton", "pattern", "sendButton"]) {
    document.getElementById(id).disabled = busy || !connected;
  }
  document.querySelector("#runButton").disabled = busy || !connected || runActive;
  document.querySelector("#pauseButton").disabled = !runActive;
  document.querySelector("#sendButton").textContent = busy && pendingCommand?.command.type === "say"
    ? "Qwen is responding…" : "Send to Qwen";
}

function drawSnapshot(state) {
  const width = state.width;
  const height = state.height;
  const cellWidth = canvas.width / width;
  const cellHeight = canvas.height / height;
  context.fillStyle = "#080b10";
  context.fillRect(0, 0, canvas.width, canvas.height);

  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const alive = state.cells[row * width + column] === 1;
      context.fillStyle = alive ? "#b9ff66" : "#111720";
      context.fillRect(
        column * cellWidth + 1,
        row * cellHeight + 1,
        Math.max(1, cellWidth - 2),
        Math.max(1, cellHeight - 2),
      );
    }
  }

  generation.textContent = state.generation;
  population.textContent = state.live_cells;
  backendBadge.textContent = state.backend;
  backendBadge.classList.toggle("mock", state.backend.startsWith("mock"));
  statusMessage.textContent = feedbackError || state.message || "Neural state received";
  if (state.parsed_command) parsedCommand.textContent = state.parsed_command;
  syncConversation(state.conversation || [], state.conversation_epoch ?? 0);
  if (pendingCommand?.offset !== null && pendingCommand?.offset !== undefined
      && state.completed_offset >= pendingCommand.offset) {
    const finished = pendingCommand;
    pendingCommand = null;
    if (["REJECTED", "ERROR", "STOPPED"].includes(state.parsed_command)) {
      feedbackError = state.message;
      pause();
    } else if (finished.command.type === "tick" && state.generation <= finished.generation) {
      feedbackError = "Qwen finished without advancing the board. Try again.";
      pause();
    }
  }
  updateActivity();
  scheduleRun();
}

async function refreshState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`state request failed: ${response.status}`);
    const state = await response.json();
    // Polls can overlap. A response from before Clear must never restore the
    // old chat or board after a newer snapshot has already been displayed.
    if (latestState && Number(state.event_id) < Number(latestState.event_id)) return;
    if (latestState && Number(state.completed_offset ?? 0) < Number(latestState.completed_offset ?? 0)) return;
    connected = true;
    latestState = state;
    drawSnapshot(latestState);
  } catch (error) {
    connected = false;
    pause();
    statusMessage.textContent = error.message;
    backendBadge.textContent = "Disconnected";
    updateActivity();
  }
}

function addBubble(text, kind) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${kind}`;
  bubble.textContent = text;
  conversation.appendChild(bubble);
  conversation.scrollTop = conversation.scrollHeight;
  return bubble;
}

function syncConversation(records, epoch) {
  if (conversationEpoch !== null && epoch !== conversationEpoch) {
    conversation.querySelectorAll(".bubble:not(.system)").forEach((bubble) => bubble.remove());
    renderedConversationIds.clear();
    pendingUserBubbles.length = 0;
    document.querySelector("#chatInput").value = "";
  }
  conversationEpoch = epoch;
  for (const record of records) {
    const id = String(record.id);
    if (renderedConversationIds.has(id)) continue;
    const role = record.role === "user" ? "user" : (record.kind === "error" ? "error" : "assistant");
    if (role === "user") {
      const pending = pendingUserBubbles.findIndex((item) => item.text === record.text);
      if (pending >= 0) {
        pendingUserBubbles.splice(pending, 1);
      } else {
        addBubble(record.text, role);
      }
    } else {
      addBubble(record.text, role);
    }
    renderedConversationIds.add(id);
  }
}

async function submitCommand(command, humanLabel = null) {
  if (isBusy() || !connected) return false;
  pendingCommand = { command, offset: null, sending: true, generation: Number(latestState?.generation ?? 0) };
  feedbackError = null;
  pausedWhileWorking = false;
  updateActivity();
  if (humanLabel) {
    pendingUserBubbles.push({ text: humanLabel, bubble: addBubble(humanLabel, "user") });
  }
  try {
    const response = await fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(command),
      signal: AbortSignal.timeout(15000),
    });
    const result = await response.json();
    if (!response.ok || !result.accepted) throw new Error(result.error || "Command rejected");
    pendingCommand.sending = false;
    if (Number.isInteger(result.queue_offset)) {
      pendingCommand.offset = result.queue_offset;
    } else {
      // Mock controls finish synchronously; the real bridge always gives a receipt.
      pendingCommand = null;
    }
    parsedCommand.textContent = result.parsed_command;
    statusMessage.textContent = result.message;
    updateActivity();
    await refreshState();
    return true;
  } catch (error) {
    pendingCommand = null;
    const uncertain = error instanceof TypeError || error.name === "TimeoutError";
    if (uncertain) connected = false;
    feedbackError = uncertain
      ? "Could not confirm the request. It may still run; checking the world before you retry."
      : error.message;
    pause();
    statusMessage.textContent = feedbackError;
    if (humanLabel) addBubble(feedbackError, "error");
    if (humanLabel) {
      const pending = pendingUserBubbles.findIndex((item) => item.text === humanLabel);
      if (pending >= 0) pendingUserBubbles.splice(pending, 1);
    }
    updateActivity();
    if (uncertain) await refreshState();
    return false;
  }
}

function pause(afterCurrent = false) {
  if (runTimer !== null) window.clearInterval(runTimer);
  runTimer = null;
  runActive = false;
  if (afterCurrent && isBusy()) pausedWhileWorking = true;
  updateActivity();
}

function scheduleRun() {
  if (!runActive || runTimer !== null || isBusy() || !connected) return;
  runTimer = window.setTimeout(async () => {
    runTimer = null;
    if (!runActive || isBusy() || !connected) return;
    const accepted = await submitCommand({ type: "tick", count: 1 });
    if (!accepted) {
      pause();
    }
  }, Number(speed.value));
}

function stopRunForEdit() {
  pause();
}

document.querySelector("#runButton").addEventListener("click", () => {
  pause();
  runActive = true;
  updateActivity();
  scheduleRun();
});
document.querySelector("#pauseButton").addEventListener("click", () => pause(true));
document.querySelector("#stepButton").addEventListener("click", () => {
  stopRunForEdit();
  submitCommand({ type: "tick", count: 1 });
});
document.querySelector("#clearButton").addEventListener("click", () => {
  stopRunForEdit();
  submitCommand({ type: "clear", reset_chat: true });
});
document.querySelector("#placeButton").addEventListener("click", () => {
  stopRunForEdit();
  submitCommand({
    type: "place",
    pattern: document.querySelector("#pattern").value,
    row: 1,
    column: 1,
  });
});
speed.addEventListener("change", () => {
  if (!runActive || runTimer === null) return;
  window.clearTimeout(runTimer);
  runTimer = null;
  scheduleRun();
});

document.querySelector("#chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.querySelector("#chatInput");
  const text = input.value.trim();
  if (!text || isBusy() || !connected) return;
  stopRunForEdit();
  input.value = "";
  if (!await submitCommand({ type: "say", text }, text) && !input.value) input.value = text;
});

canvas.addEventListener("click", (event) => {
  if (!latestState || isBusy() || !connected) return;
  stopRunForEdit();
  const bounds = canvas.getBoundingClientRect();
  const column = Math.floor(((event.clientX - bounds.left) / bounds.width) * latestState.width);
  const row = Math.floor(((event.clientY - bounds.top) / bounds.height) * latestState.height);
  const current = latestState.cells[row * latestState.width + column];
  submitCommand({ type: "set_cell", row, column, alive: current === 1 ? 0 : 1 });
});

window.addEventListener("beforeunload", () => pause());
window.setInterval(refreshState, 250);
window.setInterval(updateActivity, 100);
updateActivity();
refreshState();
