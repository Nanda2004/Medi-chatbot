const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("medicalImage");
const previewWrapper = document.getElementById("previewWrapper");
const placeholderContent = document.getElementById("placeholderContent");
const imagePreview = document.getElementById("imagePreview");
const changeImageBtn = document.getElementById("changeImageBtn");
const analyzerForm = document.getElementById("analyzerForm");
const analyzeBtn = document.getElementById("analyzeBtn");
const loader = analyzeBtn.querySelector(".loader");
const btnText = analyzeBtn.querySelector(".btn-text");
const errorBanner = document.getElementById("errorBanner");
const chatStream = document.getElementById("chatStream");
const reportCard = document.getElementById("reportCard");
const urgencyBadge = document.getElementById("urgencyBadge");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatSendBtn = document.getElementById("chatSendBtn");
const chatLoader = document.querySelector(".chat-loader");
const chatBtnText = document.querySelector(".chat-btn-text");
const chatError = document.getElementById("chatError");

let selectedFile = null;

const urgencyStyles = {
  Normal: "bg-emerald-50 text-emerald-700 border border-emerald-200",
  "Mild Concern": "bg-amber-50 text-amber-700 border border-amber-200",
  Concerning: "bg-orange-50 text-orange-700 border border-orange-200",
  Urgent: "bg-rose-50 text-rose-700 border border-rose-200"
};

function toggleLoading(state) {
  analyzeBtn.disabled = state;
  loader.classList.toggle("hidden", !state);
  btnText.textContent = state ? "Analyzing" : "Analyze Image";
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
}

function clearError() {
  errorBanner.classList.add("hidden");
  errorBanner.textContent = "";
}

function showChatError(message) {
  chatError.textContent = message;
  chatError.classList.remove("hidden");
}

function clearChatError() {
  chatError.textContent = "";
  chatError.classList.add("hidden");
}

function updatePreview(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Please upload a valid image file (JPEG, PNG, etc.)");
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  imagePreview.src = url;
  previewWrapper.classList.remove("hidden");
  placeholderContent.classList.add("hidden");
  dropZone.classList.add("border-sky-400");
}

function resetPreview() {
  selectedFile = null;
  imagePreview.src = "";
  previewWrapper.classList.add("hidden");
  placeholderContent.classList.remove("hidden");
  dropZone.classList.remove("border-sky-400");
  reportCard.classList.add("hidden");
  chatStream.innerHTML = "";
  clearError();
}

function addChatBubble(role, content) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = content;
  chatStream.appendChild(bubble);
  chatStream.scrollTop = chatStream.scrollHeight;
}

function setChatLoading(state) {
  chatSendBtn.disabled = state;
  chatLoader.classList.toggle("hidden", !state);
  chatBtnText.textContent = state ? "Sending" : "Send";
}

function formatReportSection(section, heading, body) {
  const container = reportCard.querySelector(`.report-section[data-section="${section}"]`);
  if (!container) return;

  if (section === "differential_diagnoses" && (!body || body.length === 0)) {
    container.innerHTML = "";
    return;
  }

  let bodyMarkup = "";
  if (Array.isArray(body)) {
    bodyMarkup = `<ul class="list-disc pl-5 space-y-1 text-sm text-slate-600">${body.map((item) => `<li>${item}</li>`).join("")}</ul>`;
  } else {
    bodyMarkup = `<p class="section-body">${body}</p>`;
  }

  container.innerHTML = `
    <div class="section-heading">${heading}</div>
    ${bodyMarkup}
  `;
}

function renderReport(report) {
  formatReportSection("findings", "Findings", report.findings);
  formatReportSection("impression", "Impression", report.impression);
  formatReportSection(
    "differential_diagnoses",
    "Differential Diagnoses",
    report.differential_diagnoses
  );
  formatReportSection("recommendations", "Recommendations", report.recommendations);
  formatReportSection("disclaimer", "AI Safety Disclaimer", report.disclaimer);

  urgencyBadge.textContent = `Urgency: ${report.urgency_level}`;
  urgencyBadge.className = `rounded-full px-4 py-1 text-sm font-medium ${urgencyStyles[report.urgency_level] || ""}`;

  reportCard.classList.remove("hidden");
}

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("drag-active");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-active");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-active");
  const file = event.dataTransfer.files?.[0];
  if (file) {
    clearError();
    updatePreview(file);
  }
});

fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) {
    clearError();
    updatePreview(file);
  }
});

changeImageBtn.addEventListener("click", () => resetPreview());

analyzerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  if (!selectedFile) {
    showError("Please upload an image before requesting analysis.");
    return;
  }

  addChatBubble("user", "Please analyze the uploaded medical image.");
  toggleLoading(true);

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Unable to analyze the image right now.");
    }

    const data = await response.json();
    addChatBubble("assistant", "Structured report generated with cautious interpretation.");
    renderReport(data);
  } catch (error) {
    console.error(error);
    showError(error.message || "An unexpected error occurred.");
  } finally {
    toggleLoading(false);
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearChatError();

  const message = chatInput.value.trim();
  if (!message) {
    showChatError("Please enter a question or description.");
    return;
  }

  addChatBubble("user", message);
  setChatLoading(true);
  chatInput.value = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ message })
    });

    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Chat assistant is unavailable.");
    }

    const data = await response.json();
    addChatBubble("assistant", data.reply);
  } catch (error) {
    console.error(error);
    showChatError(error.message || "Unable to send your message right now.");
  } finally {
    setChatLoading(false);
  }
});

