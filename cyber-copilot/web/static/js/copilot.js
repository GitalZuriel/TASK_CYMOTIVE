/**
 * Cybersecurity Copilot — Frontend Logic
 * Handles API calls, rendering results, and UI interactions.
 */

// --- Admin Panel visibility ---
let adminVisible = true;

// --- Fetch with automatic retry for transient errors ---
async function fetchWithRetry(url, options = {}, retryConfig = {}) {
    const maxRetries = retryConfig.maxRetries ?? 3;
    const baseDelay = retryConfig.baseDelay ?? 1000;
    const maxDelay = retryConfig.maxDelay ?? 10000;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            const response = await fetch(url, options);

            // Non-retryable client errors (400, 401, 403, 404) — return immediately
            if (response.status >= 400 && response.status < 500 && response.status !== 429) {
                return response;
            }

            // Retryable: 429 (rate limit) or 5xx (server error)
            if (response.status === 429 || response.status >= 500) {
                if (attempt < maxRetries) {
                    const delay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay);
                    const jitter = Math.random() * delay * 0.5;
                    console.warn(
                        `[fetchWithRetry] HTTP ${response.status} from ${url}, ` +
                        `retry ${attempt + 1}/${maxRetries} in ${Math.round(delay + jitter)}ms`
                    );
                    await new Promise(r => setTimeout(r, delay + jitter));
                    continue;
                }
                return response;
            }

            // Success (2xx, 3xx)
            return response;

        } catch (err) {
            // Network errors (TypeError: Failed to fetch) — retryable
            if (attempt < maxRetries) {
                const delay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay);
                const jitter = Math.random() * delay * 0.5;
                console.warn(
                    `[fetchWithRetry] Network error for ${url}: ${err.message}, ` +
                    `retry ${attempt + 1}/${maxRetries} in ${Math.round(delay + jitter)}ms`
                );
                await new Promise(r => setTimeout(r, delay + jitter));
                continue;
            }
            throw err;
        }
    }
}

// --- i18n Translations ---
const TRANSLATIONS = {
    en: {
        brand_title: 'Cyber Copilot',
        brand_subtitle: 'Automotive Incident Analysis',
        status_ready: 'Pipeline Ready',
        status_incidents: 'Loading incidents...',
        section_incident_report: 'Incident Report',
        btn_example: 'For Example',
        textarea_placeholder: 'Paste or type an automotive cybersecurity incident report here...\n\nExample: Anomalous CAN frames detected on the powertrain bus originating from the OBD-II port. Arbitration ID 0x130 targeting EBCM at 10ms intervals...',
        btn_analyze: 'Analyze Incident',
        btn_analyzing: 'Analyzing...',
        loading_text: 'Analyzing Incident Report...',
        step_preprocess: 'Preprocessing & entity extraction',
        step_queryrewrite: 'LLM query rewrite (symptoms → technical terms)',
        step_retrieval: 'Hybrid retrieval (semantic + BM25)',
        step_rerank: 'Cross-encoder reranking',
        step_summary: 'LLM incident summary',
        step_mitigation: 'LLM mitigation planning',
        stat_latency: 'Latency',
        stat_model: 'Model',
        stat_tokens: 'Tokens',
        stat_cost: 'Cost',
        empty_title: 'Ready to Analyze',
        empty_desc: 'Enter an automotive cybersecurity incident report above, or click one of the example buttons to get started.',
        footer_text: 'Cybersecurity Copilot PoC \u2014 Hybrid RAG | Cross-Encoder Reranking | Multi-Model LLM',
        card_summary: 'Incident Summary',
        card_mitigation: 'Mitigation Plan',
        card_similar: 'Similar Past Incidents',
        card_entities: 'Extracted Entities',
        card_clarification: 'Clarification Needed',
        card_error: 'Error',
        th_id: 'ID',
        th_title: 'Title',
        th_severity: 'Severity',
        th_relevance: 'Relevance',
        th_source: 'Source',
        entity_cve: 'CVE IDs',
        entity_ip: 'IP Addresses',
        entity_ecu: 'ECU / Systems',
        entity_attack: 'Attack Indicators',
        entity_none: 'No entities matched our current patterns (CVE/IP/ECU/attack terms).',
        entity_none_tip: 'Tip: adding ECU name (e.g., TCU) or indicator (e.g., spoofing) will improve extraction.',
        error_check: 'Check that your API key is configured in .env and the model is available.',
        card_retrieval_metrics: 'Retrieval Quality Metrics',
        rm_metric: 'Metric',
        rm_value: 'Result',
        rm_meaning: 'Meaning',
        rm_precision: 'Precision@k',
        rm_recall: 'Recall@k',
        rm_mrr: 'MRR',
        rm_ndcg: 'nDCG@k',
        rm_precision_desc: 'How many of the k results are actually relevant?',
        rm_recall_desc: 'Did we find at least one relevant document in Top k?',
        rm_mrr_desc: 'At what position did the first relevant document appear?',
        rm_ndcg_desc: 'How good is the ranking order by relevance?',
        rm_retrieved: 'Retrieved',
        rm_expected: 'Expected',
        rm_ground_truth: 'Ground Truth',
        rm_no_ground_truth: 'Metrics unavailable — no ground truth defined for this incident.',
        rm_no_ground_truth_hint: 'Retrieval metrics (Precision, Recall, MRR, nDCG) require a labeled benchmark incident with expected results.',
        badge_structured: 'Structured',
        confidence_high: 'High',
        confidence_medium: 'Medium',
        confidence_low: 'Low',
        section_overview: 'Incident Overview',
        section_severity: 'Severity',
        section_attack_vector: 'Attack Vector',
        section_affected_systems: 'Affected Systems',
        section_key_indicators: 'Key Indicators',
        section_timeline: 'Timeline',
        section_immediate: 'Immediate Actions (24h)',
        section_short_term: 'Short-term Actions (1-7d)',
        section_long_term: 'Long-term Recommendations',
        section_standards: 'Related Standards',
        grounding_label: 'Grounding',
        low_confidence_warning: 'Low confidence detected in',
        btn_admin: 'Admin',
        admin_section_title: 'Admin Panel — Pipeline Diagnostics',
        card_extracted_fields: 'Extracted Fields',
        ef_field: 'Field',
        ef_value: 'Value',
        ef_confidence: 'Confidence',
        ef_source: 'Source',
        card_warnings: 'Pipeline Warnings',
        admin_detailed_stats: 'Detailed Pipeline Stats',
        admin_stat_summary_tokens: 'Summary Tokens',
        admin_stat_summary_latency: 'Summary Latency',
        admin_stat_mitigation_tokens: 'Mitigation Tokens',
        admin_stat_mitigation_latency: 'Mitigation Latency',
        // Progressive mitigation
        needs_input_title: 'Incomplete Information Detected',
        needs_input_desc: 'The report is missing key details that affect analysis accuracy. How would you like to proceed?',
        btn_complete_details: 'Complete Details',
        btn_analyze_as_is: 'Analyze As-Is',
        partial_banner: 'Partial Analysis — Only immediate actions shown. Provide additional details for full mitigation plan.',
        complete_form_title: 'Provide Additional Details',
        complete_form_placeholder: 'Add the missing information here (e.g., affected ECU, attack symptoms, timeline, severity)...',
        btn_submit_details: 'Re-Analyze with Details',
        btn_cancel_details: 'Cancel',
        loading_complete: 'Analysis Complete',
        step_validate: 'Input completeness check',
        ctx_in_context: 'Used in analysis',
        ctx_reference: 'Reference only',
        thread_system: 'System Identified',
        thread_analyst: 'Your Details',
    }
};

function t(key) {
    return TRANSLATIONS['en'][key] || key;
}

// --- Example Incidents ---
// Each example has: text (the report) + expected_ids (ground truth for eval metrics).
let activeExpectedIds = null;  // set when an example is loaded; sent to API for metrics

const EXAMPLES = {
    can_anomaly: {
        text: `seeing strange behavior from few delivery vans after maintenance visit.\ndrivers reporting random dashboard warnings and occasional engine limp mode.\nchecked logs — burst of CAN messages coming from unknown ID.\nonly vehicles that had external diagnostic tool connected yesterday are affected.\nmight be unauthorized CAN activity.`,
        expected_ids: ["INC-002", "INC-006"],
    },
    ota_suspicion: {
        text: `drivers reported intermittent loss of connectivity and delayed telemetry data after recent vehicle update.\nno diagnostic trouble codes were triggered.\nnetwork logs show irregular communication bursts but destination endpoints appear legitimate.\nissue started gradually over several days and affects only part of the fleet.\npossible misconfiguration or unintended software behavior.`,
        expected_ids: ["INC-003", "INC-015"],
    },
    keyless_complaints: {
        text: `insurance partners reporting spike in theft cases.\nowners say vehicles unlocked without key interaction.\nno forced entry signs.\nall affected models use same keyless entry platform revision.\nlooks like wireless relay or similar attack.`,
        expected_ids: ["INC-009", "INC-013"],
    },
    supplier_firmware: {
        text: `during audit found ECU running firmware hash not matching approved build.\nsupplier claims no deployment happened.\nsecure boot logs partially missing.\ndevice had physical access during testing phase last month.\npossible firmware tampering.`,
        expected_ids: ["INC-004", "INC-015"],
    },
    telematics_exposure: {
        text: `noticed repeated API calls pulling vehicle location data without authentication.\nrequests coming through telematics backend endpoint.\nrate increasing over last 48h.\nfleet tracking information potentially exposed.`,
        expected_ids: ["INC-001", "INC-006"],
    },
    charging_issue: {
        text: `field engineers report EV charging sessions failing intermittently.\nanalysis shows unexpected certificates presented during handshake.\nproblem only occurs at specific public charging stations.\nsuspecting communication interception between vehicle and charger.`,
        expected_ids: ["INC-012", "INC-003"],
    },
    adas_anomaly: {
        text: `autonomous test vehicle misidentified road signs during controlled run.\ncamera input normal but perception output inconsistent.\nteam suspects manipulated visual pattern placed near road.\nADAS stack behaved incorrectly under specific visual trigger.`,
        expected_ids: ["INC-011", "INC-008"],
    },
    gps_irregularity: {
        text: `navigation systems in several taxis suddenly showed same incorrect position.\nvehicles physically far apart but reporting identical coordinates.\nissue lasted around 20 minutes then disappeared.\nlikely external signal interference.`,
        expected_ids: ["INC-008", "INC-005"],
    },
    workshop_risk: {
        text: `third-party workshop installed aftermarket fleet dongles last week.\nsince then vehicles sending abnormal diagnostic responses.\nUDS sessions remain open longer than expected.\npossible misuse of diagnostic access.`,
        expected_ids: ["INC-006", "INC-002"],
    },
    gateway_behavior: {
        text: `internal monitoring flagged unusual routing between vehicle networks.\ngateway started forwarding traffic that normally should be blocked.\nfirewall rules appear modified.\nno official configuration change recorded.\ninvestigation ongoing.`,
        expected_ids: ["INC-004", "INC-010"],
    },
    short: {
        text: `ecu hacked help`,
        expected_ids: null,
    },
};

// --- Loading Animation (step-by-step control) ---
const LOADING_STEPS = ['step-preprocess', 'step-validate', 'step-queryrewrite', 'step-retrieval', 'step-rerank', 'step-summary', 'step-mitigation'];
let loadingInterval = null;
let _loadingStepIndex = 0;

function showLoading() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.add('active');
    _loadingStepIndex = 0;

    // Reset all steps
    LOADING_STEPS.forEach(id => {
        const el = document.getElementById(id);
        el.className = 'loading-step';
        el.querySelector('i').className = 'fas fa-circle';
    });

    // Start first step
    _markStepActive(0);
}

function _markStepDone(index) {
    if (index < 0 || index >= LOADING_STEPS.length) return;
    const el = document.getElementById(LOADING_STEPS[index]);
    el.className = 'loading-step done';
    el.querySelector('i').className = 'fas fa-check-circle';
}

function _markStepActive(index) {
    if (index < 0 || index >= LOADING_STEPS.length) return;
    const el = document.getElementById(LOADING_STEPS[index]);
    el.className = 'loading-step active';
    el.querySelector('i').className = 'fas fa-spinner fa-spin';
}

function _markStepFailed(index) {
    if (index < 0 || index >= LOADING_STEPS.length) return;
    const el = document.getElementById(LOADING_STEPS[index]);
    el.className = 'loading-step';
    el.querySelector('i').className = 'fas fa-exclamation-circle';
    el.style.color = 'var(--accent-orange)';
}

function advanceLoadingTo(stepId) {
    const targetIndex = LOADING_STEPS.indexOf(stepId);
    if (targetIndex < 0) return;
    // Mark all previous steps as done
    for (let i = 0; i < targetIndex; i++) {
        _markStepDone(i);
    }
    _markStepActive(targetIndex);
    _loadingStepIndex = targetIndex;
}

function startAutoLoading(fromStepId) {
    // Auto-animate remaining steps from a given step onward
    const startIndex = LOADING_STEPS.indexOf(fromStepId);
    if (startIndex < 0) return;
    let current = startIndex;

    loadingInterval = setInterval(() => {
        if (current > startIndex) {
            _markStepDone(current - 1);
        }
        if (current < LOADING_STEPS.length) {
            _markStepActive(current);
        }
        current++;
        if (current > LOADING_STEPS.length + 1) current = LOADING_STEPS.length;
    }, 2000);
}

function hideLoading() {
    clearInterval(loadingInterval);
    document.getElementById('loadingOverlay').classList.remove('active');
}

// --- Load Example (cycles through examples) ---
const EXAMPLE_KEYS = Object.keys(EXAMPLES);
let exampleIndex = -1;

function loadNextExample() {
    exampleIndex = (exampleIndex + 1) % EXAMPLE_KEYS.length;
    const key = EXAMPLE_KEYS[exampleIndex];
    const example = EXAMPLES[key];
    document.getElementById('reportInput').value = example.text;
    activeExpectedIds = example.expected_ids || null;
    document.getElementById('reportInput').focus();
}

function loadPrevExample() {
    if (exampleIndex <= 0) exampleIndex = EXAMPLE_KEYS.length;
    exampleIndex = (exampleIndex - 1) % EXAMPLE_KEYS.length;
    const key = EXAMPLE_KEYS[exampleIndex];
    const example = EXAMPLES[key];
    document.getElementById('reportInput').value = example.text;
    activeExpectedIds = example.expected_ids || null;
    document.getElementById('reportInput').focus();
}

// --- Progressive Mitigation State ---
let _lastAnalysisData = null;   // cached full analysis response
let _lastPrecheckData = null;   // cached precheck response (when validation fails)
let _lastPrecheckReport = null; // original report text for re-analysis
let _conversationThread = [];   // [{role: 'system'|'user', text: '...'}]

function renderDecisionCard(data) {
    const sections = (data.low_confidence_sections || []).join(', ');
    const clarText = data.auto_clarification ? renderMarkdown(data.auto_clarification) : '';

    return `
    <div class="result-card full-width fade-in" style="border-color: rgba(245,158,11,0.4);">
        <div class="card-header-cyber" style="background: linear-gradient(90deg, rgba(245,158,11,0.1), transparent); color: var(--accent-orange);">
            <i class="fas fa-exclamation-triangle"></i> ${t('needs_input_title')}
        </div>
        <div class="card-body-cyber">
            <p style="margin-bottom: 12px;">${t('needs_input_desc')}</p>
            ${sections ? `<p style="font-size:13px;color:var(--text-muted);margin-bottom:16px;"><i class="fas fa-chart-line"></i> ${t('low_confidence_warning')}: <strong>${sections}</strong></p>` : ''}
            ${clarText ? `<div style="background:var(--bg-input);border:1px solid var(--border-color);border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:14px;">${clarText}</div>` : ''}
            <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <button class="btn-decision btn-complete" onclick="showCompletionForm()">
                    <i class="fas fa-edit"></i> ${t('btn_complete_details')}
                </button>
                <button class="btn-decision btn-as-is" onclick="analyzeAsIs()">
                    <i class="fas fa-forward"></i> ${t('btn_analyze_as_is')}
                </button>
            </div>
        </div>
    </div>`;
}

function renderPartialBanner() {
    return `
    <div class="partial-banner fade-in">
        <i class="fas fa-info-circle"></i> ${t('partial_banner')}
    </div>`;
}

function renderCompletionForm() {
    return `
    <div class="result-card full-width fade-in" id="completionFormCard" style="border-color: rgba(6,182,212,0.4);">
        <div class="card-header-cyber" style="background: linear-gradient(90deg, rgba(6,182,212,0.1), transparent); color: var(--accent-cyan);">
            <i class="fas fa-edit"></i> ${t('complete_form_title')}
        </div>
        <div class="card-body-cyber">
            <textarea id="additionalContext" rows="5" class="completion-textarea"
                placeholder="${t('complete_form_placeholder')}"></textarea>
            <div style="display:flex;gap:12px;margin-top:12px;">
                <button class="btn-decision btn-complete" onclick="submitAdditionalDetails()">
                    <i class="fas fa-redo"></i> ${t('btn_submit_details')}
                </button>
                <button class="btn-decision btn-as-is" onclick="cancelCompletion()">
                    <i class="fas fa-times"></i> ${t('btn_cancel_details')}
                </button>
            </div>
        </div>
    </div>`;
}

function showCompletionForm() {
    const clarArea = document.getElementById('clarificationArea');
    clarArea.innerHTML = renderCompletionForm();
    document.getElementById('additionalContext').focus();
}

function cancelCompletion() {
    if (_lastAnalysisData) {
        renderResults(_lastAnalysisData);
    } else if (_lastPrecheckData) {
        // Go back to decision card
        const clarArea = document.getElementById('clarificationArea');
        clarArea.innerHTML = renderDecisionCard({
            low_confidence_sections: [],
            auto_clarification: _lastPrecheckData.clarification || '',
            analysis_mode: 'needs_input',
        });
    }
}

async function analyzeAsIs() {
    // Case 1: We already have full analysis data — just show partial
    if (_lastAnalysisData) {
        const data = { ..._lastAnalysisData, _forcePartial: true };
        renderResults(data);
        return;
    }

    // Case 2: We only have precheck — run full analysis and force partial
    if (_lastPrecheckReport) {
        const report = _lastPrecheckReport;
        const model = document.getElementById('modelSelect').value;
        const btn = document.getElementById('analyzeBtn');

        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> ${t('btn_analyzing')}`;
        showLoading();
        advanceLoadingTo('step-retrieval');
        _markStepDone(0); // preprocess
        _markStepDone(1); // validate (skipped)
        _markStepDone(2); // query rewrite (skipped — happens server-side)
        startAutoLoading('step-retrieval');

        try {
            const payload = { report, model };
            if (activeExpectedIds) payload.expected_ids = activeExpectedIds;
            const response = await fetchWithRetry('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Analysis failed');

            _lastAnalysisData = data;
            renderResults({ ...data, _forcePartial: true });
        } catch (err) {
            renderError(err.message);
        } finally {
            hideLoading();
            btn.disabled = false;
            btn.innerHTML = `<i class="fas fa-search-plus"></i> <span data-i18n="btn_analyze">${t('btn_analyze')}</span>`;
        }
    }
}

async function submitAdditionalDetails() {
    const additional = document.getElementById('additionalContext').value.trim();
    if (!additional) return;

    const report = _lastPrecheckReport || document.getElementById('reportInput').value.trim();
    const model = document.getElementById('modelSelect').value;
    const btn = document.getElementById('analyzeBtn');

    // Save user details to conversation thread
    _conversationThread.push({ role: 'user', text: additional });

    // Merge original report with analyst-supplied details
    const enrichedReport = report + '\n\n--- Analyst-Supplied Details ---\n' + additional;

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> ${t('btn_analyzing')}`;
    showLoading();
    // Skip precheck for enriched reports — go straight to full analysis
    advanceLoadingTo('step-retrieval');
    _markStepDone(0); // preprocess
    _markStepDone(1); // validate
    _markStepDone(2); // query rewrite (happens server-side)
    startAutoLoading('step-retrieval');

    try {
        const enrichedPayload = { report: enrichedReport, model };
        if (activeExpectedIds) enrichedPayload.expected_ids = activeExpectedIds;
        const response = await fetchWithRetry('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(enrichedPayload),
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Analysis failed');

        _lastAnalysisData = data;
        _lastPrecheckData = null;
        _lastPrecheckReport = null;
        renderResults(data);

    } catch (err) {
        renderError(err.message);
    } finally {
        hideLoading();
        btn.disabled = false;
        btn.innerHTML = `<i class="fas fa-search-plus"></i> <span data-i18n="btn_analyze">${t('btn_analyze')}</span>`;
    }
}

// --- Main Analysis (with precheck validation) ---
async function analyzeReport() {
    const report = document.getElementById('reportInput').value.trim();
    if (!report) return;

    _conversationThread = []; // reset thread for new analysis

    const model = document.getElementById('modelSelect').value;
    const btn = document.getElementById('analyzeBtn');

    // Disable button
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> ${t('btn_analyzing')}`;

    showLoading();

    try {
        // Phase 1: Preprocessing
        advanceLoadingTo('step-preprocess');
        const precheckResp = await fetchWithRetry('/api/precheck', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ report }),
        });
        const precheckData = await precheckResp.json();
        if (!precheckResp.ok) throw new Error(precheckData.error || 'Precheck failed');

        _markStepDone(0);  // preprocess done

        // Phase 2: Validation check
        advanceLoadingTo('step-validate');
        await new Promise(r => setTimeout(r, 400));  // brief pause for visual

        if (precheckData.status === 'needs_input') {
            // Validation failed — stop loading, show decision
            _markStepFailed(1);  // validate = warning
            await new Promise(r => setTimeout(r, 600));
            hideLoading();
            _lastAnalysisData = null;
            _lastPrecheckData = precheckData;
            _lastPrecheckReport = report;

            // Save system clarification to thread
            if (precheckData.clarification) {
                _conversationThread.push({ role: 'system', text: precheckData.clarification });
            }

            // Show decision card with precheck info
            const clarArea = document.getElementById('clarificationArea');
            const emptyState = document.getElementById('emptyState');
            if (emptyState) emptyState.style.display = 'none';
            clarArea.innerHTML = renderDecisionCard({
                low_confidence_sections: [],
                auto_clarification: precheckData.clarification || '',
                analysis_mode: 'needs_input',
            });
            document.getElementById('resultsArea').innerHTML = '';
            btn.disabled = false;
            btn.innerHTML = `<i class="fas fa-search-plus"></i> <span data-i18n="btn_analyze">${t('btn_analyze')}</span>`;
            return;
        }

        _markStepDone(1);  // validate passed

        // Phase 3: Full analysis (retrieval → LLM)
        startAutoLoading('step-retrieval');
        const analyzePayload = { report, model };
        if (activeExpectedIds) analyzePayload.expected_ids = activeExpectedIds;
        const response = await fetchWithRetry('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(analyzePayload),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }

        _lastAnalysisData = data;
        renderResults(data);

    } catch (err) {
        renderError(err.message);
    } finally {
        hideLoading();
        btn.disabled = false;
        btn.innerHTML = `<i class="fas fa-search-plus"></i> <span data-i18n="btn_analyze">${t('btn_analyze')}</span>`;
    }
}

// --- Structured Output Renderers ---
function renderConfidenceBadge(score) {
    let color, label;
    if (score >= 0.8) { color = 'var(--accent-green, #22c55e)'; label = t('confidence_high'); }
    else if (score >= 0.5) { color = 'var(--accent-orange, #f59e0b)'; label = t('confidence_medium'); }
    else { color = 'var(--accent-red, #ef4444)'; label = t('confidence_low'); }
    const pct = Math.round(score * 100);
    return `<span class="confidence-badge" style="color:${color};">
        <i class="fas fa-chart-line"></i> ${pct}% ${label}
    </span>`;
}

function renderStructuredSummary(s) {
    let html = '';

    // Incident Overview
    html += `<div class="structured-section">
        <div class="section-header-line">
            <strong>${t('section_overview')}</strong> ${renderConfidenceBadge(s.incident_overview.confidence)}
        </div>
        <p>${s.incident_overview.text}</p>
    </div>`;

    // Severity
    html += `<div class="structured-section">
        <div class="section-header-line">
            <strong>${t('section_severity')}</strong> ${renderConfidenceBadge(s.severity.confidence)}
        </div>
        <span class="severity-badge severity-${s.severity.level}">${s.severity.level}</span>
        <span style="margin-inline-start:8px;">${s.severity.justification}</span>
    </div>`;

    // Attack Vector
    html += `<div class="structured-section">
        <div class="section-header-line">
            <strong>${t('section_attack_vector')}</strong> ${renderConfidenceBadge(s.attack_vector.confidence)}
        </div>
        <p><strong>${s.attack_vector.method}</strong>${s.attack_vector.details ? ' \u2014 ' + s.attack_vector.details : ''}</p>
    </div>`;

    // Affected Systems
    html += `<div class="structured-section">
        <div class="section-header-line">
            <strong>${t('section_affected_systems')}</strong> ${renderConfidenceBadge(s.affected_systems.confidence)}
        </div>
        <div class="entity-group">
            ${s.affected_systems.systems.map(sys =>
                `<span class="entity-tag tag-ecu"><i class="fas fa-microchip"></i> ${sys}</span>`
            ).join(' ')}
        </div>
    </div>`;

    // Key Indicators
    html += `<div class="structured-section">
        <div class="section-header-line">
            <strong>${t('section_key_indicators')}</strong> ${renderConfidenceBadge(s.key_indicators.confidence)}
        </div>
        <div class="entity-group">
            ${s.key_indicators.indicators.map(ind =>
                `<span class="entity-tag tag-attack"><i class="fas fa-exclamation-triangle"></i> ${ind}</span>`
            ).join(' ')}
        </div>
    </div>`;

    // Timeline
    html += `<div class="structured-section">
        <div class="section-header-line">
            <strong>${t('section_timeline')}</strong> ${renderConfidenceBadge(s.timeline.confidence)}
        </div>
        <ol class="timeline-list">
            ${s.timeline.events.map(e =>
                `<li>${e.step}${e.is_estimated ? ' <span class="estimated-tag">[Estimated]</span>' : ''}</li>`
            ).join('')}
        </ol>
    </div>`;

    return html;
}

function renderStructuredMitigation(m) {
    let html = '';

    function renderActionList(section, titleKey) {
        // Skip sections with empty action arrays (progressive mitigation)
        if (!section.actions || section.actions.length === 0) return '';
        let out = `<div class="structured-section">
            <div class="section-header-line">
                <strong>${t(titleKey)}</strong> ${renderConfidenceBadge(section.confidence)}
            </div>
            <ol class="action-list">`;
        for (const item of section.actions) {
            out += `<li>
                <div>${item.action}</div>
                ${item.grounding ? `<div class="grounding-note"><i class="fas fa-link"></i> ${t('grounding_label')}: ${item.grounding}</div>` : ''}
            </li>`;
        }
        out += `</ol></div>`;
        return out;
    }

    html += renderActionList(m.immediate_actions, 'section_immediate');
    html += renderActionList(m.short_term_actions, 'section_short_term');
    html += renderActionList(m.long_term_recommendations, 'section_long_term');

    // Related Standards — skip if empty
    if (m.related_standards.standards && m.related_standards.standards.length > 0) {
        html += `<div class="structured-section">
            <div class="section-header-line">
                <strong>${t('section_standards')}</strong> ${renderConfidenceBadge(m.related_standards.confidence)}
            </div>
            <ul class="standards-list">
                ${m.related_standards.standards.map(std =>
                    `<li>
                        <strong>${std.standard}</strong>${std.is_general_practice ? ' <span class="estimated-tag">[General best practice]</span>' : ''}
                        <br><span style="color:var(--text-muted);font-size:13px;">${std.relevance}</span>
                    </li>`
                ).join('')}
            </ul>
        </div>`;
    }

    return html;
}

// --- Render Results ---
function renderResults(data) {
    const area = document.getElementById('resultsArea');
    const adminArea = document.getElementById('adminResultsArea');
    const clarArea = document.getElementById('clarificationArea');
    const emptyState = document.getElementById('emptyState');
    if (emptyState) emptyState.style.display = 'none';

    // Show conversation thread if available, otherwise clear
    if (_conversationThread.length > 0) {
        clarArea.innerHTML = renderConversationThread();
    } else {
        clarArea.innerHTML = '';
    }

    // Show admin section now that we have results (if admin toggle is active)
    if (adminVisible) {
        document.getElementById('adminSection').style.display = 'block';
    }

    // --- ADMIN: Stats Bar (populate regardless of admin visibility) ---
    if (data.stats) {
        const statsBar = document.getElementById('statsBar');
        statsBar.style.display = 'flex';
        document.getElementById('statLatency').textContent = data.stats.total_latency + 's';
        document.getElementById('statModel').textContent = data.stats.model || '-';
        document.getElementById('statTokens').textContent = (data.stats.total_tokens || 0).toLocaleString();
        document.getElementById('statCost').textContent = '$' + (data.stats.total_cost || 0).toFixed(4);
    }

    // Preprocessor clarification (extracted fields) — show if no thread and has clarification
    if (_conversationThread.length === 0 && data.clarification && data.clarification.trim()) {
        clarArea.innerHTML = renderClarificationCard(data.clarification);
    }

    // Clarification-only response — no results to show
    if (data.status === 'clarification_needed') {
        area.innerHTML = '';
        adminArea.innerHTML = '';
        return;
    }

    // --- Progressive Mitigation: Decision Point ---
    const isNeedsInput = data.analysis_mode === 'needs_input' && !data._forcePartial;
    const isPartial = data._forcePartial && data.analysis_mode === 'needs_input';

    if (isNeedsInput) {
        // Show decision card — analyst chooses: complete details or analyze as-is
        clarArea.innerHTML = renderDecisionCard(data);
        // Still show summary + similar incidents (no mitigation yet)
    }

    // === USER SECTION ===
    let html = '';

    // Partial analysis banner
    if (isPartial) {
        html += renderPartialBanner();
    }

    // Summary + Mitigation grid
    html += '<div class="results-grid fade-in">';

    if (data.summary || data.structured_summary) {
        const summaryBody = (data.is_structured && data.structured_summary)
            ? renderStructuredSummary(data.structured_summary)
            : renderMarkdown(data.summary);
        const structBadge = data.is_structured
            ? `<span class="badge-structured" title="Structured JSON output"><i class="fas fa-code"></i> ${t('badge_structured')}</span>`
            : '';
        html += `
        <div class="result-card">
            <div class="card-header-cyber summary-header">
                <i class="fas fa-clipboard-list"></i> ${t('card_summary')} ${structBadge}
            </div>
            <div class="card-body-cyber" id="summaryContent">
                ${summaryBody}
            </div>
        </div>`;
    }

    // Mitigation: skip entirely for needs_input, show partial for as-is, full otherwise
    if (!isNeedsInput) {
        // Determine which mitigation data to render
        const useMitigation = isPartial && data.partial_mitigation
            ? data.partial_mitigation
            : data.structured_mitigation;
        const rawMitigation = data.mitigation;

        if (rawMitigation || useMitigation) {
            const mitigationBody = (data.is_structured && useMitigation)
                ? renderStructuredMitigation(useMitigation)
                : renderMarkdown(rawMitigation);
            const structBadge = data.is_structured
                ? `<span class="badge-structured" title="Structured JSON output"><i class="fas fa-code"></i> ${t('badge_structured')}</span>`
                : '';
            html += `
            <div class="result-card">
                <div class="card-header-cyber mitigation-header">
                    <i class="fas fa-shield-alt"></i> ${t('card_mitigation')} ${structBadge}
                </div>
                <div class="card-body-cyber" id="mitigationContent">
                    ${mitigationBody}
                </div>
            </div>`;
        }
    }

    html += '</div>';

    // Bottom row: Similar Incidents + Entities
    html += '<div class="results-grid fade-in">';

    // DEBUG: dump raw API scores to console
    if (data.similar_incidents) {
        console.table(data.similar_incidents.map(inc => ({
            id: inc.id,
            rerank_score: inc.rerank_score,
            raw_score: inc.raw_score,
        })));
    }

    if (data.similar_incidents && data.similar_incidents.length > 0) {
        html += `
        <div class="result-card">
            <div class="card-header-cyber incidents-header">
                <i class="fas fa-project-diagram"></i> ${t('card_similar')}
            </div>
            <div class="card-body-cyber" style="padding: 0;">
                <table class="incidents-table">
                    <thead>
                        <tr>
                            <th>${t('th_id')}</th>
                            <th>${t('th_title')}</th>
                            <th>${t('th_severity')}</th>
                            <th>${t('th_relevance')}</th>
                            <th>${t('th_source')}</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.similar_incidents.map(inc => `
                        <tr>
                            <td style="font-family: var(--font-mono); color: var(--accent-cyan);">${inc.id}</td>
                            <td><strong>${inc.title}</strong><br>
                                <span style="font-size: 12px; color: var(--text-muted);">${inc.attack_vector}</span>
                            </td>
                            <td><span class="severity-badge severity-${inc.severity}">${inc.severity}</span></td>
                            <td title="raw: ${inc.raw_score != null ? inc.raw_score.toFixed(4) : '?'}">
                                <div class="score-bar">
                                    <div class="score-bar-fill" style="width: ${Math.min(100, Math.max(5, inc.rerank_score))}%"></div>
                                </div>
                                <span style="font-family: var(--font-mono); font-size: 12px;">${Number(inc.rerank_score).toFixed(1)}%</span>
                            </td>
                            <td style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted);">
                                ${inc.source}
                                ${inc.in_context
                                    ? `<div style="margin-top:3px;font-size:10px;color:var(--accent-green, #22c55e);"><i class="fas fa-check-circle"></i> ${t('ctx_in_context')}</div>`
                                    : `<div style="margin-top:3px;font-size:10px;color:var(--text-muted);opacity:0.6;"><i class="fas fa-eye"></i> ${t('ctx_reference')}</div>`
                                }
                            </td>
                        </tr>`).join('')}
                    </tbody>
                </table>
            </div>
        </div>`;
    }

    if (data.entities) {
        html += `
        <div class="result-card">
            <div class="card-header-cyber entities-header">
                <i class="fas fa-tags"></i> ${t('card_entities')}
            </div>
            <div class="card-body-cyber">
                ${renderEntities(data.entities)}
            </div>
        </div>`;
    }

    html += '</div>';
    area.innerHTML = html;

    // === ADMIN SECTION ===
    let adminHtml = '';

    if (data.retrieval_metrics) {
        adminHtml += renderRetrievalMetrics(data.retrieval_metrics);
    }

    if (data.extracted_fields && Object.keys(data.extracted_fields).length > 0) {
        adminHtml += renderExtractedFieldsAdmin(data.extracted_fields);
    }

    if (data.stats) {
        adminHtml += renderDetailedStats(data.stats);
    }

    if (data.warnings && data.warnings.length > 0) {
        adminHtml += renderWarnings(data.warnings);
    }

    adminArea.innerHTML = adminHtml;
}

// --- Render Helpers ---
function renderMarkdown(text) {
    try {
        return marked.parse(text);
    } catch (e) {
        return text.replace(/\n/g, '<br>');
    }
}

function renderEntities(entities) {
    let html = '';

    if (entities.cve_ids && entities.cve_ids.length > 0) {
        html += `<div class="entity-group">
            <div class="entity-group-label">${t('entity_cve')}</div>
            ${entities.cve_ids.map(e => `<span class="entity-tag tag-cve"><i class="fas fa-bug"></i> ${e}</span>`).join(' ')}
        </div>`;
    }

    if (entities.ip_addresses && entities.ip_addresses.length > 0) {
        html += `<div class="entity-group">
            <div class="entity-group-label">${t('entity_ip')}</div>
            ${entities.ip_addresses.map(e => `<span class="entity-tag tag-ip"><i class="fas fa-network-wired"></i> ${e}</span>`).join(' ')}
        </div>`;
    }

    if (entities.ecu_names && entities.ecu_names.length > 0) {
        html += `<div class="entity-group">
            <div class="entity-group-label">${t('entity_ecu')}</div>
            ${entities.ecu_names.map(e => `<span class="entity-tag tag-ecu"><i class="fas fa-microchip"></i> ${e}</span>`).join(' ')}
        </div>`;
    }

    if (entities.attack_indicators && entities.attack_indicators.length > 0) {
        html += `<div class="entity-group">
            <div class="entity-group-label">${t('entity_attack')}</div>
            ${entities.attack_indicators.map(e => `<span class="entity-tag tag-attack"><i class="fas fa-exclamation-triangle"></i> ${e}</span>`).join(' ')}
        </div>`;
    }

    if (!html) {
        html = `<p style="color: var(--text-muted); text-align: center;">${t('entity_none')}<br><small>${t('entity_none_tip')}</small></p>`;
    }

    return html;
}

function renderRetrievalMetrics(metrics) {
    const k = metrics.k;
    const noGroundTruth = !metrics.expected_ids || metrics.expected_ids.length === 0;

    // No ground truth — show informative warning card
    if (noGroundTruth) {
        return `
        <div class="result-card full-width fade-in admin-card" style="margin-top: 20px;">
            <div class="card-header-cyber admin-card-header" style="color: var(--accent-cyan) !important;">
                <i class="fas fa-chart-bar"></i> ${t('card_retrieval_metrics')}
            </div>
            <div class="card-body-cyber" style="padding:20px;text-align:center;">
                <div style="font-size:28px;margin-bottom:8px;">&#9888;&#65039;</div>
                <div style="font-size:14px;font-weight:600;color:var(--accent-orange);">${t('rm_no_ground_truth')}</div>
                <div style="font-size:12px;color:var(--text-muted);margin-top:6px;">${t('rm_no_ground_truth_hint')}</div>
                <div style="margin-top:14px;font-size:12px;color:var(--text-muted);">
                    <strong>${t('rm_retrieved')}:</strong> <code style="color:var(--accent-cyan);">${metrics.retrieved_ids.join(', ')}</code>
                </div>
            </div>
        </div>`;
    }

    // Ground truth available — show full metrics table
    function scoreColor(val) {
        if (val >= 0.8) return 'var(--accent-green, #22c55e)';
        if (val >= 0.5) return 'var(--accent-orange, #f59e0b)';
        return 'var(--accent-red, #ef4444)';
    }
    function bar(val) {
        const pct = Math.round(val * 100);
        return `<div style="display:flex;align-items:center;gap:8px;">
            <div style="flex:1;height:6px;background:#E8E8F0;border-radius:3px;min-width:60px;">
                <div style="width:${pct}%;height:100%;background:${scoreColor(val)};border-radius:3px;transition:width 0.5s;"></div>
            </div>
            <span style="font-family:var(--font-mono);font-size:14px;font-weight:600;color:${scoreColor(val)};min-width:42px;">${val.toFixed(2)}</span>
        </div>`;
    }

    const rows = [
        { metric: t('rm_precision'), val: metrics.precision_at_k, desc: t('rm_precision_desc') },
        { metric: t('rm_recall'),    val: metrics.recall_at_k,    desc: t('rm_recall_desc') },
        { metric: t('rm_mrr'),       val: metrics.mrr,            desc: t('rm_mrr_desc') },
        { metric: t('rm_ndcg'),      val: metrics.ndcg_at_k,      desc: t('rm_ndcg_desc') },
    ];

    return `
    <div class="result-card full-width fade-in admin-card" style="margin-top: 20px;">
        <div class="card-header-cyber admin-card-header" style="color: var(--accent-cyan) !important;">
            <i class="fas fa-chart-bar"></i> ${t('card_retrieval_metrics')}
            <span style="font-size:11px;opacity:0.7;margin-inline-start:8px;">k=${k} | ${metrics.expected_ids.length} expected</span>
        </div>
        <div class="card-body-cyber" style="padding:0;">
            <table class="incidents-table" style="margin:0;">
                <thead>
                    <tr>
                        <th>${t('rm_metric')}</th>
                        <th>${t('rm_value')}</th>
                        <th>${t('rm_meaning')}</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows.map(r => `
                    <tr>
                        <td style="font-weight:600;white-space:nowrap;">${r.metric}</td>
                        <td style="min-width:150px;">${bar(r.val)}</td>
                        <td style="font-size:12px;color:var(--text-muted);">${r.desc}</td>
                    </tr>`).join('')}
                </tbody>
            </table>
            <div style="padding:12px 16px;font-size:12px;color:var(--text-muted);border-top:1px solid var(--border-color);display:flex;gap:24px;flex-wrap:wrap;">
                <span><strong>${t('rm_retrieved')}:</strong> <code style="color:var(--accent-cyan);">${metrics.retrieved_ids.join(', ')}</code></span>
                <span><strong>${t('rm_expected')}:</strong> <code style="color:var(--accent-green, #22c55e);">${metrics.expected_ids.join(', ')}</code></span>
            </div>
        </div>
    </div>`;
}

function renderClarificationCard(text) {
    return `
    <div class="clarification-card fade-in" style="margin-top: 20px;">
        <div class="card-header-cyber" style="background: linear-gradient(90deg, rgba(245,158,11,0.08), transparent); color: var(--accent-orange);">
            <i class="fas fa-question-circle"></i> ${t('card_clarification')}
        </div>
        <div class="card-body-cyber">
            ${renderMarkdown(text)}
        </div>
    </div>`;
}

function renderConversationThread() {
    if (_conversationThread.length === 0) return '';
    return `<div class="conversation-thread fade-in">
        ${_conversationThread.map(msg => {
            const isSystem = msg.role === 'system';
            const icon = isSystem ? 'fa-robot' : 'fa-user-edit';
            const label = isSystem ? t('thread_system') : t('thread_analyst');
            const cssClass = isSystem ? 'thread-msg-system' : 'thread-msg-user';
            return `<div class="thread-msg ${cssClass}">
                <div class="thread-msg-header"><i class="fas ${icon}"></i> ${label}</div>
                <div class="thread-msg-body">${renderMarkdown(msg.text)}</div>
            </div>`;
        }).join('')}
    </div>`;
}

// --- Admin Section Render Helpers ---
function renderExtractedFieldsAdmin(extractedFields) {
    const fieldLabels = {
        affected_subsystem: 'Affected Subsystem',
        attack_type: 'Attack Type',
        severity: 'Severity',
        timestamp: 'Timestamp',
        cve: 'CVE',
    };

    let rows = '';
    for (const [name, field] of Object.entries(extractedFields)) {
        const label = fieldLabels[name] || name.replace(/_/g, ' ');
        const isHigh = field.confidence === 'high';
        const confColor = isHigh ? 'var(--accent-green, #22c55e)' : 'var(--accent-orange, #f59e0b)';
        const confIcon = isHigh ? 'fa-check-circle' : 'fa-question-circle';
        rows += `
        <tr>
            <td style="font-weight:600;">${label}</td>
            <td><code style="color:var(--accent-cyan);">${field.value}</code></td>
            <td style="color:${confColor};"><i class="fas ${confIcon}"></i> ${field.confidence}</td>
            <td style="font-size:12px;color:var(--text-muted);">${field.source}</td>
        </tr>`;
    }

    return `
    <div class="result-card full-width fade-in admin-card" style="margin-top: 16px;">
        <div class="card-header-cyber admin-card-header">
            <i class="fas fa-microscope"></i> ${t('card_extracted_fields')}
        </div>
        <div class="card-body-cyber" style="padding:0;">
            <table class="incidents-table" style="margin:0;">
                <thead>
                    <tr>
                        <th>${t('ef_field')}</th>
                        <th>${t('ef_value')}</th>
                        <th>${t('ef_confidence')}</th>
                        <th>${t('ef_source')}</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    </div>`;
}

function renderDetailedStats(stats) {
    const items = [
        { label: t('admin_stat_summary_tokens'), value: (stats.summary_tokens || 0).toLocaleString() },
        { label: t('admin_stat_summary_latency'), value: (stats.summary_latency || 0).toFixed(2) + 's' },
        { label: t('admin_stat_mitigation_tokens'), value: (stats.mitigation_tokens || 0).toLocaleString() },
        { label: t('admin_stat_mitigation_latency'), value: (stats.mitigation_latency || 0).toFixed(2) + 's' },
    ];

    return `
    <div class="result-card full-width fade-in admin-card" style="margin-top: 16px;">
        <div class="card-header-cyber admin-card-header">
            <i class="fas fa-tachometer-alt"></i> ${t('admin_detailed_stats')}
        </div>
        <div class="card-body-cyber">
            <div style="display:flex;flex-wrap:wrap;gap:16px;">
                ${items.map(i => `
                <div style="background:var(--bg-input);border:1px solid var(--border-color);border-radius:8px;padding:10px 16px;min-width:160px;">
                    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);">${i.label}</div>
                    <div style="font-family:var(--font-mono);font-size:16px;font-weight:600;color:var(--accent-cyan);margin-top:4px;">${i.value}</div>
                </div>`).join('')}
            </div>
        </div>
    </div>`;
}

function renderWarnings(warnings) {
    return `
    <div class="result-card full-width fade-in admin-card" style="margin-top: 16px;">
        <div class="card-header-cyber admin-card-header" style="color:var(--accent-orange);">
            <i class="fas fa-exclamation-triangle"></i> ${t('card_warnings')}
        </div>
        <div class="card-body-cyber">
            <ul style="margin:0;padding-left:20px;">
                ${warnings.map(w => `<li style="color:var(--accent-orange);margin-bottom:4px;">${w}</li>`).join('')}
            </ul>
        </div>
    </div>`;
}

function renderError(message) {
    const area = document.getElementById('resultsArea');
    area.innerHTML = `
    <div class="result-card full-width fade-in" style="border-color: rgba(239,68,68,0.3);">
        <div class="card-header-cyber" style="background: linear-gradient(90deg, rgba(239,68,68,0.08), transparent); color: var(--accent-red);">
            <i class="fas fa-exclamation-circle"></i> ${t('card_error')}
        </div>
        <div class="card-body-cyber">
            <p>${message}</p>
            <p style="color: var(--text-muted); margin-top: 8px;">${t('error_check')}</p>
        </div>
    </div>`;
}

// Admin section is always visible — no toggle needed

// --- Keyboard Shortcut ---
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        analyzeReport();
    }
});
