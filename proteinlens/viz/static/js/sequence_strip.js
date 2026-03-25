/**
 * sequence_strip.js — Canvas-based residue activation strip with InterPro domain overlay,
 * plus text-based sequence display and MSA-style alignment view.
 *
 * Provides three visualization components:
 *   1. createSequenceStrip()  — Canvas heatmap strip with InterPro domain overlay
 *   2. createTextSequence()   — Inline text AA letters with activation-colored backgrounds
 *   3. createAlignmentView()  — Multi-sequence alignment with anchor-based horizontal shifting
 *
 * Dependencies: None (standalone vanilla JS).
 */

// ============================================================
// Shared color utility
// ============================================================

/**
 * Interpolate between white and red based on normalized activation.
 *
 * @param {number} normalizedValue - Value between 0 (no activation) and 1 (max activation).
 * @returns {string} CSS rgb() color string.
 */
function activationToColor(normalizedValue) {
    // Clamp to [0, 1]
    const t = Math.min(Math.max(normalizedValue, 0), 1);
    // White (255,255,255) -> Red (220, 38, 38)
    const r = 255;
    const g = Math.round(255 - t * (255 - 38));
    const b = Math.round(255 - t * (255 - 38));
    return `rgb(${r},${g},${b})`;
}


// ============================================================
// Shared DOM helpers for text sequence views
// ============================================================

/**
 * Build a label element with accession name and clipboard copy button.
 *
 * @param {string} accession  - UniProt accession string.
 * @param {number|null} maxAct - Per-protein max activation (for tooltip), or null.
 * @returns {HTMLElement}      - A div.seq-row-label element.
 */
function _buildLabel(accession, maxAct) {
    const label = document.createElement("div");
    label.className = "seq-row-label";

    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.textContent = "\u{1F4CB}";
    copyBtn.title = "Copy accession";
    copyBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        navigator.clipboard.writeText(accession).catch(() => {});
    });
    label.appendChild(copyBtn);

    const nameSpan = document.createElement("span");
    nameSpan.textContent = accession;
    nameSpan.title = maxAct !== null
        ? `${accession} · max: ${maxAct.toFixed(3)}`
        : accession;
    label.appendChild(nameSpan);

    return label;
}

/**
 * Build a div of colored AA letter spans, optionally prepended with gap dashes.
 *
 * Each residue is a <span class="aa"> with:
 *   - Background color: white->red by activation (normalized to maxActivation)
 *   - White text for high activation (norm > 0.6)
 *   - Tooltip: residue letter + 1-based position + activation value
 *
 * @param {string} sequence          - Amino acid sequence string.
 * @param {Array<number>} activations - Per-residue activation values.
 * @param {number} maxActivation     - Feature-level max activation for normalization.
 * @param {number} [padLeft=0]       - Number of leading dash characters for alignment.
 * @returns {HTMLElement}             - A div.seq-letters element.
 */
function _buildLettersDiv(sequence, activations, maxActivation, padLeft = 0) {
    const lettersDiv = document.createElement("div");
    lettersDiv.className = "seq-letters";

    // Leading dashes (gap characters for alignment)
    for (let i = 0; i < padLeft; i++) {
        const dash = document.createElement("span");
        dash.className = "aa aa-gap";
        dash.textContent = "-";
        lettersDiv.appendChild(dash);
    }

    // Actual residues with activation coloring
    for (let i = 0; i < sequence.length; i++) {
        const span = document.createElement("span");
        span.className = "aa";
        span.textContent = sequence[i];

        const act = activations[i] || 0;
        const norm = maxActivation > 0 ? act / maxActivation : 0;
        span.style.backgroundColor = activationToColor(norm);

        if (norm > 0.6) {
            span.style.color = "#fff";
        }

        // Tooltip: residue letter, 1-based position, raw activation value
        span.title = `${sequence[i]}${i + 1}: ${act.toFixed(3)}`;
        lettersDiv.appendChild(span);
    }

    return lettersDiv;
}


// ============================================================
// 1. Canvas-based activation strip with InterPro domain overlay
// ============================================================

/**
 * Create a sequence activation strip with hover tooltip.
 *
 * Draws a horizontal canvas where each residue gets a column colored by its
 * activation value (normalized to maxActivation). Appends the canvas and a
 * floating tooltip to the given container.
 *
 * @param {HTMLElement} container       - DOM element to append the strip to.
 * @param {Object} opts                 - Configuration object.
 * @param {string} opts.sequence        - Amino acid sequence string.
 * @param {Array<number>} opts.activations - Per-residue activation values (same length as sequence).
 * @param {number} opts.maxActivation   - Global max activation for this feature (for normalization).
 * @param {string} [opts.accession]     - UniProt accession (for fetching InterPro domain overlay).
 * @param {string} [opts.bestAnnotationName] - InterPro annotation name to highlight (optional).
 */
function createSequenceStrip(container, opts) {
    const { sequence, activations, maxActivation, accession, bestAnnotationName } = opts;
    const seqLen = sequence.length;

    // Create a wrapper div for positioning the tooltip
    const wrapper = document.createElement("div");
    wrapper.className = "sequence-strip-container";

    // --- Main activation strip canvas ---
    const canvas = document.createElement("canvas");
    const stripHeight = 30;
    const domainBarHeight = 14;
    canvas.style.width = "100%";
    canvas.style.height = (stripHeight + domainBarHeight + 2) + "px";
    wrapper.appendChild(canvas);

    // --- Tooltip ---
    const tooltip = document.createElement("div");
    tooltip.className = "strip-tooltip";
    wrapper.appendChild(tooltip);

    container.appendChild(wrapper);

    // --- Draw function (called on mount and resize) ---
    function draw() {
        const displayWidth = wrapper.clientWidth;
        if (displayWidth === 0) return; // not yet in DOM

        // Set canvas pixel dimensions (2x for retina clarity)
        const dpr = window.devicePixelRatio || 1;
        const totalHeight = stripHeight + domainBarHeight + 2;
        canvas.width = displayWidth * dpr;
        canvas.height = totalHeight * dpr;
        canvas.style.height = totalHeight + "px";

        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);

        const colWidth = displayWidth / seqLen;

        for (let i = 0; i < seqLen; i++) {
            const act = activations[i] || 0;
            const norm = maxActivation > 0 ? act / maxActivation : 0;
            ctx.fillStyle = activationToColor(norm);
            ctx.fillRect(i * colWidth, 0, colWidth + 0.5, stripHeight);
        }

        canvas._colWidth = colWidth;
        canvas._seqLen = seqLen;
    }

    requestAnimationFrame(draw);

    // --- Hover tooltip ---
    canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const colWidth = canvas._colWidth || 1;
        const idx = Math.floor(x / colWidth);
        if (idx < 0 || idx >= seqLen) {
            tooltip.style.display = "none";
            return;
        }
        const residue = sequence[idx] || "?";
        const act = activations[idx] || 0;
        tooltip.textContent = `${residue}${idx + 1}: ${act.toFixed(3)}`;
        tooltip.style.display = "block";
        tooltip.style.left = Math.min(x + 10, wrapper.clientWidth - 100) + "px";
        tooltip.style.top = "-24px";
    });

    canvas.addEventListener("mouseleave", () => {
        tooltip.style.display = "none";
    });

    // --- InterPro domain overlay (fetched async) ---
    if (accession) {
        fetchAndDrawDomains(canvas, accession, seqLen, stripHeight, bestAnnotationName);
    }
}

/**
 * Fetch InterPro domain data for an accession and draw colored bars
 * below the activation strip.
 *
 * Each domain is drawn as a colored rectangle. Domains matching the
 * bestAnnotationName are drawn in a distinct highlight color (blue).
 * Other domains use a neutral gray.
 *
 * @param {HTMLCanvasElement} canvas - The sequence strip canvas.
 * @param {string} accession        - UniProt accession to fetch domains for.
 * @param {number} seqLen           - Sequence length (for scaling positions).
 * @param {number} yOffset          - Y position to start drawing the domain bar.
 * @param {string} [bestAnnotationName] - Annotation name to highlight.
 */
async function fetchAndDrawDomains(canvas, accession, seqLen, yOffset, bestAnnotationName) {
    try {
        const res = await fetch(`/api/interpro/${accession}`);
        if (!res.ok) return;

        const data = await res.json();
        const domains = data.domains || [];
        if (domains.length === 0) return;

        const ctx = canvas.getContext("2d");
        const dpr = window.devicePixelRatio || 1;
        const displayWidth = canvas.width / dpr;
        const colWidth = displayWidth / seqLen;
        const barHeight = 10;
        const barY = yOffset + 2;

        const palette = [
            "#6c757d", "#17a2b8", "#ffc107", "#28a745", "#fd7e14",
            "#6f42c1", "#e83e8c", "#20c997",
        ];
        let colorIdx = 0;
        const nameToColor = {};

        ctx.save();
        ctx.scale(dpr, dpr);

        for (const domain of domains) {
            const name = domain.interpro_name || domain.member_accession || "Unknown";
            const start = (domain.start - 1);
            const end = domain.end;

            if (!(name in nameToColor)) {
                if (bestAnnotationName && name === bestAnnotationName) {
                    nameToColor[name] = "#0d6efd";
                } else {
                    nameToColor[name] = palette[colorIdx % palette.length];
                    colorIdx++;
                }
            }

            ctx.fillStyle = nameToColor[name];
            ctx.globalAlpha = 0.7;
            ctx.fillRect(start * colWidth, barY, (end - start) * colWidth, barHeight);
        }

        ctx.restore();
    } catch (err) {
        console.warn(`Failed to fetch InterPro domains for ${accession}:`, err);
    }
}


// ============================================================
// 2. Inline text sequence (used in per-protein detail entries)
// ============================================================

/**
 * Create a text-based sequence display where each amino acid letter is shown
 * in a monospace font with a background color proportional to its activation.
 *
 * @param {HTMLElement} container       - DOM element to append the sequence row to.
 * @param {Object} opts                 - Configuration object.
 * @param {string} opts.sequence        - Amino acid sequence string.
 * @param {Array<number>} opts.activations - Per-residue activation values.
 * @param {number} opts.maxActivation   - Global max activation for normalization.
 * @param {string} [opts.accession]     - UniProt accession (shown in label).
 * @param {number} [opts.maxAct]        - Per-protein max activation (shown in label).
 * @param {boolean} [opts.showLabel=true] - Whether to show the accession label.
 * @param {string} [opts.cssClass]      - Optional extra CSS class for the wrapper.
 */
function createTextSequence(container, opts) {
    const {
        sequence,
        activations,
        maxActivation,
        accession = "",
        maxAct = null,
        showLabel = true,
        cssClass = "",
    } = opts;

    const row = document.createElement("div");
    row.className = "seq-row";

    if (showLabel) {
        row.appendChild(_buildLabel(accession, maxAct));
    }

    row.appendChild(_buildLettersDiv(sequence, activations, maxActivation));

    const wrapper = document.createElement("div");
    wrapper.className = cssClass ? `seq-text-inline ${cssClass}` : "seq-text-inline";
    wrapper.appendChild(row);
    container.appendChild(wrapper);
}


// ============================================================
// 3. MSA-style alignment view with anchor-based shifting
// ============================================================

// Counter for generating unique radio button group names
let _alignViewCounter = 0;

/**
 * Find the anchor position for a protein's activation profile.
 *
 * "first" mode: index of the first residue with activation above 5% of the
 *               feature max (i.e. the first meaningfully activated residue).
 * "max" mode:   index of the residue with the highest activation value.
 *
 * @param {Array<number>} activations  - Per-residue activation values.
 * @param {number} maxActivation       - Feature-level max activation.
 * @param {string} mode                - "first" or "max".
 * @returns {number} 0-based index of the anchor residue, or 0 if none found.
 */
function _findAnchor(activations, maxActivation, mode) {
    if (mode === "max") {
        let bestIdx = 0;
        let bestVal = -1;
        for (let i = 0; i < activations.length; i++) {
            if ((activations[i] || 0) > bestVal) {
                bestVal = activations[i] || 0;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    // mode === "first": first residue above 5% of feature max
    const threshold = maxActivation * 0.05;
    for (let i = 0; i < activations.length; i++) {
        if ((activations[i] || 0) > threshold) return i;
    }
    return 0;
}

/**
 * Create a multi-sequence alignment view with anchor-based alignment.
 *
 * Sequences are horizontally shifted so that a chosen anchor position
 * (first activation or max activation) lines up vertically across all rows.
 * Leading positions before the sequence start are shown as dashes.
 *
 * A toggle control lets the user switch between "first activation" and
 * "max activation" alignment modes. Default is "first activation".
 *
 * @param {HTMLElement} container       - DOM element to append the alignment block to.
 * @param {Array<Object>} proteins      - Array of protein objects from top_sequences.
 * @param {number} maxActivation        - Feature-level max activation for normalization.
 * @param {number} [topN=8]             - Number of proteins to show.
 */
function createAlignmentView(container, proteins, maxActivation, topN = 8) {
    const toShow = proteins.slice(0, topN);
    if (toShow.length === 0) return;

    // Unique name for radio group so multiple alignment views don't conflict
    const radioName = `align-mode-${_alignViewCounter++}`;

    // --- Mode toggle control ---
    const controls = document.createElement("div");
    controls.className = "align-controls";
    controls.innerHTML = `
        <span class="align-controls-label">Align by:</span>
        <label><input type="radio" name="${radioName}" value="first" checked> First activation</label>
        <label><input type="radio" name="${radioName}" value="max"> Max activation</label>
    `;
    container.appendChild(controls);

    // --- Alignment block (re-rendered on mode change) ---
    const block = document.createElement("div");
    block.className = "seq-alignment";
    container.appendChild(block);

    function render(mode) {
        block.innerHTML = "";

        // Compute anchor position for each protein
        const anchors = toShow.map((p) =>
            _findAnchor(p.per_residue_activations || [], maxActivation, mode)
        );

        // The maximum anchor offset determines how much left-padding each row needs.
        // We shift every sequence right so that the largest anchor lines up.
        const maxAnchor = Math.max(...anchors);

        for (let i = 0; i < toShow.length; i++) {
            const padLeft = maxAnchor - anchors[i];
            const protein = toShow[i];

            const row = document.createElement("div");
            row.className = "seq-row";
            row.appendChild(_buildLabel(protein.accession || "", protein.max_activation ?? null));
            row.appendChild(_buildLettersDiv(
                protein.sequence || "",
                protein.per_residue_activations || [],
                maxActivation,
                padLeft
            ));
            block.appendChild(row);
        }
    }

    // Initial render with default mode
    render("first");

    // Toggle handler
    controls.addEventListener("change", (e) => {
        if (e.target.name === radioName) {
            render(e.target.value);
        }
    });
}
