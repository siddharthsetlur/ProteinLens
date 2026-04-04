/**
 * mol_viewer.js — 3Dmol.js wrapper for per-residue activation coloring.
 *
 * Provides functions to:
 *   1. Create a 3Dmol viewer in a container div
 *   2. Load a PDB from /api/pdb/{accession}
 *   3. Color residues white->red by activation value
 *   4. Lazy-load viewers via IntersectionObserver (only init when scrolled into view)
 *
 * WebGL context management:
 *   Browsers limit active WebGL contexts (~8-16). We track active viewers
 *   and dispose the oldest when the limit is reached.
 *
 * Dependencies: 3Dmol.js (loaded via CDN in feature.html)
 */

// Track active viewers for WebGL context management
const _activeViewers = [];
const MAX_WEBGL_CONTEXTS = 8;

/**
 * Create a 3Dmol.js viewer, load a PDB, and color residues by activation.
 *
 * @param {HTMLElement} container      - DOM element (should have fixed width/height, e.g. 400x300).
 * @param {string} accession           - UniProt accession to fetch PDB for.
 * @param {Array<number>} activations  - Per-residue activation values.
 * @param {number} maxActivation       - Feature-level max activation for normalization.
 * @returns {Promise<Object|null>}     - The 3Dmol viewer object, or null on failure.
 */
async function createMolViewer(container, accession, activations, maxActivation) {
    // Enforce WebGL context limit by disposing oldest viewer if at capacity
    if (_activeViewers.length >= MAX_WEBGL_CONTEXTS) {
        const oldest = _activeViewers.shift();
        try {
            oldest.viewer.clear();
            // Remove the canvas from the DOM to release the WebGL context
            const oldCanvas = oldest.container.querySelector("canvas");
            if (oldCanvas) oldCanvas.remove();
        } catch (e) {
            // Best-effort cleanup
        }
        // Show a placeholder indicating the viewer was recycled
        oldest.container.innerHTML = '<div class="viewer-placeholder">Scrolled out of view</div>';
    }

    try {
        // Fetch PDB data
        const res = await fetch(`/api/pdb/${accession}`);
        if (!res.ok) {
            container.innerHTML = '<div class="viewer-placeholder">No structure available</div>';
            return null;
        }
        const pdbData = await res.text();

        // Clear container and create viewer
        container.innerHTML = "";
        const viewer = $3Dmol.createViewer(container, {
            backgroundColor: "white",
            antialias: true,
        });

        viewer.addModel(pdbData, "pdb");

        // Color residues by activation: white (0) -> red (max)
        // 3Dmol uses 0-indexed residue serial numbers (resi), but PDB files
        // use 1-based. We iterate activations and set color per residue index.
        const model = viewer.getModel();
        const atoms = model.selectedAtoms({});

        // Build a color map: residue number (1-based in PDB) -> hex integer
        // 3Dmol.js colorfunc expects numeric hex colors (e.g. 0xFF2626)
        const colorMap = {};
        for (let i = 0; i < activations.length; i++) {
            const act = activations[i] || 0;
            const norm = maxActivation > 0 ? Math.min(act / maxActivation, 1) : 0;
            // White (0xFFFFFF) -> Red (0xFF2626)
            const r = 255;
            const g = Math.round(255 - norm * (255 - 38));
            const b = Math.round(255 - norm * (255 - 38));
            // PDB residue numbering is 1-based
            colorMap[i + 1] = (r << 16) | (g << 8) | b;
        }

        // Apply cartoon style with per-residue coloring
        viewer.setStyle({}, {
            cartoon: {
                colorfunc: function (atom) {
                    return colorMap[atom.resi] ?? 0xFFFFFF;
                },
            },
        });

        viewer.zoomTo();
        viewer.render();

        // Track for WebGL management
        _activeViewers.push({ viewer, container });

        return viewer;
    } catch (err) {
        console.error(`Failed to create viewer for ${accession}:`, err);
        container.innerHTML = '<div class="viewer-placeholder">Viewer error</div>';
        return null;
    }
}

/**
 * Set up lazy loading for a viewer container using IntersectionObserver.
 *
 * The viewer will only be initialized when the container scrolls into the
 * viewport (or when a parent <details> element is opened).
 *
 * @param {HTMLElement} container      - DOM element for the viewer (400x300).
 * @param {string} accession           - UniProt accession.
 * @param {Array<number>} activations  - Per-residue activation values.
 * @param {number} maxActivation       - Feature-level max activation.
 */
function lazyLoadViewer(container, accession, activations, maxActivation) {
    // Show placeholder initially
    container.innerHTML = '<div class="viewer-placeholder">Scroll to load 3D structure</div>';

    // Mark container with data attributes for re-initialization if needed
    container.dataset.accession = accession;
    container.dataset.loaded = "false";

    const observer = new IntersectionObserver(
        (entries) => {
            for (const entry of entries) {
                if (entry.isIntersecting && container.dataset.loaded === "false") {
                    container.dataset.loaded = "true";
                    container.innerHTML = '<div class="viewer-placeholder"><div class="loading-spinner"></div> Loading structure...</div>';
                    createMolViewer(container, accession, activations, maxActivation);
                    observer.unobserve(container);
                }
            }
        },
        { rootMargin: "200px" } // Start loading slightly before visible
    );

    observer.observe(container);
}

/**
 * Create a 3Dmol viewer for a motif superposition structure.
 *
 * Colors residues by per_position_flexibility: blue (rigid, 0) -> red (flexible, max).
 *
 * @param {HTMLElement} container         - DOM element for the viewer.
 * @param {string} pdbString             - PDB-format string of the mean motif structure.
 * @param {Array<number>} flexibility    - Per-position flexibility values.
 */
function createMotifViewer(container, pdbString, flexibility) {
    if (!pdbString) {
        container.innerHTML = '<div class="viewer-placeholder">No motif structure</div>';
        return;
    }

    container.innerHTML = "";
    const viewer = $3Dmol.createViewer(container, {
        backgroundColor: "white",
        antialias: true,
    });

    // Inject CONECT records so 3Dmol draws bonds between consecutive CAs
    const atomCount = (pdbString.match(/^ATOM/gm) || []).length;
    let conect = "";
    for (let i = 1; i < atomCount; i++) {
        conect += `\nCONECT${String(i).padStart(5)}${String(i + 1).padStart(5)}`;
    }
    const pdbWithConect = pdbString.replace(/\nEND/, conect + "\nEND");

    viewer.addModel(pdbWithConect, "pdb");

    // Color by flexibility: blue (rigid) -> red (flexible)
    const maxFlex = Math.max(...flexibility.filter((v) => typeof v === "number"), 1);
    const colorMap = {};
    for (let i = 0; i < flexibility.length; i++) {
        const val = typeof flexibility[i] === "number" ? flexibility[i] : 0;
        const norm = maxFlex > 0 ? Math.min(val / maxFlex, 1) : 0;
        // Blue (0x0000FF) -> Red (0xFF0000)
        const r = Math.round(norm * 255);
        const g = 0;
        const b = Math.round((1 - norm) * 255);
        colorMap[i + 1] = (r << 16) | (g << 8) | b;
    }

    // CA-only PDB: spheres at each CA + stick bonds along the chain
    viewer.setStyle({}, {
        sphere: {
            radius: 0.6,
            colorfunc: function (atom) {
                return colorMap[atom.resi] ?? 0x808080;
            },
        },
        stick: {
            radius: 0.2,
            colorfunc: function (atom) {
                return colorMap[atom.resi] ?? 0x808080;
            },
        },
    });

    viewer.zoomTo();
    viewer.render();

    _activeViewers.push({ viewer, container });
}

/**
 * Kabsch-align mobile (N,3) onto target (N,3) and return the transformed mobile.
 * Pure JS implementation matching the Python pipeline's kabsch_align().
 *
 * @param {Array<Array<number>>} mobile  - Coordinates to align.
 * @param {Array<Array<number>>} target  - Reference coordinates.
 * @returns {Array<Array<number>>}       - Aligned mobile coordinates.
 */
function kabschAlign(mobile, target) {
    const n = Math.min(mobile.length, target.length);

    // Compute centroids
    const pMean = [0, 0, 0], qMean = [0, 0, 0];
    for (let i = 0; i < n; i++) {
        for (let d = 0; d < 3; d++) {
            pMean[d] += mobile[i][d];
            qMean[d] += target[i][d];
        }
    }
    for (let d = 0; d < 3; d++) { pMean[d] /= n; qMean[d] /= n; }

    // Center both
    const P = mobile.map(r => [r[0] - pMean[0], r[1] - pMean[1], r[2] - pMean[2]]);
    const Q = [];
    for (let i = 0; i < n; i++) {
        Q.push([target[i][0] - qMean[0], target[i][1] - qMean[1], target[i][2] - qMean[2]]);
    }

    // Cross-covariance H = P^T @ Q (3x3)
    const H = [[0,0,0],[0,0,0],[0,0,0]];
    for (let i = 0; i < n; i++) {
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                H[r][c] += P[i][r] * Q[i][c];
            }
        }
    }

    // SVD of 3x3 matrix via analytic method
    const { U, S, Vt } = svd3x3(H);

    // Determinant correction for reflection
    const det = det3x3(matMul3x3(transpose3x3(Vt), transpose3x3(U)));
    const sign = det < 0 ? -1 : 1;
    const signMat = [[1,0,0],[0,1,0],[0,0,sign]];

    // R = V @ sign @ U^T
    const R = matMul3x3(matMul3x3(transpose3x3(Vt), signMat), transpose3x3(U));

    // Apply: aligned = (mobile - pMean) @ R^T + qMean
    return P.map(row => {
        const out = [0, 0, 0];
        for (let c = 0; c < 3; c++) {
            for (let k = 0; k < 3; k++) {
                out[c] += row[k] * R[c][k];  // R^T transposed access
            }
            out[c] += qMean[c];
        }
        return out;
    });
}

// --- 3x3 linear algebra helpers for Kabsch ---

function transpose3x3(m) {
    return [[m[0][0],m[1][0],m[2][0]],[m[0][1],m[1][1],m[2][1]],[m[0][2],m[1][2],m[2][2]]];
}

function matMul3x3(a, b) {
    const r = [[0,0,0],[0,0,0],[0,0,0]];
    for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
            for (let k = 0; k < 3; k++)
                r[i][j] += a[i][k] * b[k][j];
    return r;
}

function det3x3(m) {
    return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
         - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
         + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
}

/**
 * SVD of a 3x3 matrix using the iterative Jacobi method.
 * Returns { U, S (diagonal values), Vt } where M = U @ diag(S) @ Vt.
 */
function svd3x3(M) {
    // Compute M^T M
    const Mt = transpose3x3(M);
    const MtM = matMul3x3(Mt, M);

    // Jacobi eigendecomposition of MtM -> eigenvalues + V
    const { eigvals, eigvecs } = jacobiEigen3x3(MtM);

    // Sort by descending eigenvalue
    const order = [0, 1, 2].sort((a, b) => eigvals[b] - eigvals[a]);
    const S = order.map(i => Math.sqrt(Math.max(eigvals[i], 0)));
    const V = [[0,0,0],[0,0,0],[0,0,0]];
    for (let r = 0; r < 3; r++)
        for (let ci = 0; ci < 3; ci++)
            V[r][ci] = eigvecs[r][order[ci]];

    // U = M V S^{-1}
    const MV = matMul3x3(M, V);
    const U = [[0,0,0],[0,0,0],[0,0,0]];
    for (let c = 0; c < 3; c++) {
        const s = S[c] > 1e-10 ? S[c] : 1;
        for (let r = 0; r < 3; r++) U[r][c] = MV[r][c] / s;
    }

    const Vt = transpose3x3(V);
    return { U, S, Vt };
}

/**
 * Jacobi eigendecomposition for a 3x3 symmetric matrix.
 */
function jacobiEigen3x3(A) {
    const a = [[A[0][0],A[0][1],A[0][2]],[A[1][0],A[1][1],A[1][2]],[A[2][0],A[2][1],A[2][2]]];
    const v = [[1,0,0],[0,1,0],[0,0,1]];

    for (let iter = 0; iter < 50; iter++) {
        // Find largest off-diagonal
        let p = 0, q = 1;
        let maxVal = Math.abs(a[0][1]);
        if (Math.abs(a[0][2]) > maxVal) { p = 0; q = 2; maxVal = Math.abs(a[0][2]); }
        if (Math.abs(a[1][2]) > maxVal) { p = 1; q = 2; maxVal = Math.abs(a[1][2]); }
        if (maxVal < 1e-12) break;

        const theta = 0.5 * Math.atan2(2 * a[p][q], a[p][p] - a[q][q]);
        const c = Math.cos(theta), s = Math.sin(theta);

        // Givens rotation
        const G = [[1,0,0],[0,1,0],[0,0,1]];
        G[p][p] = c; G[q][q] = c; G[p][q] = s; G[q][p] = -s;

        // A = G^T A G
        const tmp = matMul3x3(matMul3x3(transpose3x3(G), a), G);
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) a[i][j] = tmp[i][j];
        // V = V G
        const tmp2 = matMul3x3(v, G);
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) v[i][j] = tmp2[i][j];
    }

    return { eigvals: [a[0][0], a[1][1], a[2][2]], eigvecs: v };
}

/**
 * Compute RMSD between two coordinate arrays.
 *
 * @param {Array<Array<number>>} a
 * @param {Array<Array<number>>} b
 * @returns {number}
 */
function computeRmsd(a, b) {
    const n = Math.min(a.length, b.length);
    let sum = 0;
    for (let i = 0; i < n; i++) {
        for (let d = 0; d < 3; d++) {
            const diff = a[i][d] - b[i][d];
            sum += diff * diff;
        }
    }
    return Math.sqrt(sum / n);
}

/**
 * Build a CA-only PDB string from an array of [x,y,z] coordinates.
 * Includes CONECT records for consecutive CA bonds.
 *
 * @param {Array<Array<number>>} coords - (N,3) coordinates.
 * @param {string} chainId              - PDB chain identifier.
 * @param {number} startSerial          - Starting atom serial number.
 * @param {number} startResi            - Starting residue number.
 * @returns {string}
 */
function coordsToPdb(coords, chainId, startSerial, startResi) {
    const lines = [];
    for (let i = 0; i < coords.length; i++) {
        const serial = startSerial + i;
        const resi = startResi + i;
        const [x, y, z] = coords[i];
        lines.push(
            `ATOM  ${String(serial).padStart(5)}  CA  ALA ${chainId}` +
            `${String(resi).padStart(4)}    ` +
            `${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}` +
            `  1.00  0.00           C  `
        );
    }
    for (let i = 0; i < coords.length - 1; i++) {
        const s = startSerial + i;
        lines.push(`CONECT${String(s).padStart(5)}${String(s + 1).padStart(5)}`);
    }
    return lines.join("\n");
}

/**
 * Create a 3D viewer showing a protein structure with an overlaid motif.
 *
 * The protein is shown as a cartoon colored by activation (white->red).
 * The motif is Kabsch-aligned onto the protein fragment at the peak
 * activation position, then shown as green spheres+sticks.
 *
 * @param {HTMLElement} container        - DOM element for the viewer.
 * @param {string} accession            - UniProt accession (to fetch full PDB).
 * @param {Array<number>} activations   - Per-residue SAE activations.
 * @param {number} maxActivation        - Feature-level max activation.
 * @param {Array<Array<number>>} caBackbone - Full CA backbone coords from geometry data.
 * @param {number} peakPosition         - 0-indexed residue with highest activation.
 * @param {string} motifPdb             - PDB string of the mean motif structure.
 * @param {number} motifLength          - Number of residues in the motif.
 * @returns {Promise<void>}
 */
async function createMotifOverlayViewer(container, accession, activations, maxActivation,
                                         caBackbone, peakPosition, motifPdb, motifLength) {
    // Enforce WebGL limit
    if (_activeViewers.length >= MAX_WEBGL_CONTEXTS) {
        const oldest = _activeViewers.shift();
        try {
            oldest.viewer.clear();
            const oldCanvas = oldest.container.querySelector("canvas");
            if (oldCanvas) oldCanvas.remove();
        } catch (e) {}
        oldest.container.innerHTML = '<div class="viewer-placeholder">Recycled</div>';
    }

    const halfW = Math.floor((motifLength - 1) / 2);
    const fragStart = peakPosition - halfW;
    const fragEnd = peakPosition + halfW + 1;

    if (fragStart < 0 || fragEnd > caBackbone.length) {
        container.innerHTML = '<div class="viewer-placeholder">Peak too close to terminus for motif overlay</div>';
        return;
    }

    // Extract the protein fragment at the peak position
    const fragment = caBackbone.slice(fragStart, fragEnd);

    // Parse motif PDB into coords
    const motifCoords = [];
    for (const line of motifPdb.split("\n")) {
        if (line.startsWith("ATOM")) {
            const x = parseFloat(line.substring(30, 38));
            const y = parseFloat(line.substring(38, 46));
            const z = parseFloat(line.substring(46, 54));
            motifCoords.push([x, y, z]);
        }
    }

    // Kabsch-align motif onto the protein fragment
    const alignedMotif = kabschAlign(motifCoords, fragment);
    const rmsd = computeRmsd(alignedMotif, fragment);

    // Fetch the full protein PDB
    container.innerHTML = '<div class="viewer-placeholder"><div class="loading-spinner"></div> Loading structure...</div>';
    let pdbData;
    try {
        const res = await fetch(`/api/pdb/${accession}`);
        if (!res.ok) {
            container.innerHTML = '<div class="viewer-placeholder">No structure available for ' + accession + '</div>';
            return;
        }
        pdbData = await res.text();
    } catch (e) {
        container.innerHTML = '<div class="viewer-placeholder">Failed to fetch PDB</div>';
        return;
    }

    container.innerHTML = "";
    const viewer = $3Dmol.createViewer(container, {
        backgroundColor: "white",
        antialias: true,
    });

    // Model 0: full protein
    viewer.addModel(pdbData, "pdb");

    // Color protein by activation: white -> red
    const colorMap = {};
    for (let i = 0; i < activations.length; i++) {
        const act = activations[i] || 0;
        const norm = maxActivation > 0 ? Math.min(act / maxActivation, 1) : 0;
        const r = 255;
        const g = Math.round(255 - norm * (255 - 38));
        const b = Math.round(255 - norm * (255 - 38));
        colorMap[i + 1] = (r << 16) | (g << 8) | b;
    }

    viewer.setStyle({ model: 0 }, {
        cartoon: {
            colorfunc: function (atom) {
                return colorMap[atom.resi] ?? 0xFFFFFF;
            },
        },
    });

    // Model 1: aligned motif overlay (green spheres + sticks)
    const motifPdbAligned = coordsToPdb(alignedMotif, "M", 9001, 1);
    viewer.addModel(motifPdbAligned + "\nEND", "pdb");

    viewer.setStyle({ model: 1 }, {
        sphere: { radius: 0.7, color: "#16a34a" },
        stick: { radius: 0.25, color: "#16a34a" },
    });

    // Zoom to the motif region
    viewer.zoomTo({ model: 1 });
    viewer.render();

    _activeViewers.push({ viewer, container });

    // Show RMSD below the viewer
    const rmsdLabel = document.createElement("div");
    rmsdLabel.className = "secondary";
    rmsdLabel.style.fontSize = "0.85rem";
    rmsdLabel.style.marginTop = "0.3rem";
    rmsdLabel.textContent = `RMSD to motif: ${rmsd.toFixed(3)} \u00c5 ` +
        `(${motifLength} positions, peak at residue ${peakPosition + 1})`;
    container.parentNode.insertBefore(rmsdLabel, container.nextSibling);
}
