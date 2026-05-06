// Feature detail view — methods grid, top activating proteins (3D + seq + per-method confusion),
// activation bins, geometric profile. All wired to live /api endpoints.

// We render four method rows per protein. CATH residue is intentionally excluded
// (the per-residue test isn't supported per-protein in the current build).
//
//  - Geometric (m7): TP/FP/FN/TN per residue read straight from
//      geometry_enrichment[fid].plot_data.top_proteins[i].concordance_labels.
//  - MEME (m6): scan the protein's sequence with the discovered PWM and threshold,
//      compute confusion against the SAE truth on this protein.
//  - InterPro residue (m2): use cached InterPro domain spans for the protein
//      (when available) to flag residues inside the top annotation; else show
//      "no cached spans" so we don't fabricate.
//  - Position (m5): apply the top position predicate (e.g. pct_30_40, third_C)
//      to the residue range, then compute confusion against truth.
const PROT_METHODS = [
  { id: 7, key: "m7", short: "Geometric",        kind: "geom" },
  { id: 6, key: "m6", short: "MEME motif",       kind: "bio"  },
  { id: 2, key: "m2", short: "InterPro residue", kind: "bio"  },
  { id: 5, key: "m5", short: "Position",         kind: "bio"  },
];

function FeatureView({ layer, featureId, onBack }) {
  const sig    = useFetch(() => API.significance(layer, featureId), [layer, featureId]);
  const detail = useFetch(() => API.feature(layer, featureId),      [layer, featureId]);
  // Pull geometry / motif / position once at the feature level so each panel
  // (top proteins, activation bins, metagenomic hits) can reuse them without
  // refetching. Errors here aren't fatal — the panels degrade gracefully.
  const geom   = useFetch(() => API.geometry(layer, featureId).catch(() => null), [layer, featureId]);
  const motif  = useFetch(() => API.motif(layer, featureId).catch(() => null),    [layer, featureId]);
  const posn   = useFetch(() => API.position(layer, featureId).catch(() => null), [layer, featureId]);

  if (sig.loading || detail.loading) {
    return <div className="container"><Loading what={`feature L${layer}/f${featureId}`} /></div>;
  }
  if (sig.error)    return <div className="container"><ErrorBox err={sig.error}    title={`Failed to load f/${featureId}`} /></div>;
  if (detail.error) return <div className="container"><ErrorBox err={detail.error} title={`Failed to load f/${featureId}`} /></div>;

  const feat = sig.data || {};
  const det  = detail.data || {};
  const cat  = rowCategory(feat);

  // Build a per-feature "context" of shared inputs used by every protein card.
  const proteinCtx = {
    feat,
    featureMaxAct: geom.data?.feature_max_activation ?? null,
    saeThreshold:
      geom.data?.geometric_residue_level?.activation_threshold ??
      (geom.data?.feature_max_activation != null
        ? geom.data.feature_max_activation * 0.5
        : 0),
    motif: {
      pwm: motif.data?.motifs?.[0]?.pwm || null,
      aaOrder: motif.data?.motifs?.[0]?.aa_order || "ACDEFGHIKLMNPQRSTVWY",
      threshold: motif.data?.motifs?.[0]?.best_pwm_threshold ?? null,
      consensus: motif.data?.motifs?.[0]?.consensus,
    },
    posPredicate: posn.data?.top_positions?.[0]?.position || null,
    posF1: posn.data?.top_positions?.[0]?.best_f1,
  };

  return (
    <div className="container">
      <div className="fd-header">
        <div>
          <div className="fid-line">
            <span>Layer {layer}</span>
            <span style={{ color: 'var(--ink-3)' }}>·</span>
            <span>f/{feat.feature_id ?? featureId}</span>
            <Pill kind={cat === "geom_only" ? "geom" : cat === "bio_only" ? "bio" : cat === "both" ? "both" : ""}>
              {cat === "geom_only" ? "Geometry only" :
               cat === "bio_only"  ? "Database only" :
               cat === "both"      ? "Database + Geometry" : "No significant method"}
            </Pill>
          </div>
        </div>
      </div>

      {/* Methods grid */}
      <div style={{ padding: '28px 0 0' }}>
        <span className="eyebrow">Annotation methods · BH-corrected</span>
        <h3 style={{ fontFamily: 'var(--serif)', fontSize: '22px', fontWeight: 500, margin: '8px 0 16px' }}>
          7 annotation methods, highlighted bars mark q&nbsp;&lt;&nbsp;0.05
          {det.dataset_coverage && det.dataset_coverage.pct_proteins_activated != null && (
            <span style={{ fontFamily: 'var(--mono)', fontSize: '13px', color: 'var(--ink-2)', fontWeight: 400, marginLeft: 14 }}>
              · activates on <strong style={{color:'var(--ink)'}}>{det.dataset_coverage.pct_proteins_activated.toFixed(1)}%</strong> of proteins
            </span>
          )}
        </h3>
        <div className="method-grid">
          {[7, 6, 2, 1, 4, 3, 5].map((mid) => METHOD_DEFS[mid - 1]).map((def) => {
            const k = def.id;
            const score = feat[`m${k}_score`];
            const label = feat[`m${k}_label`];
            const q = feat[`m${k}_q`];
            const sigBool = isSig(q);
            const cls = "method-card" + (sigBool ? " is-sig" : "") + " " + def.kind;
            return (
              <div key={k} className={cls} title={def.name}>
                <span className="qmark">{sigBool ? `q ${fmtQ(q)}` : `n.s.`}</span>
                <div className="name">{def.name}</div>
                <div className="metric">{def.metric}</div>
                <div className="score">{fmt(score, 3)}</div>
                <div className="label" title={label || ""}>{label || "—"}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Top activating proteins — 3D structure + activation-shaded sequence
          + per-method confusion (geom from concordance_labels, MEME/Position computed) */}
      <TopActivatingProteins ctx={proteinCtx} geomData={geom.data} loading={geom.loading} error={geom.error} />

      {/* Geometric profile (radar) + GBM importance */}
      <GeometryRow feat={feat} layer={layer} featureId={featureId} />

      {/* Activation bins — every protein renders the full ProteinEntry card
          (3D + sequence + MEME / Position confusion). Geom confusion is
          unavailable beyond the top 5 (concordance_labels aren't stored
          for bin proteins) so the geom row gracefully shows a note. */}
      <ActivationBins det={det} ctx={proteinCtx} />

      {/* NMPFams metagenomic hits — same structure as Swiss-Prot panel,
          but only the Geometric confusion row (no bio annotations exist
          for unannotated metagenomic sequences). */}
      <MetagenomicHits layer={layer} featureId={featureId} feat={feat} />
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Per-residue method prediction utilities
//
// All operate on a single protein given its sequence and the SAE truth mask
// (act > 0). They return a length-N array of booleans (1 = predicted).
// ────────────────────────────────────────────────────────────────────────

// All predicate names emitted by the position-enrichment pipeline. Each is a
// boolean test (i, n) -> bool that says whether residue index i (0-based) is
// inside the predicate's region of an n-residue protein.
const POSITION_PREDICATES = (() => {
  const pct = (lo, hi) => (i, n) => {
    const f = i / Math.max(n - 1, 1);
    return f >= lo && f < hi;
  };
  const third = (which) => (i, n) => {
    const f = i / Math.max(n - 1, 1);
    return which === "N" ? f < 1/3 : which === "C" ? f >= 2/3 : (f >= 1/3 && f < 2/3);
  };
  // first_<k> / last_<k> are absolute counts (residue index, not percentage).
  const firstN  = (k) => (i, n) => i < k;
  const lastN   = (k) => (i, n) => i >= n - k;
  // Named ranges used by the pipeline.
  const interior80 = (i, n) => { const f = i / Math.max(n - 1, 1); return f >= 0.1 && f < 0.9; };
  const terminal10 = (i, n) => { const f = i / Math.max(n - 1, 1); return f < 0.1 || f >= 0.9; };
  const mid20      = (i, n) => { const f = i / Math.max(n - 1, 1); return f >= 0.4 && f < 0.6; };

  return {
    pct_0_10:        pct(0.0,  0.1),
    pct_10_20:       pct(0.1,  0.2),
    pct_20_30:       pct(0.2,  0.3),
    pct_30_40:       pct(0.3,  0.4),
    pct_40_50:       pct(0.4,  0.5),
    pct_50_60:       pct(0.5,  0.6),
    pct_60_70:       pct(0.6,  0.7),
    pct_70_80:       pct(0.7,  0.8),
    pct_80_90:       pct(0.8,  0.9),
    pct_90_100:      pct(0.9,  1.001),
    third_N:         third("N"),
    third_M:         third("M"),
    third_C:         third("C"),
    first_5:         firstN(5),
    first_10:        firstN(10),
    first_20:        firstN(20),
    last_10:         lastN(10),
    last_20:         lastN(20),
    interior_80pct:  interior80,
    terminal_10pct:  terminal10,
    mid_20pct:       mid20,
  };
})();

function predictPosition(predicateName, length) {
  const f = POSITION_PREDICATES[predicateName];
  if (!f || !length) return null;
  const out = new Array(length);
  for (let i = 0; i < length; i++) out[i] = f(i, length) ? 1 : 0;
  return out;
}

// Score the PWM at every position (sliding window). Mark every residue inside
// a window whose log-likelihood ratio >= threshold as predicted=1.
//
// `pwm` rows are *probabilities* in `aa_order` (rows sum to 1, exactly as the
// pipeline writes them in motif_pwm_enrichment.motifs[].pwm). The background
// is uniform over 20 letters; the score is the standard MEME log-odds:
//
//   score = sum_k log( pwm[k][a] / (1/20) ) = sum_k ( log(pwm[k][a]) + log(20) )
//
// Zero-probability cells get a small pseudocount so log doesn't blow up.
function predictMEME(seq, pwm, aaOrder, threshold) {
  if (!seq || !pwm || !pwm.length || threshold == null) return null;
  const w = pwm.length;
  const n = seq.length;
  const aaIdx = {};
  for (let i = 0; i < aaOrder.length; i++) aaIdx[aaOrder[i]] = i;
  const logBg = Math.log(1 / aaOrder.length);
  const eps = 1e-6;  // pseudocount for zero-probability cells
  const out = new Array(n).fill(0);
  for (let p = 0; p + w <= n; p++) {
    let s = 0;
    let valid = true;
    for (let k = 0; k < w; k++) {
      const ix = aaIdx[seq[p + k]];
      if (ix == null) { valid = false; break; }
      const prob = pwm[k][ix];
      s += Math.log(prob + eps) - logBg;
    }
    if (valid && s >= threshold) {
      for (let k = 0; k < w; k++) out[p + k] = 1;
    }
  }
  return out;
}

function confusion(truth, pred) {
  if (!truth || !pred || !truth.length || truth.length !== pred.length) return null;
  const labels = new Array(truth.length);
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < truth.length; i++) {
    const t = truth[i] ? 1 : 0;
    const p = pred[i] ? 1 : 0;
    let lab;
    if (t && p)       { lab = "tp"; tp++; }
    else if (!t && p) { lab = "fp"; fp++; }
    else if (t && !p) { lab = "fn"; fn++; }
    else              { lab = "tn"; tn++; }
    labels[i] = lab;
  }
  const total = tp + fp + fn + tn;
  return { labels, tp, fp, fn, tn, agree_pct: total ? 100 * (tp + tn) / total : 0 };
}

// Pretty-print agree pct so 199/200 reads as "99.5", not "100". Anything with
// any disagreement floors below 100 so the value never lies; perfect agreement
// reports as "100".
function fmtAgreePct(conf) {
  if (!conf) return "—";
  const total = conf.tp + conf.fp + conf.fn + conf.tn;
  if (!total) return "—";
  const disagreement = conf.fp + conf.fn;
  if (disagreement === 0) return "100";
  // Always strictly below 100 — use a 0.05% floor and 1 decimal place.
  const raw = (100 * (conf.tp + conf.tn)) / total;
  const capped = Math.min(raw, 99.95);
  return capped.toFixed(1);
}

// Per-residue colour map for the 3D structure: pure white → red activation
// intensity. The same ramp mol_viewer.js uses for Swiss-Prot — applied
// consistently to NMPFams cartoons too, so the backbone never gets class
// colours. Confusion classes belong on the strip below, not on the structure.
function buildActivationColorMap(acts, maxAct) {
  const map = {};
  if (!acts) return map;
  for (let i = 0; i < acts.length; i++) {
    const norm = maxAct > 0 ? Math.min((acts[i] || 0) / maxAct, 1) : 0;
    const r = 255;
    const g = Math.round(255 - norm * (255 - 38));
    const b = Math.round(255 - norm * (255 - 38));
    map[i + 1] = (r << 16) | (g << 8) | b;
  }
  return map;
}

// Initialise a 3Dmol viewer for the given PDB URL, colouring residues by an
// arbitrary residue-number → hex map (e.g. TP/FP/FN/TN palette). Container
// must have non-zero size at call time.
async function init3DViewerWithMap(container, pdbUrl, colorMap) {
  if (!container || !window.$3Dmol) return;
  try {
    const r = await fetch(pdbUrl);
    if (!r.ok) {
      container.innerHTML = '<div class="viewer-placeholder">no structure</div>';
      return;
    }
    const pdbData = await r.text();
    container.innerHTML = "";
    const viewer = $3Dmol.createViewer(container, { backgroundColor: "white", antialias: true });
    viewer.addModel(pdbData, "pdb");
    viewer.setStyle({}, {
      cartoon: { colorfunc: (atom) => colorMap[atom.resi] ?? 0xECE7DA },
    });
    viewer.zoomTo();
    viewer.render();
  } catch (e) {
    if (container) container.innerHTML = '<div class="viewer-placeholder">no structure</div>';
  }
}

function confusionFromLabels(labels) {
  // The pipeline writes per-residue labels using "agree" for the joint-positive
  // case (act AND geom) and "tn" for joint-negative; "fp" / "fn" mean the geom
  // prediction disagrees with the SAE truth. Normalise to the canonical
  // tp/fp/fn/tn schema used by the rest of the SPA.
  if (!labels) return null;
  let tp = 0, fp = 0, fn = 0, tn = 0;
  const norm = labels.map(l => {
    if (l === "agree" || l === "tp") { tp++; return "tp"; }
    if (l === "fp")                  { fp++; return "fp"; }
    if (l === "fn")                  { fn++; return "fn"; }
    tn++;
    return "tn";
  });
  const total = tp + fp + fn + tn;
  return { labels: norm, tp, fp, fn, tn, agree_pct: total ? 100 * (tp + tn) / total : 0 };
}

// ────────────────────────────────────────────────────────────────────────
// Per-protein card — 3D structure (left) + sequence + method confusion stack
// ────────────────────────────────────────────────────────────────────────
function TopActivatingProteins({ ctx, geomData, loading, error }) {
  if (loading) return <div className="panel"><div className="panel-body"><Loading what="top activating proteins" /></div></div>;
  if (error)   return <div className="panel"><div className="panel-body"><ErrorBox err={error} title="Failed to load geometry enrichment" /></div></div>;

  const tps = geomData?.plot_data?.top_proteins || [];
  if (!tps.length) {
    return (
      <div className="panel"><div className="panel-body">
        No per-protein geometry plot available for this feature.
      </div></div>
    );
  }

  return (
    <div className="panel" style={{ marginTop: 24 }}>
      <div className="panel-head">
        <h4>Top activating proteins · structure · per-method residue agreement</h4>
        <span className="sub">
          {tps.length} protein{tps.length === 1 ? "" : "s"} ·
          activation in orange ·
          per-method confusion strips (TP / FP / FN / TN) · hover a residue to align across rows
        </span>
      </div>
      <div className="panel-body" style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        {tps.map((p, i) => (
          <ProteinEntry key={p.accession + i} protein={p} ctx={ctx} />
        ))}
      </div>
    </div>
  );
}

function ProteinEntry({ protein, ctx }) {
  const { feat, featureMaxAct, saeThreshold, motif, posPredicate, posF1 } = ctx;
  const containerRef = React.useRef(null);
  const [hover, setHover] = React.useState(null);
  const seq = protein.sequence || "";
  // Two possible schemas for the activation source:
  //   - geometry_enrichment.plot_data.top_proteins: `sae_activation_profile`
  //   - features.{fid}.activation_bins[*]:          `per_residue_activations`
  const acts = protein.sae_activation_profile || protein.per_residue_activations || [];
  const maxAct = Math.max(...acts, featureMaxAct || 0.001);

  // Truth mask matches the pipeline's truth: residues whose raw SAE activation
  // exceeds the per-feature SAE threshold (geometric_residue_level.activation_threshold).
  // This is what concordance_labels in geometry_enrichment is computed against,
  // so the MEME / Position rows here use the same truth as the geom row above.
  const truthMask = React.useMemo(
    () => acts.map(a => (a > saeThreshold ? 1 : 0)),
    [acts, saeThreshold]
  );

  // Geom confusion — only available when the pipeline pre-computed concordance
  // labels (i.e. for the top 5 proteins in plot_data.top_proteins). Activation
  // bin / arbitrary proteins won't have it — null falls through to the row's
  // graceful "concordance not stored" message.
  const geomConf = React.useMemo(
    () => protein.concordance_labels && protein.concordance_labels.length
      ? confusionFromLabels(protein.concordance_labels)
      : null,
    [protein.concordance_labels]
  );

  // MEME confusion — derive from PWM scan.
  const memeConf = React.useMemo(() => {
    const pred = motif.pwm
      ? predictMEME(seq, motif.pwm, motif.aaOrder, motif.threshold)
      : null;
    return pred ? confusion(truthMask, pred) : null;
  }, [seq, motif, truthMask]);

  // Position confusion.
  const posConf = React.useMemo(() => {
    const pred = posPredicate ? predictPosition(posPredicate, seq.length) : null;
    return pred ? confusion(truthMask, pred) : null;
  }, [posPredicate, seq, truthMask]);

  // InterPro residue confusion. Lazy-fetch the cached InterPro domain spans
  // for this protein (only present for layers that have an interpro_cache).
  // The "prediction" is residues inside any domain whose interpro_accession
  // matches the feature's top InterPro-residue annotation; if no exact match,
  // fall back to "any domain" so the user still sees a strip.
  const [iprDomains, setIprDomains] = React.useState(null);
  React.useEffect(() => {
    let cancelled = false;
    if (!protein.accession) return;
    fetch(`/api/interpro/${protein.accession}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (!cancelled) setIprDomains(d?.domains || []); })
      .catch(() => { if (!cancelled) setIprDomains([]); });
    return () => { cancelled = true; };
  }, [protein.accession]);

  const iprConf = React.useMemo(() => {
    if (iprDomains == null) return null;            // still loading
    if (!iprDomains.length) return null;             // genuinely no annotations
    const targetCode = (feat?.m2_label || "").split(" ")[0];   // e.g. "IPR036291 NAD..."
    const matches = targetCode
      ? iprDomains.filter(d => d.interpro_accession === targetCode)
      : iprDomains;
    if (!matches.length) return null;
    const pred = new Array(seq.length).fill(0);
    for (const d of matches) {
      const start = Math.max(0, (d.start || 1) - 1);
      const end   = Math.min(seq.length, (d.end || seq.length));
      for (let i = start; i < end; i++) pred[i] = 1;
    }
    return confusion(truthMask, pred);
  }, [iprDomains, feat?.m2_label, seq.length, truthMask]);

  // 3D viewer — lazy-init via IntersectionObserver. Colour map is the hybrid:
  // activation orange ramp at TP/unlabeled residues, blue at FP, red at FN.
  // For activation-bin proteins (no labels) this becomes pure activation
  // intensity, equivalent to the original mol_viewer.js behaviour.
  React.useEffect(() => {
    if (!protein.accession || !containerRef.current) return;
    let observer;
    let initialized = false;
    const init = () => {
      if (initialized || !containerRef.current) return;
      initialized = true;
      const norm = featureMaxAct && featureMaxAct > 0 ? featureMaxAct : maxAct;
      const colorMap = buildActivationColorMap(acts, norm);
      init3DViewerWithMap(containerRef.current,
        `/api/pdb/${protein.accession}`, colorMap);
    };
    if ("IntersectionObserver" in window) {
      observer = new IntersectionObserver((entries) => {
        for (const e of entries) {
          if (e.isIntersecting) { init(); observer.disconnect(); break; }
        }
      }, { rootMargin: "120px" });
      observer.observe(containerRef.current);
    } else {
      init();
    }
    return () => { if (observer) observer.disconnect(); };
  }, [protein.accession]);

  // Count of residues above the SAE truth threshold. The previous "active
  // window N–M" reading was misleading because most positions in that range
  // were quiet — what matters is the fraction of residues that fired.
  const nActive = React.useMemo(
    () => acts.reduce((s, a) => s + (a > saeThreshold ? 1 : 0), 0),
    [acts, saeThreshold]
  );

  // Build a single shared sequence row + the four method rows so they all
  // align at residue level. We render them in a horizontally-scrolling
  // container so long proteins remain navigable.
  return (
    <div className="prot-entry">
      <div className="pe-head">
        <div className="uid">{protein.accession || "—"}</div>
        <div className="meta">
          <span><span className="lbl">Length</span> <strong>{seq.length} aa</strong></span>
          <span><span className="lbl">Active</span>
            <strong>{nActive}/{seq.length}</strong>
            <span style={{ color: 'var(--ink-3)', marginLeft: 4 }}>
              ({seq.length ? (100 * nActive / seq.length).toFixed(1) : "0.0"}%)
            </span>
          </span>
          <span><span className="lbl">Feature PR-AUC</span> <strong>{fmt(feat?.m7_score, 2)}</strong></span>
        </div>
        <div className="max-act">max act <strong>{fmt(maxAct, 2)}</strong></div>
      </div>
      <div className="pe-body">
        {/* LEFT — 3D structure */}
        <div className="pe-struc">
          <div className="struc-canvas" ref={containerRef}
               style={{ width: "100%", height: 320, position: "relative", background: "var(--paper)" }}>
            <div className="viewer-placeholder"
                 style={{
                   position: "absolute", inset: 0,
                   display: "flex", alignItems: "center", justifyContent: "center",
                   fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink-3)",
                   textTransform: "uppercase", letterSpacing: "0.08em",
                 }}>
              {protein.accession ? "3D viewer loads on scroll" : "no structure"}
            </div>
          </div>
          <div className="struc-cap">
            <span>3Dmol cartoon · /api/pdb/{protein.accession}</span>
          </div>
          <div className="legend-row">
            <span className="sw">
              <span className="box" style={{ background: "rgb(255, 255, 255)", border: "1px solid var(--rule)" }}></span>
              act 0
            </span>
            <span className="sw">
              <span className="box" style={{ background: "rgb(255, 146, 146)" }}></span>
              ½ max
            </span>
            <span className="sw">
              <span className="box" style={{ background: "rgb(255, 38, 38)" }}></span>
              max act
            </span>
          </div>
        </div>

        {/* RIGHT — sequence with activation shading + per-method strips,
            sharing a single horizontal scroll so AA position N always
            aligns with column N in every confusion strip. */}
        <div className="pe-right">
          <AlignedSequenceStack
            seq={seq}
            acts={acts}
            maxAct={maxAct}
            saeThreshold={saeThreshold}
            hover={hover}
            onHover={setHover}
            rightStub={<>
              <span className="lbl">Max act</span>
              <strong>{fmt(maxAct, 2)}</strong>
            </>}
            methods={[
              {
                short: "Geometric", kind: "geom",
                conf: geomConf, q: feat?.m7_q,
                detail: geomConf
                  ? `q ${fmtQ(feat?.m7_q)}`
                  : `concordance only pre-computed for top 5 proteins · q ${fmtQ(feat?.m7_q)}`,
              },
              {
                short: "MEME motif", kind: "bio",
                conf: memeConf, q: feat?.m6_q,
                detail: motif.consensus
                  ? `consensus ${motif.consensus} · thr ${fmt(motif.threshold, 2)}`
                  : "no PWM available",
              },
              {
                short: "InterPro residue", kind: "bio",
                conf: iprConf,
                q: feat?.m2_q,
                detail: iprDomains == null
                  ? `${feat?.m2_label || "—"} · loading interpro cache…`
                  : !iprDomains.length
                    ? `${feat?.m2_label || "—"} · no interpro annotations for this protein`
                    : !iprConf
                      ? `${feat?.m2_label || "—"} · domain not in this protein's annotations`
                      : `${feat?.m2_label || "—"} · domains from interpro_cache`,
              },
              {
                short: "Position", kind: "bio",
                conf: posConf, q: feat?.m5_q,
                detail: posPredicate
                  ? `predicate "${posPredicate}" · global F1 ${fmt(posF1, 2)}`
                  : "no top predicate",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// AlignedSequenceStack — single horizontally-scrolling block that renders
// the AA sequence and N method-confusion strips on a shared residue grid
// (every row uses the same fixed cell width). Labels stick to the left
// and stats stick to the right so the user sees what they're looking at
// even when scrolled deep into a long protein.
// ────────────────────────────────────────────────────────────────────────
const CELL_W = 14;            // px per residue, shared across all rows
const NAME_W = 124;           // sticky left column width
const STATS_W = 168;          // sticky right column width

function AlignedSequenceStack({ seq, acts, maxAct, saeThreshold, hover, onHover, methods, rightStub }) {
  const trackWidth = (seq?.length || 0) * CELL_W;
  return (
    <React.Fragment>
      <div className="pe-section-label">
        <span>Sequence + per-method residue agreement</span>
        <span className="hint">
          {seq.length} aa · {CELL_W}px per residue · hover or scroll horizontally to align rows · truth = act &gt; SAE threshold {fmt(saeThreshold, 2)}
        </span>
      </div>
      <div className="aligned-scroll" onMouseLeave={() => onHover && onHover(null)}>
        {/* AA row */}
        <div className="aligned-row aligned-aa">
          <div className="aligned-name aligned-aa-name">SAE activation</div>
          <div className="aligned-track" style={{ width: trackWidth }}>
            {seq.split("").map((aa, i) => {
              const a = acts[i] || 0;
              const isHot = a > saeThreshold;
              // Only orange-shade residues that are above the pipeline's truth
              // threshold. Below-threshold residues are TN and shouldn't read
              // as "activated" — that's what was making the user think a sub-
              // threshold position should be FN.
              const bg = isHot && maxAct > 0
                ? `color-mix(in oklch, var(--geom) ${Math.round(Math.min(a / maxAct, 1) * 80)}%, var(--paper))`
                : "transparent";
              return (
                <span
                  key={i}
                  className={"aligned-aa-cell" + (isHot ? " hot" : "") + (hover === i ? " hovered" : "")}
                  style={{ background: bg }}
                  onMouseEnter={() => onHover && onHover(i)}
                  title={`pos ${i + 1} · ${aa} · act ${a.toFixed(3)}${isHot ? " (truth=1)" : " (truth=0)"}`}>
                  {aa}
                </span>
              );
            })}
          </div>
          <div className="aligned-stats aligned-aa-stats">{rightStub}</div>
        </div>
        {/* Method rows */}
        {methods.map((m, i) => (
          <AlignedMethodRow key={i} m={m} hover={hover} onHover={onHover} trackWidth={trackWidth} />
        ))}
      </div>
    </React.Fragment>
  );
}

function AlignedMethodRow({ m, hover, onHover, trackWidth }) {
  const sigBool = isSig(m.q);
  return (
    <div className={"aligned-row aligned-method " + m.kind + (sigBool ? " is-sig" : "")}>
      <div className="aligned-name">
        <span className="dot"></span>
        {m.short}
      </div>
      <div className="aligned-track aligned-conf" style={{ width: trackWidth }}>
        {m.conf
          ? m.conf.labels.map((c, i) => {
              const cellClass =
                "aligned-conf-cell" +
                (c === "tp" ? " tp" : c === "fp" ? " fp" : c === "fn" ? " fn" : " tn") +
                (hover === i ? " hovered" : "") +
                (m.kind === "geom" ? " geom" : " bio");
              return (
                <span
                  key={i}
                  className={cellClass}
                  onMouseEnter={() => onHover && onHover(i)}
                />
              );
            })
          : <div className="aligned-conf-placeholder">{m.detail}</div>}
      </div>
      <div className="aligned-stats">
        <div className="row1">
          {m.conf
            ? <span>tp {m.conf.tp} · fp {m.conf.fp} · fn {m.conf.fn}</span>
            : <span style={{ color: 'var(--ink-3)' }}>—</span>}
        </div>
        <div className="row2">
          <span>q {fmtQ(m.q)}</span>
        </div>
      </div>
    </div>
  );
}

// Per-protein method row. Stat column shows ONLY per-protein numbers
// (agreement %, residue counts, q-value). The feature-level F1/PR-AUC
// already lives in the methods grid at the top of the feature page,
// so we don't repeat it here.
function MethodRow({ m, conf, q, detail, hover, onHover }) {
  const sigBool = isSig(q);
  const accent = m.kind === "geom" ? "var(--geom)" : "var(--bio)";
  return (
    <div className={"method-row " + m.kind + (sigBool ? " is-sig" : "")}>
      <div className="m-name">
        <span className="dot"></span>
        {m.short}
      </div>
      <div className="conf-track">
        {conf
          ? conf.labels.map((c, i) => {
              const cellClass =
                "cell" +
                (c === "tp" ? " tp" : c === "fp" ? " fp" : c === "fn" ? " fn" : " tn") +
                (hover === i ? " hovered" : "") +
                (m.kind === "geom" ? " geom" : " bio");
              return (
                <span
                  key={i}
                  className={cellClass}
                  onMouseEnter={() => onHover && onHover(i)}
                />
              );
            })
          : <div style={{
              flex: 1, padding: '6px 10px',
              fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--ink-3)',
              fontStyle: 'italic'
            }}>{detail}</div>}
      </div>
      <div className="m-stats">
        <div className="row1">
          {conf
            ? <span>tp {conf.tp} · fp {conf.fp} · fn {conf.fn}</span>
            : <span style={{ color: 'var(--ink-3)' }}>—</span>}
        </div>
        <div className="row2">
          <span>q {fmtQ(q)}</span>
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Geometric profile (radar) + GBM feature importance
// ────────────────────────────────────────────────────────────────────────
function GeometryRow({ feat, layer, featureId }) {
  const radar = feat && feat.geometry_radar
    ? {
        curvature:   feat.geometry_radar.curvature   ?? 0,
        torsion:     feat.geometry_radar.torsion     ?? 0,
        planarity:   feat.geometry_radar.planarity   ?? 0,
        compactness: feat.geometry_radar.compactness ?? 0,
        contacts:    feat.geometry_radar.contacts    ?? 0,
        composition: feat.geometry_radar.composition ?? 0,
      }
    : null;

  const geomEnrich = useFetch(() => API.geometry(layer, featureId).catch(() => null), [layer, featureId]);

  const importances = (() => {
    const data = geomEnrich.data;
    if (!data) return null;
    const lvl = data.geometric_residue_level || data.geometric_protein_level || {};
    return lvl.feature_importances || null;
  })();

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginTop: '24px' }}>
      <div className="panel" style={{ margin: 0 }}>
        <div className="panel-head">
          <h4>Geometric profile</h4>
          <span className="sub">6-dim summary · normalised importance per category</span>
        </div>
        <div className="panel-body">
          {radar ? (
            <div className="radar-wrap">
              <div style={{ display: 'flex', justifyContent: 'center' }}>
                <RadarGlyph
                  scores={[radar.curvature, radar.torsion, radar.planarity, radar.compactness, radar.contacts, radar.composition]}
                  size={170}
                  color="var(--geom)" />
              </div>
              <div className="radar-bars">
                {Object.entries(radar).map(([k, v]) => (
                  <div key={k} className="radar-bar">
                    <span className="lbl">{k}</span>
                    <span className="track"><span className="fill" style={{ width: `${v * 100}%` }}></span></span>
                    <span className="v">{(v ?? 0).toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="spinner">No geometric profile available for this feature.</div>
          )}
        </div>
      </div>
      <div className="panel" style={{ margin: 0 }}>
        <div className="panel-head">
          <h4>GBM feature importance</h4>
          <span className="sub">Cα descriptor importances from the geometric classifier</span>
        </div>
        <div className="panel-body">
          {geomEnrich.loading
            ? <Loading what="GBM importance" />
            : importances
              ? <GbmImportanceVector importances={importances} />
              : <div className="spinner">No importance vector available.</div>}
        </div>
      </div>
    </div>
  );
}

function GbmImportanceVector({ importances }) {
  // importances is a dict of {feature_name: weight}. Sort descending and render
  // the top-K as a horizontal bar list with a small label-totals strip on top.
  const entries = Object.entries(importances)
    .map(([k, v]) => [k, Number(v) || 0])
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]);
  if (!entries.length) return <div className="spinner">No non-zero importances.</div>;
  const max = entries[0][1] || 1;
  const SHOW = Math.min(entries.length, 16);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {entries.slice(0, SHOW).map(([k, v], i) => (
        <div key={k} style={{
          display: 'grid',
          gridTemplateColumns: '180px 1fr 56px',
          gap: 10, alignItems: 'center',
          fontFamily: 'var(--mono)', fontSize: 11,
        }}>
          <span style={{ color: i < 3 ? 'var(--ink)' : 'var(--ink-2)', fontWeight: i < 3 ? 600 : 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={k}>{k}</span>
          <span style={{
            height: 10, background: 'var(--paper-2)',
            border: '1px solid var(--rule)', position: 'relative',
          }}>
            <span style={{
              position: 'absolute', left: 0, top: 0, bottom: 0,
              width: `${(v / max) * 100}%`,
              background: i < 3 ? 'var(--geom)' : 'var(--ink)',
              opacity: i < 3 ? 0.85 : 0.5,
            }}></span>
          </span>
          <span style={{ color: 'var(--ink-2)', textAlign: 'right' }}>{(v * 100).toFixed(1)}%</span>
        </div>
      ))}
      {entries.length > SHOW && (
        <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--ink-3)', marginTop: 4 }}>
          + {entries.length - SHOW} more descriptors below threshold
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Activation bins — click each to expand the proteins that activate in that bin.
// Real schema: activation_bins is a dict keyed by "0.0-0.25", ..., values are
// arrays of {accession, max_activation, sequence, per_residue_activations, ...}.
// ────────────────────────────────────────────────────────────────────────
function ActivationBins({ det, ctx }) {
  const bins = (det && det.activation_bins) || {};
  const order = ["0.75-1.0", "0.5-0.75", "0.25-0.5", "0.0-0.25"];
  const [open, setOpen] = React.useState(null);

  const items = order
    .filter(k => Array.isArray(bins[k]))
    .map(k => ({ key: k, label: k.replace("-", " – "), proteins: bins[k] }));
  if (!items.length) return null;

  return (
    <div className="panel" style={{ marginTop: 24 }}>
      <div className="panel-head">
        <h4>Activation bins</h4>
        <span className="sub">activating proteins grouped by max-activation range — click to expand</span>
      </div>
      <div className="panel-body" style={{ padding: 0 }}>
        {items.map((bin, i) => {
          const isOpen = open === bin.key;
          const top = parseFloat(bin.key.split("-")[1]);
          const fill = `color-mix(in oklch, var(--geom) ${Math.round(top * 80)}%, var(--paper))`;
          const maxCount = Math.max(...items.map(b => b.proteins.length));
          return (
            <div key={bin.key} style={{ borderBottom: i < items.length - 1 ? '1px solid var(--rule)' : 'none' }}>
              <button
                onClick={() => setOpen(isOpen ? null : bin.key)}
                style={{
                  width: '100%',
                  display: 'grid',
                  gridTemplateColumns: '24px 1fr auto auto',
                  alignItems: 'center', gap: 14,
                  padding: '14px 18px', background: 'transparent',
                  border: 'none', borderLeft: `4px solid ${fill}`,
                  font: 'inherit', textAlign: 'left', cursor: 'pointer',
                }}>
                <span style={{
                  fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--ink-3)',
                  transform: isOpen ? 'rotate(90deg)' : 'rotate(0deg)',
                  transition: 'transform 120ms',
                }}>▶</span>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--ink)', fontWeight: 600 }}>
                  {bin.label}
                </span>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--ink-2)' }}>
                  {bin.proteins.length} proteins
                </span>
                <span style={{
                  width: 80, height: 6, background: 'var(--paper-2)',
                  border: '1px solid var(--rule)', position: 'relative',
                }}>
                  <span style={{
                    position: 'absolute', left: 0, top: 0, bottom: 0,
                    width: `${maxCount ? (bin.proteins.length / maxCount) * 100 : 0}%`,
                    background: fill,
                  }}></span>
                </span>
              </button>
              {isOpen && (
                <div style={{ padding: '14px 18px 18px', background: 'var(--paper-2)', display: 'flex', flexDirection: 'column', gap: 14 }}>
                  {bin.proteins.map((p, j) => (
                    <ProteinEntry
                      key={`${p.accession}-${j}`}
                      // Adapt the activation_bins schema to ProteinEntry's expectations.
                      // No concordance_labels here (those only exist for top_proteins),
                      // so the geom row will fall back to the "concordance not stored"
                      // hint. MEME / Position rows compute live from sequence + acts.
                      protein={{
                        accession: p.accession,
                        sequence: p.sequence,
                        sae_activation_profile: p.per_residue_activations,
                      }}
                      ctx={ctx}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// MetagenomicHits — NMPFams metagenomic protein clusters that activate
// this feature. Same per-residue confusion structure as Swiss-Prot, but
// only the Geometric row is meaningful (these sequences have no biological
// annotations by construction). Real data, no synthesis:
//   - sequence, sae_activation_profile, geom_prob_profile, concordance_labels
//     are pre-computed and stored in nmpfam_enrichment[fid].nmpfam_hits[i].
// ────────────────────────────────────────────────────────────────────────
function MetagenomicHits({ layer, featureId, feat }) {
  const nmp = useFetch(() => API.nmpfam(layer, featureId).catch(() => null), [layer, featureId]);
  const [open, setOpen] = React.useState(false);
  const [tier, setTier] = React.useState("triple");

  if (nmp.loading) return null;
  // Tolerate 404s — many features simply have no NMPFam hits.
  if (!nmp.data || !nmp.data.nmpfam_hits || !nmp.data.nmpfam_hits.length) return null;

  const allHits = nmp.data.nmpfam_hits;
  // Sort by concordance (n_agree) descending — the most striking metagenomic
  // hits are the ones where geom prediction lines up with SAE activation.
  const sorted = [...allHits].sort((a, b) => (b.n_agree || 0) - (a.n_agree || 0));
  // Top 8 strong-agreement hits in the default expansion
  const SHOW = 8;

  return (
    <div className="panel" style={{ marginTop: 24, borderTop: '2px solid var(--geom)' }}>
      <div className="panel-head" style={{ background: 'color-mix(in oklch, var(--geom) 6%, var(--paper-2))' }}>
        <h4 style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            fontFamily: 'var(--mono)', fontSize: 9,
            background: 'var(--ink)', color: 'var(--paper)',
            padding: '3px 8px', letterSpacing: '0.06em',
            textTransform: 'uppercase', borderRadius: 1,
          }}>Metagenomic</span>
          NMPFams hits · Geometric per-residue agreement
        </h4>
      </div>
      <div className="panel-body" style={{ padding: 0 }}>
        <button
          onClick={() => setOpen(!open)}
          style={{
            width: '100%', display: 'grid',
            gridTemplateColumns: '24px 1fr auto auto',
            alignItems: 'center', gap: 14,
            padding: '14px 18px', background: 'transparent',
            border: 'none', borderLeft: '4px solid var(--geom)',
            font: 'inherit', textAlign: 'left', cursor: 'pointer',
            borderBottom: open ? '1px solid var(--rule)' : 'none',
          }}>
          <span style={{
            fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--ink-3)',
            transform: open ? 'rotate(90deg)' : 'rotate(0deg)',
            transition: 'transform 120ms',
          }}>▶</span>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--ink)', fontWeight: 600 }}>
            All NMPFams metagenomic hits
          </span>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--ink-2)' }}>
            {allHits.length} families · max SAE act {fmt(Math.max(...allHits.map(h => h.max_sae_activation || 0)), 2)}
          </span>
          <span style={{
            width: 80, height: 6, background: 'var(--paper-2)', border: '1px solid var(--rule)',
            position: 'relative',
          }}>
            <span style={{
              position: 'absolute', left: 0, top: 0, bottom: 0,
              width: `${Math.min(100, (sorted.filter(h => h.n_agree > 0).length / Math.max(allHits.length, 1)) * 100)}%`,
              background: 'var(--geom)', opacity: 0.7,
            }}></span>
          </span>
        </button>
        {open && (
          <div style={{ padding: '14px 18px 18px', background: 'var(--paper-2)', display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--ink-3)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Showing top {Math.min(SHOW, sorted.length)} hits by n_agree (residues where SAE truth and geom prediction overlap).
              Each hit's structure is fetched live from the Fleming Institute NMPFams DB.
            </div>
            {sorted.slice(0, SHOW).map((hit, i) => (
              <MetagenomicEntry
                key={hit.family_id + i}
                hit={hit}
                feat={feat}
                activationThreshold={nmp.data.activation_threshold_sae}
              />
            ))}
            {sorted.length > SHOW && (
              <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--ink-3)' }}>
                + {sorted.length - SHOW} more NMPFams families with weaker concordance
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function MetagenomicEntry({ hit, feat, activationThreshold }) {
  const seq = hit.sequence || "";
  const acts = hit.sae_activation_profile || [];
  const maxAct = Math.max(...acts, hit.max_sae_activation || 0.001);
  // Use the file-level SAE truth threshold from nmpfam_enrichment.json so the
  // AA shading + threshold readout agree with the pre-computed concordance
  // labels. Falls back to half-of-max only if the prop didn't make it through.
  const saeThreshold = activationThreshold
    ?? hit.activation_threshold_sae
    ?? (hit.max_sae_activation != null ? hit.max_sae_activation * 0.5 : 0);
  const geomConf = React.useMemo(
    () => confusionFromLabels(hit.concordance_labels || []),
    [hit.concordance_labels]
  );
  const [hover, setHover] = React.useState(null);
  const containerRef = React.useRef(null);

  React.useEffect(() => {
    if (!hit.family_id || !containerRef.current) return;
    let observer, initialized = false;
    const init = () => {
      if (initialized || !containerRef.current) return;
      initialized = true;
      // Pure white → red activation intensity, same ramp as Swiss-Prot.
      const colorMap = buildActivationColorMap(acts, maxAct);
      init3DViewerWithMap(containerRef.current,
        `/api/nmpfam-pdb/${hit.family_id}`, colorMap);
    };
    if ("IntersectionObserver" in window) {
      observer = new IntersectionObserver((entries) => {
        for (const e of entries) if (e.isIntersecting) { init(); observer.disconnect(); break; }
      }, { rootMargin: "120px" });
      observer.observe(containerRef.current);
    } else {
      init();
    }
    return () => { if (observer) observer.disconnect(); };
  }, [hit.family_id]);

  return (
    <div className="prot-entry">
      <div className="pe-head" style={{ background: 'color-mix(in oklch, var(--geom) 8%, var(--paper-2))' }}>
        <div className="uid" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            fontFamily: 'var(--mono)', fontSize: 9,
            background: 'var(--ink)', color: 'var(--paper)',
            padding: '2px 6px', letterSpacing: '0.06em',
            textTransform: 'uppercase', borderRadius: 1,
          }}>NMPFams</span>
          <a href={hit.nmpfams_url} target="_blank" rel="noreferrer"
             style={{ color: 'var(--ink)', textDecoration: 'none', fontWeight: 600 }}>
            {hit.family_id}
          </a>
        </div>
        <div className="meta">
          <span><span className="lbl">Origin</span> <strong>{hit.category}</strong></span>
          <span><span className="lbl">Length</span> <strong>{hit.n_residues} aa</strong></span>
          <span><span className="lbl">Sequences</span> <strong>{(hit.sequence_count ?? 0).toLocaleString()}</strong></span>
          <span><span className="lbl">Geom PR-AUC (Swiss-Prot)</span> <strong>{fmt(feat?.m7_score, 2)}</strong></span>
        </div>
        <div className="max-act">max SAE act <strong>{fmt(hit.max_sae_activation, 2)}</strong></div>
      </div>
      <div className="pe-body">
        <div className="pe-struc">
          <div className="struc-canvas" ref={containerRef}
               style={{ width: "100%", height: 320, position: "relative", background: "var(--paper)" }}>
            <div className="viewer-placeholder"
                 style={{
                   position: "absolute", inset: 0,
                   display: "flex", alignItems: "center", justifyContent: "center",
                 }}>
              ESMFold cartoon loads on scroll
            </div>
          </div>
          <div className="struc-cap">
            <span>ESMFold (predicted) · {hit.family_id}</span>
          </div>
          <div className="legend-row">
            <span className="sw">
              <span className="box" style={{ background: "rgb(255, 255, 255)", border: "1px solid var(--rule)" }}></span>
              act 0
            </span>
            <span className="sw">
              <span className="box" style={{ background: "rgb(255, 146, 146)" }}></span>
              ½ max
            </span>
            <span className="sw">
              <span className="box" style={{ background: "rgb(255, 38, 38)" }}></span>
              max act
            </span>
          </div>
        </div>

        <div className="pe-right">
          <AlignedSequenceStack
            seq={seq}
            acts={acts}
            maxAct={maxAct}
            saeThreshold={saeThreshold}
            hover={hover}
            onHover={setHover}
            rightStub={<>
              <span className="lbl">Max act</span>
              <strong>{fmt(maxAct, 2)}</strong>
            </>}
            methods={[
              {
                short: "Geometric", kind: "geom",
                conf: geomConf, q: feat?.m7_q,
                detail: `Swiss-Prot PR-AUC ${fmt(feat?.m7_score, 2)} · q ${fmtQ(feat?.m7_q)}`,
              },
            ]}
          />
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--ink-3)',
            marginTop: 8, fontStyle: 'italic',
          }}>
            bio annotation methods don't apply to unannotated metagenomic sequences
          </div>
        </div>
      </div>
    </div>
  );
}

window.FeatureView = FeatureView;
