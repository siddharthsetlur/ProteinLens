// Reusable small components: pills, sigstrip, radar glyph, sequence strip, helpers.

const isSig = (q) => q != null && q < 0.05;

function fmt(v, d = 2) {
  if (v == null) return "—";
  return Number(v).toFixed(d);
}
function fmtQ(q) {
  if (q == null) return "—";
  if (q < 1e-3) return q.toExponential(1).replace("e", "·10");
  return q.toFixed(3);
}

function Pill({ kind, children }) {
  return <span className={`pill ${kind || ""}`}><span className="dot"></span>{children}</span>;
}

// Per-axis colours for the vertex dots. Order matches the radar axes:
// curvature, torsion, planarity, compactness, contacts, composition.
// Same hues the old viz used so the visual identity is preserved.
const RADAR_AXIS_COLORS = [
  "#3b82f6",  // curvature   — blue
  "#ef4444",  // torsion     — red
  "#22c55e",  // planarity   — green
  "#f97316",  // compactness — orange
  "#8b5cf6",  // contacts    — purple
  "#14b8a6",  // composition — teal
];

// Hexagonal radar glyph. Renders 2 concentric guide rings, 6 axis spokes,
// the data polygon (filled + stroked in the geom accent), and a colored
// dot at each vertex so the user can read which axis is which at a glance.
// `scores` is a length-6 array in [0..1] for axes:
//   curvature, torsion, planarity, compactness, contacts, composition.
function RadarGlyph({ scores, size = 28, color = "var(--geom)" }) {
  if (!scores || !Array.isArray(scores) || scores.length < 6) {
    return <svg className="radar-svg" width={size} height={size}></svg>;
  }
  const cx = size / 2, cy = size / 2, r = size / 2 - 2;
  const n = 6;
  // Vertex dots/decorations get suppressed at very small sizes so the
  // 28px table glyph stays clean.
  const detailed = size >= 60;

  const angle = (i) => (Math.PI * 2 * i) / n - Math.PI / 2;
  const vertexAt = (i, radius) =>
    [cx + Math.cos(angle(i)) * radius, cy + Math.sin(angle(i)) * radius];

  const dataPts = scores.slice(0, 6).map((s, i) => {
    const v = Math.max(0.02, Math.min(1, s ?? 0));
    return vertexAt(i, r * v);
  });
  const polyPts = dataPts.map((p) => p.join(",")).join(" ");
  const outerRingPts = Array.from({ length: n }, (_, i) => vertexAt(i, r).join(",")).join(" ");
  const innerRingPts = Array.from({ length: n }, (_, i) => vertexAt(i, r * 0.5).join(",")).join(" ");

  return (
    <svg className="radar-svg" width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {detailed && (
        <polygon points={innerRingPts} fill="none" stroke="var(--rule-2)" strokeWidth="0.6" />
      )}
      <polygon points={outerRingPts} fill="none" stroke="var(--rule)" strokeWidth="0.6" />
      {detailed && Array.from({ length: n }, (_, i) => {
        const [x, y] = vertexAt(i, r);
        return (
          <line key={i}
                x1={cx} y1={cy} x2={x} y2={y}
                stroke="var(--rule-2)" strokeWidth="0.6" />
        );
      })}
      <polygon points={polyPts} fill={color} fillOpacity="0.22" stroke={color} strokeWidth="1.4" />
      {detailed && dataPts.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={size > 130 ? 3.5 : 2.5}
                fill={RADAR_AXIS_COLORS[i] || color}
                stroke="var(--paper)" strokeWidth="0.6" />
      ))}
    </svg>
  );
}

// 7-dot strip showing per-method significance status.
function SigStrip({ feat }) {
  const dots = [];
  for (let k = 1; k <= 7; k++) {
    const sig = isSig(feat[`m${k}_q`]);
    const cls = sig ? (k === 7 ? "sd geom" : "sd bio") : "sd";
    dots.push(<span key={k} className={cls} title={`${METHOD_DEFS[k - 1].short}: ${sig ? "q<0.05" : "n.s."}`}></span>);
  }
  return <span className="sigstrip">{dots}</span>;
}

function bioBest(feat) {
  const xs = [feat.m1_score, feat.m2_score, feat.m3_score, feat.m4_score, feat.m5_score, feat.m6_score]
    .filter((v) => v != null);
  return xs.length ? Math.max(...xs) : null;
}
function bioSig(feat) {
  for (let k = 1; k <= 6; k++) if (isSig(feat[`m${k}_q`])) return true;
  return false;
}
function geomSig(feat) { return isSig(feat.m7_q); }
function rowCategory(feat) {
  const g = geomSig(feat), b = bioSig(feat);
  if (g && b) return "both";
  if (g) return "geom_only";
  if (b) return "bio_only";
  return "none";
}

// Convert a geometry_radar dict {curvature, torsion, ...} to ordered array.
function radarFromObj(obj) {
  if (!obj) return null;
  return [
    obj.curvature, obj.torsion, obj.planarity,
    obj.compactness, obj.contacts, obj.composition,
  ].map(v => Number(v) || 0);
}

// Boot/loading and error states
function Loading({ what }) {
  return <div className="spinner">Loading {what || "data"}…</div>;
}
function ErrorBox({ err, title }) {
  return (
    <div className="error-box">
      <div><strong>{title || "Failed to load"}</strong></div>
      <div style={{ marginTop: 4, whiteSpace: "pre-wrap" }}>{String(err && err.message ? err.message : err)}</div>
    </div>
  );
}

// Pull data from an async function once on mount.
function useFetch(fn, deps) {
  const [state, setState] = React.useState({ data: null, error: null, loading: true });
  React.useEffect(() => {
    let cancelled = false;
    setState({ data: null, error: null, loading: true });
    fn()
      .then((data) => { if (!cancelled) setState({ data, error: null, loading: false }); })
      .catch((error) => { if (!cancelled) setState({ data: null, error, loading: false }); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps || []);
  return state;
}

window.isSig = isSig;
window.fmt = fmt;
window.fmtQ = fmtQ;
window.Pill = Pill;
window.RadarGlyph = RadarGlyph;
window.SigStrip = SigStrip;
window.bioBest = bioBest;
window.bioSig = bioSig;
window.geomSig = geomSig;
window.rowCategory = rowCategory;
window.radarFromObj = radarFromObj;
window.Loading = Loading;
window.ErrorBox = ErrorBox;
window.useFetch = useFetch;
