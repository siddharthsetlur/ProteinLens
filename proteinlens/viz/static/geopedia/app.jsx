// Top-level app — owns view state and renders topbar + active view.
// View routing is purely client-side; the SPA reads window.location on first
// load so deep links like /feature/4/4714 land directly on the feature page.

function parseInitialView() {
  const path = window.location.pathname || "/";
  const parts = path.split("/").filter(Boolean);
  // Empty path → landing
  if (parts.length === 0) return { kind: "landing" };
  // /layer/<N>
  if (parts[0] === "layer" && parts[1]) {
    const layer = parseInt(parts[1], 10);
    if (!Number.isNaN(layer)) return { kind: "layer", layer };
  }
  // /feature/<L>/<id>  (preferred)  OR  /feature/<id>  (legacy single-layer)
  if (parts[0] === "feature") {
    if (parts.length >= 3) {
      const layer = parseInt(parts[1], 10);
      const fid   = parseInt(parts[2], 10);
      if (!Number.isNaN(layer) && !Number.isNaN(fid)) return { kind: "feature", layer, featureId: fid };
    } else if (parts.length === 2) {
      const fid = parseInt(parts[1], 10);
      if (!Number.isNaN(fid)) return { kind: "feature", layer: null, featureId: fid };
    }
  }
  // /case/<id>/<layer>[/family/<source>/<code>]
  if (parts[0] === "case" && parts[1] && parts[2]) {
    const layer = parseInt(parts[2], 10);
    if (!Number.isNaN(layer)) {
      // Family detail under granularity: /case/granularity/<layer>/family/<source>/<code>
      if (parts[1] === "granularity" && parts[3] === "family" && parts[4] && parts[5]) {
        return {
          kind: "casestudy",
          id: "granularity",
          layer,
          family: { source: parts[4], code: decodeURIComponent(parts[5]) },
        };
      }
      return { kind: "casestudy", id: parts[1], layer };
    }
  }
  return { kind: "landing" };
}

function pushUrl(view) {
  const path = (() => {
    if (view.kind === "landing") return "/";
    if (view.kind === "layer")    return `/layer/${view.layer}`;
    if (view.kind === "feature")  return `/feature/${view.layer}/${view.featureId}`;
    if (view.kind === "casestudy") {
      if (view.family) {
        return `/case/${view.id}/${view.layer}/family/${view.family.source}/${encodeURIComponent(view.family.code)}`;
      }
      return `/case/${view.id}/${view.layer}`;
    }
    return "/";
  })();
  if (window.location.pathname !== path) {
    window.history.pushState({}, "", path);
  }
}

function App() {
  const layersList = useFetch(API.layers, []);
  const landing    = useFetch(API.landing, []);

  const [view, setView] = React.useState(parseInitialView);

  React.useEffect(() => {
    pushUrl(view);
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [view]);

  React.useEffect(() => {
    const onPop = () => setView(parseInitialView());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  // If feature view is missing a layer (e.g. legacy URL), fall back to the only
  // available layer if just one is loaded; otherwise prompt to pick.
  React.useEffect(() => {
    if (view.kind === "feature" && view.layer == null && layersList.data && layersList.data.length === 1) {
      setView({ kind: "feature", layer: layersList.data[0].layer, featureId: view.featureId });
    }
  }, [view, layersList.data]);

  const breadcrumb = (
    <div className="breadcrumb">
      <a onClick={() => setView({ kind: "landing" })}>GeoPedia</a>
      {view.layer != null && (
        <React.Fragment>
          <span className="sep">/</span>
          <a onClick={() => setView({ kind: "layer", layer: view.layer })}>Layer {view.layer}</a>
        </React.Fragment>
      )}
      {view.kind === "feature" && (
        <React.Fragment>
          <span className="sep">/</span>
          <span style={{color:'var(--ink)'}}>f/{view.featureId}</span>
        </React.Fragment>
      )}
      {view.kind === "casestudy" && (
        <React.Fragment>
          <span className="sep">/</span>
          {view.family ? (
            <a onClick={() => setView({ kind: "casestudy", id: view.id, layer: view.layer })}>
              {view.id === "granularity" ? "case 02" : view.id}
            </a>
          ) : (
            <span style={{color:'var(--ink)'}}>
              {view.id === "geom-fills-db" ? "case 01"
              : view.id === "granularity"  ? "case 02"
              : view.id === "metagenomic"  ? "case 03"
              : view.id}
            </span>
          )}
          {view.family && (
            <React.Fragment>
              <span className="sep">/</span>
              <span style={{color:'var(--ink)'}}>{view.family.code}</span>
            </React.Fragment>
          )}
        </React.Fragment>
      )}
    </div>
  );

  const layersAvailable = (layersList.data || []).map(L => L.layer);

  return (
    <React.Fragment>
      <div className="topbar">
        <div className="brand">
          <div className="logo" aria-label="GeoPedia logo">
            <svg viewBox="0 0 76 40" aria-hidden="true">
              {/* Helix as flat ribbon coiling in 3D */}
              {(() => {
                const turns = 2.4, amp = 9, cy = 20, xStart = 2, xEnd = 36, segs = 56, ribbonH = 5;
                const front = [], back = [];
                for (let i = 0; i < segs; i++) {
                  const t1 = i / segs, t2 = (i + 1) / segs;
                  const x1 = xStart + t1 * (xEnd - xStart), x2 = xStart + t2 * (xEnd - xStart);
                  const a1 = t1 * turns * Math.PI * 2, a2 = t2 * turns * Math.PI * 2;
                  const y1 = cy + Math.sin(a1) * amp,   y2 = cy + Math.sin(a2) * amp;
                  const cMid = Math.cos((a1 + a2) / 2);
                  const isFront = cMid > 0;
                  const w = ribbonH * Math.max(0.35, Math.abs(cMid));
                  const pts = `${x1},${y1 - w / 2} ${x2},${y2 - w / 2} ${x2},${y2 + w / 2} ${x1},${y1 + w / 2}`;
                  (isFront ? front : back).push(
                    <polygon key={i} points={pts}
                      className={isFront ? "helix-ribbon-front" : "helix-ribbon-back"} />
                  );
                }
                return <g>{back}{front}</g>;
              })()}
              <path className="loop" d="M 36 20 Q 44 6 52 12 Q 56 16 55 14" />
              {(() => {
                const xFL = 50, xFR = 75, xBL = 55, xBR = 73, yFront = 26, yBack = 14;
                return (
                  <g>
                    <polygon className="sheet"
                      points={`${xBL},${yBack} ${xBR},${yBack} ${xFR},${yFront} ${xFL},${yFront}`} />
                    <line
                      x1={(xBL + xFL) / 2 - 0.5} y1={(yBack + yFront) / 2}
                      x2={(xBR + xFR) / 2 + 0.5} y2={(yBack + yFront) / 2}
                      stroke="var(--ink)" strokeOpacity="0.18" strokeWidth="0.6" />
                  </g>
                );
              })()}
            </svg>
          </div>
          <h1 onClick={() => setView({ kind: "landing" })} style={{cursor:'pointer'}}>GeoPedia</h1>
          <span className="ver">v0.5</span>
          {view.kind !== "landing" && <span style={{marginLeft:'12px'}}>{breadcrumb}</span>}
        </div>
        <nav className="topnav">
          <a onClick={() => setView({ kind: "landing" })}>Atlas</a>
          {layersAvailable.map(L => (
            <a key={L} onClick={() => setView({ kind: "layer", layer: L })}>L{L}</a>
          ))}
        </nav>
      </div>

      {view.kind === "landing" && (
        <Landing
          onPickLayer={(layer) => setView({ kind: "layer", layer })}
          onPickFeature={(layer, featureId) => setView({ kind: "feature", layer, featureId })}
        />
      )}
      {view.kind === "layer" && (
        <LayerView
          layer={view.layer}
          landingSummary={landing.data}
          onPickFeature={(layer, featureId) => setView({ kind: "feature", layer, featureId })}
          onOpenCaseStudy={(id) => setView({ kind: "casestudy", id, layer: view.layer })}
          layersAvailable={layersAvailable}
        />
      )}
      {view.kind === "feature" && view.layer != null && (
        <FeatureView
          layer={view.layer}
          featureId={view.featureId}
          onBack={() => setView({ kind: "layer", layer: view.layer })}
        />
      )}
      {view.kind === "feature" && view.layer == null && (
        <div className="container" style={{ padding: 32 }}>
          <ErrorBox title="Pick a layer first" err={`Several layers are loaded (${layersAvailable.join(", ")}). Open the feature from a specific layer page.`} />
        </div>
      )}
      {view.kind === "casestudy" && view.id === "geom-fills-db" && (
        <CaseStudyGeometry
          layer={view.layer}
          onPickFeature={(layer, featureId) => setView({ kind: "feature", layer, featureId })}
        />
      )}
      {view.kind === "casestudy" && view.id === "granularity" && !view.family && (
        <CaseStudyGranularity
          layer={view.layer}
          onPickFeature={(layer, featureId) => setView({ kind: "feature", layer, featureId })}
          onPickFamily={(source, code) => setView({
            kind: "casestudy", id: "granularity", layer: view.layer,
            family: { source, code },
          })}
        />
      )}
      {view.kind === "casestudy" && view.id === "granularity" && view.family && (
        <CaseStudyFamilyDetail
          layer={view.layer}
          family={view.family}
          onBack={() => setView({ kind: "casestudy", id: "granularity", layer: view.layer })}
          onPickFeature={(layer, featureId) => setView({ kind: "feature", layer, featureId })}
        />
      )}
      {view.kind === "casestudy" && view.id === "metagenomic" && (
        <CaseStudyMetagenomic
          layer={view.layer}
          onPickFeature={(layer, featureId) => setView({ kind: "feature", layer, featureId })}
          layersAvailable={layersAvailable}
        />
      )}

      <footer className="foot container" style={{paddingLeft:0, paddingRight:0}}>
        <span>GeoPedia · ESM-2 SAE feature atlas</span>
        <span className="num">
          {layersList.data ? `${layersList.data.length} layer${layersList.data.length === 1 ? '' : 's'} loaded` : ''}
        </span>
      </footer>
    </React.Fragment>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
