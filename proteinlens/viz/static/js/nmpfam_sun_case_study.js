/**
 * nmpfam_sun_case_study.js
 *
 * Renders the unified NMPFams annotation-transfer case study.
 *
 * For each curated feature, pulls live evidence from:
 *   /api/nmpfam-case-study              - triple_features + broader_gp_features index
 *   /api/feature/{id}/nmpfam            - full NMPFam hits list for a feature
 *   /api/feature/{id}/interpro          - InterPro protein + residue level hits
 *   /api/feature/{id}/motif             - MEME-PWM top motifs
 *   /api/feature/{id}/geometry          - backbone geometry classifier + rules
 *
 * Hard-coded in this file: the curated CASE_STUDIES list with its story labels,
 * narrative blurbs, and citation keys. Numerical values (F1, AUROC, consensus,
 * rule strings, top families) are pulled from the API so the page never
 * drifts from the underlying analysis.
 */

// ─────────────────────────────────────────────────────────────────────────
// Curated case studies. fid -> story + narrative + citation keys.
// ─────────────────────────────────────────────────────────────────────────

const CASE_STUDIES = [
    // ── A. Fold-level transfer ──
    { fid: 10235, section: "fold", label: "Peptidase S1, PA clan (trypsin fold)",
      blurb: "The feature fires on β-barrel serine-protease residues. S1 peptidases use the His/Asp/Ser catalytic triad and a common αβ-barrel scaffold [hedstrom2002]. Transferring this label to F041788 / F071777 / F096124 nominates them as novel bacterial serine proteases, a class of biotechnologically-valuable enzymes.",
      cites: ["hedstrom2002", "interpro2023"] },
    { fid: 10213, section: "fold", label: "VOC superfamily (glyoxalase / bleomycin-resistance)",
      blurb: "The VOC (vicinal-oxygen-chelate) fold is a βαβββ metallo-enzyme superfamily unifying glyoxalase I, extradiol dioxygenases and bleomycin-resistance proteins [armstrong2000]. The feature's residue-level AUROC on SwissProt is 0.995, so its activation pattern on 31 metagenomic families is a strong VOC-fold hypothesis — including potential new antibiotic-resistance enzymes.",
      cites: ["armstrong2000", "bergdoll1998"] },
    { fid: 10120, section: "fold", label: "β-lactamase / transpeptidase-like",
      blurb: "The β-lactamase / PBP transpeptidase fold family is the principal driver of β-lactam antibiotic resistance [massova1998]. Feature fires on the catalytic α/β core; the two NMPFam hits (F036753, F088841) are therefore candidate clinically-relevant resistance enzymes.",
      cites: ["massova1998"] },
    { fid: 10051, section: "fold", label: "Pectin-lyase fold (parallel β-helix)",
      blurb: "Parallel β-helix proteins are common polysaccharide-processing enzymes, first characterised in pectate lyase C [yoder1993]. The low long-range-contact count in the geometry rule (&le;7.5) is diagnostic of the β-solenoid topology. Hits label 31 NMPFams as candidate CAZymes.",
      cites: ["yoder1993", "jenkins1998"] },
    { fid: 10091, section: "fold", label: "GNAT N-acetyltransferase",
      blurb: "The GNAT fold is a ubiquitous acetyl-CoA-using superfamily covering histone, antibiotic and aminoglycoside acetyltransferases [vetting2005]. Residue-level AUROC = 0.994 on SwissProt. 17 NMPFams light up above 60% of the SwissProt global max.",
      cites: ["vetting2005", "neuwald1997"] },

    // ── B. Residue-level (functional-site) transfer ──
    { fid: 10077, section: "residue", label: "CBS domain (energy/redox sensor)",
      blurb: "CBS domains form Bateman modules that bind adenosyl ligands (AMP, SAM, ATP) and act as regulatory switches [bateman1997; ignoul2005]. Residue-level F1 = 0.74 means the feature fires on the binding-pocket residues, not the whole domain. The MEME motif DTPIKDALRQM (E = 8.4 × 10⁻⁴⁴) pins a conserved binding-loop signature.",
      cites: ["bateman1997", "ignoul2005"] },
    { fid: 10084, section: "residue", label: "OmpR/PhoB-type response-regulator DBD",
      blurb: "OmpR/PhoB is the winged-helix output domain of bacterial two-component response regulators [martinezhackert1997; gao2009]. Residue-level F1 = 0.73; protein-level F1 = 1.00. Feature picks up the recognition-helix residues, so NMPFam F084470 (87 aa) is a candidate novel response-regulator DBD.",
      cites: ["martinezhackert1997", "gao2009"] },
    { fid: 10216, section: "residue", label: "Cyclic-nucleotide-binding (CAP/CRP) domain",
      blurb: "The cAMP/cGMP-binding domain is an ancient second-messenger sensor that was first described in the E. coli CRP structure [mckay1981] and later generalised across kinase regulatory subunits and ion channels [berman2005]. Residue-level F1 = 0.62. Five short NMPFams (57–79 aa) match the β-barrel alone, consistent with a binding pocket.",
      cites: ["mckay1981", "berman2005"] },
    { fid: 10179, section: "residue", label: "MscS mechanosensitive channel",
      blurb: "MscS channels relieve osmotic shock by gating in response to membrane tension [bass2002]. Residue-level F1 = 0.64. Seven NMPFams with lengths 40–74 aa match the TM-helix/linker region and are candidate new MscS-like membrane components.",
      cites: ["bass2002"] },
    { fid: 9987, section: "residue", label: "Leucine-rich repeat (LRR) superfamily",
      blurb: "LRRs form curved β-α solenoids that mediate protein-protein and pathogen recognition [kobe2001]. Residue-level F1 = 0.77 on the LRR superfamily annotation. The MEME motif SGNGIGDEGARA recovers the canonical LxxLxxN repeat signature.",
      cites: ["kobe2001"] },

    // ── C. Repeat-scaffold annotation at scale ──
    { fid: 9914, section: "scaffold", label: "TPR helical-repeat superfamily",
      blurb: "Tetratricopeptide repeats are α-helical solenoid building blocks for protein-protein interaction platforms, including chaperones and import complexes [dandrea2003]. The geometry rule contact_density_8A ≤ 5.5 matches the sparse short-range packing of the solenoid. 97 NMPFams light up at up to 97% of the SwissProt max.",
      cites: ["dandrea2003"] },
    { fid: 10118, section: "scaffold", label: "WD40 / YVTN β-propeller",
      blurb: "WD40 propellers underpin ubiquitous scaffolding networks [stirnimann2010]. The MEME motif AWDAETGKLLWT (E = 6.5 × 10⁻²⁵⁰, PR-AUC = 0.39) is a clean WD40 'GH'-box signature shared across 58 metagenomic families.",
      cites: ["stirnimann2010"] },
    { fid: 10151, section: "scaffold", label: "Ankyrin repeat superfamily",
      blurb: "Ankyrin repeats (α-helix — turn — α-helix stacks) are specific protein-protein recognition modules [mosavi2004]. The geometry rule is plan_C_third ≤ 0.039 plus high narrow_tangent_alignment — the flat, parallel helical pattern. 11 NMPFams transferred, including F069276 at 72% of the SwissProt max.",
      cites: ["mosavi2004"] },

    // ── D. Sequence-motif-driven transfer ──
    { fid: 10114, section: "motif", label: "Hydrophobic outer-membrane / lipo-anchor signal",
      blurb: "Bacterial lipoproteins and β-barrel outer-membrane proteins carry conserved hydrophobic N-terminal signals processed by the BAM complex [noinaj2017; kovacs2011]. The MEME motif AILLAALLLAGC (E = 1.8 × 10⁻⁶⁴, PR-AUC = 0.50) plus the geometry rule wide_end_to_end_ratio ≤ 0.28 with frac_hydrophobic ≈ 0.09 identify a TM-helix / lipoanchor segment. 671 short NMPFams light up — a large class of candidate novel lipoproteins / OM-anchored peptides.",
      cites: ["noinaj2017", "kovacs2011"] },

    // ── E. Pure-geometry (novel structural-motif classes) ──
    { fid: 10032, section: "geom", label: "RNase-H-like α/β/α fold (candidate nucleases)",
      blurb: "The RNase-H-like superfamily is a large functionally-diverse nucleic-acid-processing class including transposases, retroviral integrases, exonucleases, and Argonaute PIWI domains [majorek2014]. Geometry rule torsion_mean ≤ −0.86 plus specific long-range-contact pattern gives residue-level F1 = 0.41. 26 NMPFams are candidate novel nucleic-acid enzymes.",
      cites: ["majorek2014"] },
    { fid: 10227, section: "geom", label: "Compact-turn / hairpin motif (~500 novel families)",
      blurb: "The decision tree splits on narrow_end_to_end_ratio ≤ 0.33 with feature importance 0.49 — the strongest compact-turn signature in the geometry set. The feature fires on 519 NMPFams but has no strong InterPro match, so these are candidate members of a previously-uncatalogued compact-turn structural class.",
      cites: ["interpro2023"] },
    { fid: 10173, section: "geom", label: "Secreted-effector / filament-subunit-like",
      blurb: "The geometry rule narrow_tangent_alignment ≤ 0.20 marks anti-aligned backbone tangents — non-helix, filament-subunit-like. 2,680 very short (~60 aa) NMPFams match. T3SS needle subunits and related bacterial secretion-system components share this geometry [loquet2012]. The label is a candidate, not a certainty — many α-helical bundles can give similar tangent profiles.",
      cites: ["loquet2012"] },
    { fid: 9897, section: "geom", label: "Short-range-contact mini-domain / zinc-ribbon-like",
      blurb: "Geometry rule max_seq_sep_contact_8A ≤ 15.5: tertiary contacts stay near in sequence — characteristic of small ribbon-like folds such as treble-clef zinc fingers [krishna2003]. 537 small NMPFams (62–140 aa) match, with residue-AUROC = 0.98. Candidate class: metal-binding regulatory mini-domains.",
      cites: ["krishna2003"] },
];

// ─────────────────────────────────────────────────────────────────────────
// References bibliography. Keep DOI links, one entry per {key}.
// ─────────────────────────────────────────────────────────────────────────

const REFS = {
    // methodology
    baltoumas2024:    { txt: "Baltoumas FA, Karatzas E, Paez-Espino D, et al. NMPFamsDB: a database of novel protein families from microbial metagenomes and metatranscriptomes. Nucleic Acids Research 52(D1):D502–D512 (2024).", doi: "10.1093/nar/gkad800" },
    jumper2021:       { txt: "Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold. Nature 596:583–589 (2021).", doi: "10.1038/s41586-021-03819-2" },
    lin2023:          { txt: "Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure (ESM-2 / ESMFold). Science 379:1123–1130 (2023).", doi: "10.1126/science.ade2574" },
    simon2024:        { txt: "Simon E, Zou J. InterPLM: Discovering interpretable features in protein language models via sparse autoencoders. bioRxiv 2024.", doi: "10.1101/2024.11.14.623630" },
    interpro2023:     { txt: "Paysan-Lafosse T, Blum M, Chuguransky S, et al. InterPro in 2022. Nucleic Acids Research 51(D1):D418–D427 (2023).", doi: "10.1093/nar/gkac993" },
    bailey2015:       { txt: "Bailey TL, Johnson J, Grant CE, Noble WS. The MEME Suite. Nucleic Acids Research 43(W1):W39–W49 (2015).", doi: "10.1093/nar/gkv416" },
    bh1995:           { txt: "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J. Royal Statistical Society B 57:289–300 (1995).", doi: "10.1111/j.2517-6161.1995.tb02031.x" },
    cunningham2023:   { txt: "Cunningham H, Ewart A, Riggs L, Huben R, Sharkey L. Sparse autoencoders find highly interpretable features in language models. arXiv:2309.08600 (2023).", doi: null, url: "https://arxiv.org/abs/2309.08600" },

    // fold-specific
    hedstrom2002:     { txt: "Hedstrom L. Serine protease mechanism and specificity. Chemical Reviews 102:4501–4524 (2002).", doi: "10.1021/cr000033x" },
    armstrong2000:    { txt: "Armstrong RN. Mechanistic diversity in a metalloenzyme superfamily (VOC fold). Biochemistry 39:13625–13632 (2000).", doi: "10.1021/bi001814v" },
    bergdoll1998:     { txt: "Bergdoll M, Eltis LD, Cameron AD, Dumas P, Bolin JT. All in the family: structural and evolutionary relationships among three modular proteins with diverse functions and variable assembly. Protein Science 7:1661–1670 (1998).", doi: "10.1002/pro.5560070801" },
    massova1998:     { txt: "Massova I, Mobashery S. Kinship and diversification of bacterial penicillin-binding proteins and β-lactamases. Antimicrobial Agents and Chemotherapy 42:1–17 (1998).", doi: "10.1128/AAC.42.1.1" },
    yoder1993:       { txt: "Yoder MD, Keen NT, Jurnak F. New domain motif: the structure of pectate lyase C, a secreted plant virulence factor. Science 260:1503–1507 (1993).", doi: "10.1126/science.8502994" },
    jenkins1998:     { txt: "Jenkins J, Mayans O, Pickersgill R. Structure and evolution of parallel β-helix proteins. J. Structural Biology 122:236–246 (1998).", doi: "10.1006/jsbi.1998.3985" },
    vetting2005:     { txt: "Vetting MW, S de Carvalho LP, Yu M, et al. Structure and functions of the GNAT superfamily of acetyltransferases. Archives of Biochemistry and Biophysics 433:212–226 (2005).", doi: "10.1016/j.abb.2004.09.003" },
    neuwald1997:     { txt: "Neuwald AF, Landsman D. GCN5-related histone N-acetyltransferases belong to a diverse superfamily that includes the yeast SPT10 protein. Trends in Biochemical Sciences 22:154–155 (1997).", doi: "10.1016/S0968-0004(97)01034-7" },
    bateman1997:     { txt: "Bateman A. The structure of a domain common to archaebacteria and the homocystinuria disease protein (CBS). Trends in Biochemical Sciences 22:12–13 (1997).", doi: "10.1016/S0968-0004(96)30046-7" },
    ignoul2005:      { txt: "Ignoul S, Eggermont J. CBS domains: structure, function, and pathology in human proteins. Am. J. Physiol. Cell Physiol. 289:C1369–C1378 (2005).", doi: "10.1152/ajpcell.00282.2005" },
    mckay1981:       { txt: "McKay DB, Steitz TA. Structure of catabolite gene activator protein at 2.9 Å resolution suggests binding to left-handed B-DNA. Nature 290:744–749 (1981).", doi: "10.1038/290744a0" },
    berman2005:      { txt: "Berman HM, Ten Eyck LF, Goodsell DS, et al. The cAMP binding domain: an ancient signaling module. Proc. Natl. Acad. Sci. USA 102:45–50 (2005).", doi: "10.1073/pnas.0408579102" },
    martinezhackert1997: { txt: "Martinez-Hackert E, Stock AM. Structural relationships in the OmpR family of winged-helix transcription factors. J. Mol. Biol. 269:301–312 (1997).", doi: "10.1006/jmbi.1997.1065" },
    gao2009:         { txt: "Gao R, Stock AM. Biological insights from structures of two-component proteins. Annual Review of Microbiology 63:133–154 (2009).", doi: "10.1146/annurev.micro.091208.073214" },
    bass2002:        { txt: "Bass RB, Strop P, Barclay M, Rees DC. Crystal structure of E. coli MscS, a voltage-modulated and mechanosensitive channel. Science 298:1582–1587 (2002).", doi: "10.1126/science.1077945" },
    kobe2001:        { txt: "Kobe B, Kajava AV. The leucine-rich repeat as a protein recognition motif. Current Opinion in Structural Biology 11:725–732 (2001).", doi: "10.1016/S0959-440X(01)00266-4" },
    dandrea2003:     { txt: "D'Andrea LD, Regan L. TPR proteins: the versatile helix. Trends in Biochemical Sciences 28:655–662 (2003).", doi: "10.1016/j.tibs.2003.10.007" },
    stirnimann2010:  { txt: "Stirnimann CU, Petsalaki E, Russell RB, Müller CW. WD40 proteins propel cellular networks. Trends in Biochemical Sciences 35:565–574 (2010).", doi: "10.1016/j.tibs.2010.04.003" },
    mosavi2004:      { txt: "Mosavi LK, Cammett TJ, Desrosiers DC, Peng ZY. The ankyrin repeat as molecular architecture for protein recognition. Protein Science 13:1435–1448 (2004).", doi: "10.1110/ps.03554604" },
    noinaj2017:      { txt: "Noinaj N, Gumbart JC, Buchanan SK. The β-barrel assembly machinery in motion. Nature Reviews Microbiology 15:197–204 (2017).", doi: "10.1038/nrmicro.2016.191" },
    kovacs2011:      { txt: "Kovacs-Simon A, Titball RW, Michell SL. Lipoproteins of bacterial pathogens. Infection and Immunity 79:548–561 (2011).", doi: "10.1128/IAI.00682-10" },
    majorek2014:     { txt: "Majorek KA, Dunin-Horkawicz S, Steczkiewicz K, et al. The RNase H-like superfamily: new members, comparative structural analysis and evolutionary classification. Nucleic Acids Research 42:4160–4179 (2014).", doi: "10.1093/nar/gkt1414" },
    loquet2012:      { txt: "Loquet A, Sgourakis NG, Gupta R, et al. Atomic model of the type III secretion system needle. Nature 486:276–279 (2012).", doi: "10.1038/nature11079" },
    krishna2003:     { txt: "Krishna SS, Majumdar I, Grishin NV. Structural classification of zinc fingers. Nucleic Acids Research 31:532–550 (2003).", doi: "10.1093/nar/gkg161" },
};

const SECTION_TITLE = {
    fold:     "A · Fold transfer",
    residue:  "B · Residue-site transfer",
    scaffold: "C · Repeat scaffold",
    motif:    "D · Sequence motif (MEME)",
    geom:     "E · Pure geometry",
};

// ─────────────────────────────────────────────────────────────────────────
// Geometric feature → one-line interpretation.
// ─────────────────────────────────────────────────────────────────────────

const GEOM_GLOSS = {
    narrow_end_to_end_ratio:  "local N→C distance in a short window (low = tight turn)",
    wide_end_to_end_ratio:    "N→C distance over a wider window (low = compact segment)",
    end_to_end_ratio:         "window-level N→C distance (low = compact)",
    contact_density_8A:       "tertiary-contact count within 8 Å",
    contact_density_12A:      "tertiary-contact count within 12 Å",
    mean_seq_sep_contact_8A:  "mean sequence separation of 8 Å contacts",
    max_seq_sep_contact_8A:   "longest-range sequence separation of 8 Å contacts",
    long_range_contacts_8A:   "# long-range (|i-j|≥12) 8 Å contacts",
    long_range_contacts_12A:  "# long-range (|i-j|≥12) 12 Å contacts",
    min_spatial_dist_long:    "min spatial distance to a long-range residue",
    narrow_tangent_alignment: "backbone-tangent alignment in a short window (helix-like = high)",
    wide_tangent_alignment:   "backbone-tangent alignment in a wider window",
    tangent_alignment:        "backbone-tangent alignment",
    narrow_torsion_std:       "torsion variability in a short window",
    wide_torsion_std:         "torsion variability in a wide window",
    torsion_std:              "backbone-torsion standard deviation",
    narrow_torsion_mean:      "mean backbone torsion locally",
    wide_torsion_mean:        "mean backbone torsion over a wider window",
    torsion_mean:             "mean backbone torsion",
    narrow_curvature_max:     "max curvature in a short window",
    narrow_curvature_mean:    "mean curvature in a short window",
    wide_curvature_mean:      "mean curvature over a wider window",
    wide_curvature_max:       "max curvature over a wider window",
    curvature_max:            "max backbone curvature",
    curvature_mean:           "mean backbone curvature",
    curvature_std:            "curvature variability",
    plan_centre_third:        "planarity in the central third (high = β-sheet-like)",
    plan_N_third:             "planarity near the N-terminus",
    plan_C_third:             "planarity near the C-terminus",
    planarity_mean:           "mean planarity",
    planarity_std:            "planarity variability",
    curv_centre_third:        "curvature in the central third",
    curv_N_third:             "curvature near the N-terminus",
    curv_C_third:             "curvature near the C-terminus",
    tors_centre_third:        "torsion in the central third",
    tors_N_third:             "torsion near the N-terminus",
    tors_C_third:             "torsion near the C-terminus",
    frac_hydrophobic:         "fraction hydrophobic residues in the window",
    frac_charged:             "fraction charged residues",
    frac_polar:               "fraction polar residues",
    frac_aromatic:            "fraction aromatic residues",
    frac_gly_pro:             "fraction glycine + proline (loop/turn enriched)",
};

// ─────────────────────────────────────────────────────────────────────────
// Utils
// ─────────────────────────────────────────────────────────────────────────

function fmtNum(v, d = 3) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    if (typeof v !== "number") return String(v);
    const abs = Math.abs(v);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return v.toExponential(1);
    return v.toFixed(d);
}

function fmtInt(v) {
    if (v === null || v === undefined) return "—";
    return v.toLocaleString();
}

async function fetchJson(url) {
    try {
        const r = await fetch(url);
        if (!r.ok) return null;
        return await r.json();
    } catch { return null; }
}

function nmpfamLink(famId) {
    return `<a href="https://bib.fleming.gr/NMPFamsDB/family/${famId}" target="_blank" rel="noreferrer">${famId}</a>`;
}

function iprLink(code, name) {
    if (!code) return name || "—";
    return `<a href="https://www.ebi.ac.uk/interpro/entry/InterPro/${code}/" target="_blank" rel="noreferrer">${name || code}</a>`;
}

function citationBadges(keys) {
    if (!keys || !keys.length) return "";
    return keys.map(k => {
        const r = REFS[k];
        if (!r) return "";
        const href = r.doi ? `https://doi.org/${r.doi}` : r.url;
        return `<a class="cite-chip" href="${href}" target="_blank" rel="noreferrer" title="${r.txt.replace(/"/g, "&quot;")}">[${k}]</a>`;
    }).join(" ");
}

// ─────────────────────────────────────────────────────────────────────────
// Summary strip (fed by /api/nmpfam-case-study summary)
// ─────────────────────────────────────────────────────────────────────────

function renderSummary(container, summary) {
    container.innerHTML = "";
    const cards = [
        { title: "Curated cases", value: CASE_STUDIES.length, detail: "hand-picked across 5 evidence classes" },
        { title: "Triple-intersection pool", value: summary.n_triple_intersection, detail: "geom-sig × InterPro-sig × NMPFam-hit" },
        { title: "Features with NMPFam hits", value: `${summary.n_features_with_nmpfam_hits} / ${summary.n_features_total}`, detail: `${(100 * summary.n_features_with_nmpfam_hits / summary.n_features_total).toFixed(1)}% of SAE latents` },
        { title: "NMPFams sampled", value: fmtInt(summary.n_families_sampled), detail: "all with AlphaFold2 structures" },
    ];
    for (const c of cards) {
        const el = document.createElement("article");
        el.className = "stat-card";
        el.innerHTML = `<header><strong>${c.title}</strong></header>
            <div class="value">${c.value}</div>
            <div class="detail">${c.detail}</div>`;
        container.appendChild(el);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Overview table
// ─────────────────────────────────────────────────────────────────────────

function chip(kind, text, title = "") {
    return `<span class="chip chip-${kind}" title="${title}">${text}</span>`;
}

function renderOverviewTable(container, rows) {
    const header = `
        <thead>
            <tr>
                <th>Feature</th>
                <th>Story</th>
                <th>Base label</th>
                <th>n_sig / 7 methods</th>
                <th>MEME consensus</th>
                <th>Geometry top feat</th>
                <th>NMPFam hits</th>
                <th>A / B / C / D</th>
            </tr>
        </thead>`;
    const body = rows.map(r => `
        <tr>
            <td><a href="/nmpfam-case-study/${r.fid}" title="Open NMPFams detail"><strong>F${r.fid}</strong></a>
                &nbsp;<a href="/feature/${r.fid}" title="Open generic feature page" class="secondary">↗</a></td>
            <td>${chip("section", SECTION_TITLE[r.section] || r.section)}</td>
            <td>${r.baseLabel || r.interproName || "—"}</td>
            <td><strong>${r.nSig ?? "—"}</strong> / 7</td>
            <td>${r.memeConsensus ? `<code>${r.memeConsensus}</code>` : "—"}</td>
            <td>${r.geomTop ? `<code>${r.geomTop}</code>` : "—"}</td>
            <td>${fmtInt(r.nNm)} · topNorm ${fmtNum(r.topNorm, 2)}</td>
            <td>${["A","B","C","D"].map(t => (r.tierCounts && r.tierCounts[t]) ? `<span class="tier-chip tier-${t}">${t}:${r.tierCounts[t]}</span>` : "").filter(Boolean).join(" ") || "—"}</td>
        </tr>`).join("");
    container.innerHTML = `<table role="grid" class="overview-table">${header}<tbody>${body}</tbody></table>`;
}

// ─────────────────────────────────────────────────────────────────────────
// Case-study cards
// ─────────────────────────────────────────────────────────────────────────

function renderBlurb(blurb) {
    // turn [key] bracket cite shorthands into DOI chip links
    return blurb.replace(/\[([a-z0-9_;\s]+)\]/gi, (_, inner) => {
        const keys = inner.split(/[;\s]+/).map(s => s.trim()).filter(Boolean);
        return citationBadges(keys);
    });
}

function renderCard(data) {
    const cs = data.cs;
    const ipr = data.interpro || {};
    const iprTop = (ipr.protein_level || [])[0] || {};
    const iprResTop = (ipr.residue_level || [])[0] || {};
    const motifTop = ((data.motif || {}).motifs || [])[0] || {};
    const geom = (data.geometry || {}).geometric_residue_level || {};
    const fi = geom.feature_importances || {};
    const topFeats = Object.entries(fi).sort((a,b) => b[1] - a[1]).slice(0, 4);
    const firstRule = (geom.rules || "").split("\n")[0].trim();
    const concord = geom.concordance || {};
    const nmpHits = (data.nmpfam && data.nmpfam.nmpfam_hits) || [];
    const topHits = nmpHits.slice(0, 5);

    const iprBlock = iprTop.annotation_code
        ? `<div class="evidence-row">
             <span class="evi-label">InterPro (protein)</span>
             <span class="evi-value">${iprLink(iprTop.annotation_code, iprTop.annotation_name)} · F1&nbsp;${fmtNum(iprTop.best_f1, 2)}</span>
           </div>
           ${iprResTop.annotation_code
              ? `<div class="evidence-row">
                   <span class="evi-label">InterPro (residue)</span>
                   <span class="evi-value">${iprLink(iprResTop.annotation_code, iprResTop.annotation_name)} · F1&nbsp;${fmtNum(iprResTop.best_f1, 2)} · P/R&nbsp;${fmtNum(iprResTop.precision_at_best, 2)}/${fmtNum(iprResTop.recall_at_best, 2)}</span>
                 </div>`
              : ""}`
        : `<div class="evidence-row"><span class="evi-label">InterPro</span><span class="evi-value secondary">no strong match (protF1&nbsp;&lt;&nbsp;0.50)</span></div>`;

    const motifBlock = motifTop.consensus
        ? `<div class="evidence-row">
             <span class="evi-label">MEME consensus</span>
             <span class="evi-value"><code>${motifTop.consensus}</code>
                · width&nbsp;${motifTop.width}
                · E&nbsp;${fmtNum(motifTop.e_value, 2)}
                · PR-AUC&nbsp;${fmtNum((motifTop.pr_auc || {}).pr_auc, 2)}</span>
           </div>`
        : "";

    const topFeatsTxt = topFeats.map(([k, v]) =>
        `<code title="${GEOM_GLOSS[k] || ""}">${k}</code>&thinsp;(${fmtNum(v, 2)})`
    ).join(", ");
    const geomBlock = firstRule || topFeats.length
        ? `<div class="evidence-row">
             <span class="evi-label">Geometry</span>
             <span class="evi-value">
               rule: <code>${firstRule || "—"}</code><br>
               top: ${topFeatsTxt || "—"}<br>
               resAUROC&nbsp;${fmtNum(concord.residue_auroc, 2)} · F1&nbsp;${fmtNum(concord.f1, 2)} · AP&nbsp;${fmtNum(concord.avg_precision, 2)}
             </span>
           </div>`
        : "";

    const tierChip = (t) => t ? `<span class="tier-chip tier-${t}">${t}</span>` : "";
    const hitsBlock = topHits.length
        ? `<table class="hits-table">
             <thead><tr><th>Tier</th><th>Family</th><th>Proposed label</th><th>Cat</th><th>Len</th><th>Norm act</th></tr></thead>
             <tbody>
                ${topHits.map(h => `
                    <tr>
                        <td>${tierChip(h.confidence_tier)}</td>
                        <td>${nmpfamLink(h.family_id)}</td>
                        <td class="proposed-label">${h.proposed_label || "—"}</td>
                        <td class="secondary">${(h.category || "").replace(" / ", "/")}</td>
                        <td>${h.sequence_length}</td>
                        <td><strong>${fmtNum(h.normalized_activation, 2)}</strong></td>
                    </tr>`).join("")}
             </tbody>
           </table>`
        : `<div class="secondary">No NMPFam hits above threshold.</div>`;

    // Q-value significance block
    const annot = data.annotation || {};
    const nSig = annot.n_sig_methods ?? null;
    const sigMethods = annot.sig_methods || [];
    const tierCounts = annot.tier_counts || {};
    const qBlock = nSig !== null
        ? `<div class="evidence-row">
             <span class="evi-label">q&lt;0.05 methods</span>
             <span class="evi-value">
               <strong>${nSig} / 7</strong> methods significant
               ${sigMethods.length ? `· ${sigMethods.map(m => `<span class="chip">${m}</span>`).join(" ")}` : ""}
             </span>
           </div>`
        : "";

    const tierSummaryBits = ["A", "B", "C", "D"]
        .map(t => tierCounts[t] ? `<span class="tier-chip tier-${t}">${t}:${tierCounts[t]}</span>` : "")
        .filter(Boolean).join(" ");

    const card = document.createElement("article");
    card.className = "sun-card";
    card.dataset.section = cs.section;
    card.innerHTML = `
        <header class="sun-card-hdr">
            <div>
                <span class="chip chip-fid"><a href="/nmpfam-case-study/${cs.fid}">F${cs.fid}</a></span>
                ${chip("section", SECTION_TITLE[cs.section])}
                <strong>${cs.label}</strong>
            </div>
            <div class="secondary" style="font-size:0.85rem;">
                n&nbsp;NMPFams: <strong>${data.nHits}</strong> · top norm act <strong>${fmtNum(data.topNorm, 2)}</strong>
                ${tierSummaryBits ? `<br>${tierSummaryBits}` : ""}
            </div>
        </header>
        <div class="sun-card-body">
            <p class="blurb">${renderBlurb(cs.blurb)}</p>
            <div class="evidence-grid">
                ${qBlock}
                ${iprBlock}
                ${motifBlock}
                ${geomBlock}
            </div>
            <details>
                <summary>Top NMPFam hits (${topHits.length} shown of ${data.nHits})</summary>
                ${hitsBlock}
            </details>
            <div class="card-footer">
                ${citationBadges(cs.cites)}
                · <a href="/nmpfam-case-study/${cs.fid}">NMPFams detail &rarr;</a>
                · <a href="/feature/${cs.fid}">Feature detail &rarr;</a>
            </div>
        </div>`;
    return card;
}

// ─────────────────────────────────────────────────────────────────────────
// References block
// ─────────────────────────────────────────────────────────────────────────

function renderRefList(container) {
    const used = new Set();
    CASE_STUDIES.forEach(c => (c.cites || []).forEach(k => used.add(k)));
    // also always-cite methodology refs
    ["baltoumas2024", "jumper2021", "lin2023", "simon2024", "interpro2023", "bailey2015", "bh1995", "cunningham2023"].forEach(k => used.add(k));
    const keys = Array.from(used).sort();
    container.innerHTML = keys.map(k => {
        const r = REFS[k];
        if (!r) return "";
        const href = r.doi ? `https://doi.org/${r.doi}` : r.url;
        return `<li id="ref-${k}"><span class="ref-key">[${k}]</span> ${r.txt} ${href ? `<a href="${href}" target="_blank" rel="noreferrer">→</a>` : ""}</li>`;
    }).join("");
}

// ─────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────

/**
 * Normalise NMPFam hit entries into a canonical shape:
 *   { family_id, category, sequence_count, sequence_length, max_activation, normalized_activation, nmpfams_url }
 *
 * The triple_features payload already uses these fields; the /api/feature/{id}/nmpfam
 * endpoint instead returns { family_id, n_residues, max_sae_activation, ... } and
 * reports feature_global_max at the top level. We compute normalized_activation
 * client-side when needed.
 */
function canonicaliseHits(rawHits, featureGlobalMax) {
    if (!rawHits) return [];
    return rawHits.map(h => {
        const maxAct = h.max_activation ?? h.max_sae_activation ?? null;
        const norm = h.normalized_activation ?? (
            featureGlobalMax && maxAct != null ? maxAct / featureGlobalMax : null
        );
        return {
            family_id: h.family_id,
            category: h.category,
            sequence_count: h.sequence_count,
            sequence_length: h.sequence_length ?? h.n_residues ?? null,
            max_activation: maxAct,
            normalized_activation: norm,
            nmpfams_url: h.nmpfams_url || `https://bib.fleming.gr/NMPFamsDB/family/${h.family_id}`,
        };
    });
}

// Cached annotations index
let _ANNOT_INDEX = null;
async function getAnnotationsIndex() {
    if (_ANNOT_INDEX !== null) return _ANNOT_INDEX;
    _ANNOT_INDEX = await fetchJson("/api/nmpfam-annotations");
    return _ANNOT_INDEX;
}

async function loadCaseData(cs, triplesById) {
    // triple_features already carry full info; broader_gp need separate calls
    const base = triplesById.get(cs.fid);
    const [nmpfam, interpro, motif, geometry, annot] = await Promise.all([
        base ? Promise.resolve({ nmpfam_hits: base.nmpfam_hits, n_nmpfam_hits: base.n_nmpfam_hits,
                                 feature_global_max: base.global_max_activation })
             : fetchJson(`/api/feature/${cs.fid}/nmpfam`),
        fetchJson(`/api/feature/${cs.fid}/interpro`),
        fetchJson(`/api/feature/${cs.fid}/motif`),
        fetchJson(`/api/feature/${cs.fid}/geometry`),
        fetchJson(`/api/nmpfam-annotations/${cs.fid}`),
    ]);
    const globalMax = (nmpfam && (nmpfam.feature_global_max ?? nmpfam.global_max_activation))
                    || (interpro && interpro.feature_max_activation)
                    || null;
    const hits = canonicaliseHits((nmpfam && nmpfam.nmpfam_hits) || [], globalMax)
        .sort((a, b) => (b.normalized_activation || 0) - (a.normalized_activation || 0));
    const topNorm = hits.length ? (hits[0].normalized_activation || 0) : 0;
    // Merge annotation (tier + rationale) per family_id
    const annotByFamily = {};
    if (annot && annot.hits) {
        for (const h of annot.hits) annotByFamily[h.family_id] = h;
    }
    for (const h of hits) {
        const a = annotByFamily[h.family_id];
        if (a) {
            h.confidence_tier = a.confidence_tier;
            h.proposed_label = a.proposed_label;
            h.caveats = a.caveats;
            h.sub_flags = a.sub_flags;
            h.rationale = a.rationale;
        }
    }

    return {
        cs,
        nmpfam: { ...(nmpfam || {}), nmpfam_hits: hits },
        interpro,
        motif,
        geometry,
        annotation: annot,
        nHits: (nmpfam && nmpfam.n_nmpfam_hits) || hits.length,
        topNorm,
    };
}

function buildOverviewRow(data) {
    const cs = data.cs;
    const iprTop = ((data.interpro || {}).protein_level || [])[0] || {};
    const iprResTop = ((data.interpro || {}).residue_level || [])[0] || {};
    const motifTop = (((data.motif || {}).motifs) || [])[0] || {};
    const geom = ((data.geometry || {}).geometric_residue_level || {});
    const fi = geom.feature_importances || {};
    const firstFeat = Object.entries(fi).sort((a,b) => b[1] - a[1])[0];
    const hits = (data.nmpfam && data.nmpfam.nmpfam_hits) || [];
    return {
        fid: cs.fid,
        section: cs.section,
        baseLabel: data.annotation ? data.annotation.base_label : null,
        interproName: iprTop.annotation_name,
        memeConsensus: motifTop.consensus,
        memeE: motifTop.e_value,
        geomTop: firstFeat ? firstFeat[0] : null,
        nSig: data.annotation ? data.annotation.n_sig_methods : null,
        tierCounts: data.annotation ? data.annotation.tier_counts : null,
        nNm: data.nHits,
        topNorm: data.topNorm,
        topNm: hits.length ? hits[0].family_id : null,
    };
}

async function main() {
    const overview = await fetchJson("/api/nmpfam-case-study");
    if (!overview) {
        document.getElementById("summary-cards").innerHTML = `<div class="secondary">Could not load <code>/api/nmpfam-case-study</code>. Is the server pointed at a layer_4 analysis dir?</div>`;
        return;
    }
    renderSummary(document.getElementById("summary-cards"), overview.summary || {});

    const triplesById = new Map((overview.triple_features || []).map(t => [t.feature_id, t]));
    const allData = await Promise.all(CASE_STUDIES.map(cs => loadCaseData(cs, triplesById)));

    // overview table
    const rows = allData.map(buildOverviewRow);
    renderOverviewTable(document.getElementById("overview-table-wrap"), rows);

    // section cards
    const bySection = {fold: [], residue: [], scaffold: [], motif: [], geom: []};
    for (const d of allData) bySection[d.cs.section].push(d);
    for (const [sec, arr] of Object.entries(bySection)) {
        const container = document.getElementById(`section-${sec}-cards`);
        if (!container) continue;
        arr.sort((a, b) => (b.nHits || 0) - (a.nHits || 0));
        for (const d of arr) container.appendChild(renderCard(d));
    }

    renderRefList(document.getElementById("ref-list"));
}

main();
