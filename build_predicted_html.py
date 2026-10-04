#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_predicted_html.py -- erzeugt predicted.html aus predictions.json.

Die Seite enthaelt:
  * eine Karte fuer ganz Mecklenburg-Vorpommern,
  * eine Detailkarte fuer den Landkreis Ludwigslust-Parchim,
  * Auswahlfelder fuer Partei, Modell und Karteninhalt
    (Prognose / Ist-Ergebnis / Ist minus Prognose),
  * Kennzahlen, Koeffizienten mit p-Werten, Rechenbeispiel und Gemeindetabelle,
  * eine ausfuehrliche Erklaerung des Verfahrens mit Quellenangaben.

Alle Karten sind reines SVG. Die Gemeindegeometrie ist partei- und modellunabhaengig
und wird nur einmal eingebettet; gewechselt wird nur die Fuellfarbe. Deshalb
bleibt die Datei trotz 7 Parteien x 5 Modelle x 3 Karteninhalten klein.

Aufruf: python3 build_predicted_html.py
"""

from __future__ import annotations

import html
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from build_results_v2 import build_geometries, load_map_paths

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "predictions.json"
OUTPUT = ROOT / "index.html"

PARTY_ORDER = ["GRÜNE", "SPD", "AfD", "CDU", "FDP", "DIE LINKE", "BSW"]
PARTY_COLORS = {
    "GRÜNE": "#64a12d", "SPD": "#e2001a", "AfD": "#009ee0", "CDU": "#333333",
    "FDP": "#c9b800", "DIE LINKE": "#be3075", "BSW": "#5a6570",
}
PREDICTOR_LABELS = {
    "LTW16": "LTW 2016", "BTW17": "BTW 2017", "BTW21": "BTW 2021",
    "LTW21": "LTW 2021", "LRW25": "LRW 2025", "BTW25": "BTW 2025",
}
CONTEXT_LABELS = {
    "einwohner": "Einwohner", "flaeche_km2": "Fläche (km²)",
    "dichte": "Einw. je km²", "ueber65": "65 Jahre und älter (%)",
    "unter18": "unter 18 Jahren (%)", "auslaender": "Ausländeranteil (%)",
    "durchschnittsalter": "Ø Alter", "miete_pro_qm": "Miete (€/m²)",
    "eigentum": "Eigentümerquote (%)", "leerstandsquote": "Leerstandsquote (%)",
    "mehrfamilienhausquote": "Mehrfamilienhausquote (%)",
    "haushalt_groesse": "Ø Haushaltsgröße", "single_quote": "Singlequote (%)",
    "haushalte": "Haushalte",
}
SOCIO_KEYS = {"ln_einwohner", "ln_flaeche", "ln_dichte", "ueber65", "auslaender",
              "miete_pro_qm", "eigentum", "haushalt_groesse"}

MISSING_FILL = "#e9edf0"

# Karteninhalte: key -> (Titel, Art, Farbleiter, CSS-Verlauf)
METRICS: Dict[str, dict] = {
    "residual": {
        "label": "Ist − Prognose",
        "title": "Wo hat das Modell daneben gelegen?",
        "kind": "diverging",
        "cmap": "RdYlGn",
        "unit": "Prozentpunkte",
        "note": "Grün = die Partei erzielte mehr, als das Modell vorausgesagt hat. "
                "Rot = sie erzielte weniger.",
        "gradient": "linear-gradient(90deg,#a50026,#d73027,#f46d43,#fdae61,#ffffbf,"
                    "#a6d96a,#1a9850,#006837)",
    },
    "predicted": {
        "label": "Prognose (Modell)",
        "title": "Was sagt das Modell voraus?",
        "kind": "sequential",
        "cmap": "YlOrBr",
        "unit": "Prozent",
        "note": "Je dunkler desto höher der vorausgesagte Anteil. Die Skala beginnt bei 0 und endet "
                "beim 98. Perzentil der Werte dieser Partei.",
        "gradient": "linear-gradient(90deg,#ffffe5,#fee391,#fec44f,#fe9929,#cc4c02,#662506)",
    },
    "actual": {
        "label": "Ist-Ergebnis LTW26",
        "title": "Was ist tatsächlich herausgekommen?",
        "kind": "sequential",
        "cmap": "PuBu",
        "unit": "Prozent",
        "note": "Je dunkler desto höher der amtliche Anteil aus der LTW26. Die Skala beginnt bei 0 "
                "und endet beim 98. Perzentil der Werte dieser Partei.",
        "gradient": "linear-gradient(90deg,#f7fcfd,#ccece6,#a1d9b4,#67b9c0,#3690b0,#045a8d)",
    },
}
DEFAULT_PARTY = "GRÜNE"
DEFAULT_METRIC = "residual"


# --------------------------------------------------------------------------
# Formatierung (deutsch)
# --------------------------------------------------------------------------
def de(value: Optional[float], digits: int = 2, dash: str = "–") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    rounded = round(float(value), digits)
    if rounded == 0:
        rounded = 0.0        # verhindert die negative Null ("-0,00")
    return f"{rounded:.{digits}f}".replace(".", ",")


def de_int(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    return f"{value:,.0f}".replace(",", ".")


def de_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    return f"{value:.{digits}f}".replace(".", ",") + "&nbsp;%"


def signed(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    text = f"{abs(value):.{digits}f}".replace(".", ",")
    if value > 1e-9:
        return "+" + text
    if value < -1e-9:
        return "−" + text
    return f"{0:.{digits}f}".replace(".", ",")


def stars(p_value: Optional[float]) -> str:
    if p_value is None or not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.&nbsp;s."


# --------------------------------------------------------------------------
# SVG-Geometrie (einmalig, unabhaengig von Partei, Modell und Karteninhalt)
# --------------------------------------------------------------------------
def ring_path(polygons) -> str:
    parts = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float)
        if len(points) < 3:
            continue
        parts.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in points) + "Z")
    return "".join(parts)


def build_region(geometries, boundaries, codes: Optional[set]) -> dict:
    if codes is not None:
        geometries = {c: p for c, p in geometries.items() if c in codes}
    shapes = []
    for code, polygons in geometries.items():
        path = ring_path(polygons)
        if path:
            shapes.append({"c": str(code), "d": path})
    outlines = []
    plotted: List[np.ndarray] = []
    for key, polygons in boundaries.items():
        is_amt = str(key).startswith("A")
        if codes is not None and not is_amt and str(key) != "76":
            continue
        for polygon in polygons:
            points = np.asarray(polygon, dtype=float)
            if codes is not None and not is_amt and plotted:
                joined = np.concatenate(plotted, axis=0)
                if not (points[:, 0].min() >= joined[:, 0].min() - 60
                        and points[:, 0].max() <= joined[:, 0].max() + 60
                        and points[:, 1].min() >= joined[:, 1].min() - 60
                        and points[:, 1].max() <= joined[:, 1].max() + 60):
                    continue
            outlines.append({"d": ring_path([points]), "amt": bool(is_amt)})
            plotted.append(points)
    all_points = [np.asarray(p, dtype=float) for polys in geometries.values() for p in polys]
    joined = np.concatenate(all_points, axis=0)
    pad = 14.0 if codes is not None else 0.0
    view_box = [float(joined[:, 0].min()) - pad, float(joined[:, 1].min()) - pad,
                float(joined[:, 0].max() - joined[:, 0].min()) + 2 * pad,
                float(joined[:, 1].max() - joined[:, 1].min()) + 2 * pad]
    return {
        "shapes": shapes,
        "outlines": outlines,
        "viewBox": " ".join(f"{v:.1f}" for v in view_box),
        "codes": [s["c"] for s in shapes],
    }


# --------------------------------------------------------------------------
# Farben
# --------------------------------------------------------------------------
def to_hex(rgba) -> str:
    return "#%02x%02x%02x" % tuple(int(round(255 * channel)) for channel in rgba[:3])


def color_scale(values: Dict[str, Optional[float]], codes: Sequence[str],
                metric: str) -> Tuple[Dict[str, str], float, float]:
    spec = METRICS[metric]
    codes = list(codes)

    def number(value) -> Optional[float]:
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    # Die Farbleiter wird aus allen Gemeinden der Region bestimmt, nicht nur
    # aus denen mit einem Wert: sonst haengt die Skala an der Abdeckung.
    present = [v for v in (number(values.get(code)) for code in codes) if v is not None]
    if not present:                      # Rueckfall auf das ganze Land
        present = [v for v in (number(v) for v in values.values()) if v is not None]
    cmap = plt.get_cmap(spec["cmap"])
    if spec["kind"] == "diverging":
        bound = max(float(np.nanpercentile(np.abs(present), 98)) if present else 1.0, 0.4)
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
        lo, hi = -bound, bound
    else:
        hi = max(float(np.nanpercentile(present, 98)) if present else 1.0, 0.5)
        lo = 0.0
        norm = Normalize(vmin=lo, vmax=hi)
    fills = {}
    for code in codes:
        value = number(values.get(code))
        fills[code] = to_hex(cmap(norm(value))) if value is not None else MISSING_FILL
    return fills, lo, hi


# --------------------------------------------------------------------------
# HTML-Bausteine
# --------------------------------------------------------------------------
def svg_region(region: dict, fills: Dict[str, str]) -> str:
    body = [f'<path class="muni" data-code="{s["c"]}" fill="{fills.get(s["c"], MISSING_FILL)}" d="{s["d"]}"/>'
            for s in region["shapes"]]
    body += [f'<path class="line {"amt" if o["amt"] else "county"}" d="{o["d"]}"/>'
             for o in region["outlines"]]
    return ('<svg class="map-svg" viewBox="' + region["viewBox"] + '" '
            'xmlns="http://www.w3.org/2000/svg" role="img" '
            'aria-label="Karte der Gemeindegebiete" preserveAspectRatio="xMidYMid meet">'
            + "".join(body) + "</svg>")


def map_block(layer_id: str, title: str, subtitle: str, note: str, svg: str) -> str:
    return f"""<section class="card" data-layer="{layer_id}">
  <div class="card-head">
    <h2>{title}</h2>
    <p>{subtitle}</p>
  </div>
  <div class="map-holder">{svg}</div>
  <p class="map-note" data-note="{layer_id}">{note}</p>
</section>"""


CSS = """
  :root {
    --ink:#16232b; --muted:#5d6b72; --line:#dfe6ea; --bg:#f5f7f8;
    --pos:#1a7f4b; --neg:#b3261e; --card:#ffffff;
  }
  * { box-sizing:border-box; }
  body { margin:0; padding:0 0 64px; background:var(--bg); color:var(--ink);
    font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  header { background:#12303d; color:#fff; padding:36px 24px 30px; }
  header .wrap, main { max-width:1180px; margin:0 auto; }
  header h1 { margin:0 0 8px; font-size:27px; letter-spacing:-.01em; }
  header p { margin:0; color:#b9cdd6; max-width:920px; }
  main { padding:26px 24px 0; }
  /* Der Footer traegt nur das Tracking-Skript und ist daher unsichtbar;
     die Regeln halten ihn trotzdem explizit ohne Abstaende. */
  footer { display:block; margin:0; padding:0; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
    padding:22px 24px 20px; margin:0 0 26px; }
  .card-head h2 { margin:0 0 4px; font-size:20px; }
  .card-head p { margin:0 0 16px; color:var(--muted); font-size:14px; }
  .map-holder { border:1px solid var(--line); border-radius:6px; background:#fff; padding:4px; }
  .map-svg { display:block; width:100%; height:auto; }
  .map-svg path.muni { stroke:#ffffff; stroke-width:.35; stroke-linejoin:round;
    pointer-events:all; cursor:crosshair; }
  .map-svg path.muni:hover { fill:rgba(18,48,61,.12); stroke:#12303d; stroke-width:1.6;
    vector-effect:non-scaling-stroke; }
  .map-svg path.line { fill:none; stroke-linejoin:round; vector-effect:non-scaling-stroke;
    pointer-events:none; }
  .map-svg path.line.county { stroke:#5d6b72; stroke-width:.7; opacity:.9; }
  .map-svg path.line.amt { stroke:#a3adb2; stroke-width:.45; opacity:.9; }
  .map-note { margin:12px 0 0; color:var(--muted); font-size:13px; }
  .legend { display:flex; flex-direction:column; gap:10px; margin:0 0 18px;
    font-size:13px; color:var(--muted); }
  .legend-row { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  .legend-votes { background:var(--card); border:1px solid var(--line); border-radius:8px;
    padding:11px 14px; color:var(--ink); line-height:1.6; }
  .legend-votes strong { color:var(--ink); }
  .legend-votes .num { font-variant-numeric:tabular-nums; font-weight:600; color:var(--ink); }
  .bar { height:14px; flex:0 0 320px; border-radius:7px; }
  .kpi { display:grid; grid-template-columns:repeat(auto-fit,minmax(178px,1fr)); gap:14px; margin:0 0 8px; }
  .kpi div { background:#fbfcfc; border:1px solid var(--line); border-radius:8px; padding:13px 15px; }
  .kpi span { display:block; color:var(--muted); font-size:11.5px; text-transform:uppercase;
    letter-spacing:.03em; }
  .kpi b { font-size:20px; font-variant-numeric:tabular-nums; }
  h3 { font-size:16px; margin:26px 0 8px; }
  h3:first-child { margin-top:6px; }
  table { width:100%; border-collapse:collapse; margin:0 0 6px; font-size:13.5px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line);
    vertical-align:top; }
  th { background:#eef2f4; font-weight:600; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
  tr.sel > td { background:#f2f8f2; }
  code { background:#eef2f4; padding:1px 5px; border-radius:4px; font-size:12.5px; }
  .pos { color:var(--pos); } .neg { color:var(--neg); }
  header .hl-pos { color:#5fe0a0; } header .hl-neg { color:#ff9c92; }
  .formula { background:#f0f4f6; border-left:3px solid #12303d; padding:12px 15px; margin:10px 0 14px;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; overflow-x:auto; }
  .controls { display:flex; gap:20px; align-items:flex-end; flex-wrap:wrap;
    margin:0 0 18px; padding:16px 18px; background:#fbfcfc;
    border:1px solid var(--line); border-radius:8px; }
  .controls label, .controls .label-spacer { display:block; font-size:12px; text-transform:uppercase;
    letter-spacing:.03em; color:var(--muted); margin-bottom:6px; }
  select, input[type=search] { font:inherit; font-size:15px; padding:8px 11px;
    border:1px solid #c3ced4; border-radius:6px; background:#fff; color:var(--ink); min-width:210px; }
  select:focus, input:focus { outline:2px solid #12303d; outline-offset:1px; }
  .swatch { display:inline-block; width:12px; height:12px; border-radius:3px;
    vertical-align:-1px; margin-right:7px; border:1px solid rgba(0,0,0,.18); }
  .party-name { font-weight:600; }
  a { color:#12556e; } a:hover { color:#12303d; }
  .src { font-size:12.5px; } .src code { word-break:break-all; }
  .note { background:#fdf6e6; border-left:3px solid #c08a00; padding:12px 15px;
    margin:14px 0 0; font-size:13.5px; }
  .note strong { color:#8a6100; }
  .mini { font-size:12.5px; color:var(--muted); }
  .bar-cell { position:relative; }
  .bar-cell i { position:absolute; left:0; top:50%; height:10px; margin-top:-5px;
    background:rgba(18,48,61,.13); border-radius:3px; display:block; }
  .bar-cell b { position:relative; font-weight:600; }
  #tip { position:fixed; pointer-events:none; z-index:99; opacity:0; transition:opacity .1s;
    background:#10222b; color:#fff; border-radius:8px; padding:11px 13px; font-size:12.5px;
    box-shadow:0 10px 26px rgba(0,0,0,.3); width:330px; max-width:92vw; }
  #tip .tt-name { font-weight:700; font-size:14px; margin-bottom:6px; }
  #tip .tt-grid { display:grid; grid-template-columns:auto auto; gap:2px 12px; }
  #tip .tt-grid span { color:#a9bfc9; }
  #tip .tt-grid b { text-align:right; font-variant-numeric:tabular-nums; }
  #tip table { width:100%; font-size:11.5px; margin:0; }
  #tip table th { background:none; color:#a9bfc9; font-weight:400; text-align:left;
    padding:1px 0; border:0; font-size:11px; }
  #tip table td { padding:1px 0; border:0; color:#fff; text-align:right;
    font-variant-numeric:tabular-nums; }
  #tip table tr.hl td { color:#5fe0a0; }
  #tip .tt-sec { margin-top:8px; padding-top:6px; border-top:1px solid rgba(255,255,255,.18);
    color:#a9bfc9; font-size:11px; text-transform:uppercase; letter-spacing:.03em; }
  #tip .tt-hist { margin-top:5px; color:#a9bfc9; font-size:11.5px; }
  #tip .tt-warn { margin-top:7px; padding:5px 7px; border-radius:5px; background:rgba(255,190,80,.16);
    color:#ffd48a; font-size:11.5px; font-weight:600; }
  #tip b.pos { color:#5fe0a0; } #tip b.neg { color:#ff9c92; }
  @media print { body { background:#fff; } .card { break-inside:avoid; border-color:#ccc; }
    #tip, .controls { display:none; } }
"""

# --------------------------------------------------------------------------
# Erklaerungstext
# --------------------------------------------------------------------------
def explanation_section(payload: dict, parties: Sequence[str],
                        naive_table_html: str, default_model_key: str) -> str:
    meta = payload["meta"]
    files = meta["source_files"]
    validation = meta["validation_ltw26"]
    per_party = validation["per_party"]
    postal = meta["postal_vote_accounting"]
    fits = payload["fits"]

    rows_validation = "".join(
        f"<tr><td>{html.escape(party)}</td>"
        f"<td class='num'>{de(per_party[party]['share_districts_only_pct'])}</td>"
        f"<td class='num'>{de(per_party[party]['share_incl_postal_pct'])}</td>"
        f"<td class='num'>{de(per_party[party]['official_share_incl_postal_pct'])}</td>"
        f"<td class='num'>{signed(per_party[party]['difference_pp'], 3)}</td></tr>"
        for party in parties
    )
    order = [k for k in ["LTW16", "BTW17", "BTW21", "LTW21", "BTW25", "LTW26"] if k in postal]
    rows_postal = "".join(
        f"<tr><td>{html.escape(meta['sources'][key]['label'])}</td>"
        f"<td class='num'>{de(postal[key]['share_of_voters_pct'], 1)}</td>"
        f"<td class='num'>{de_int(postal[key]['postal_votes'])}</td></tr>"
        for key in order
    )
    rows_sources = "".join(
        f"<tr><td>{html.escape(entry['label'])}<br><span class='mini'>{entry['date']}</span></td>"
        f"<td>{html.escape(entry['publisher'])}<br><span class='mini'>{entry.get('role', '')}</span></td>"
        f"<td class='src'><a href='{entry['url']}'>{html.escape(entry['url'])}</a></td>"
        f"<td class='src mini'><code>{entry['sha256'][:16]}…</code><br>{de_int(entry['bytes'])} Bytes</td></tr>"
        for entry in files.values()
    )
    rows_context = "".join(
        f"<tr><td>{html.escape(entry['label'])}<br><span class='mini'>{entry['date']}</span></td>"
        f"<td>{html.escape(entry['publisher'])}</td>"
        f"<td class='src'>{html.escape(entry['note'])}<br>"
        f"<a href='{entry['url']}'>{html.escape(entry['url'])}</a></td></tr>"
        for entry in meta["context_sources"].values()
    )
    dropped = []
    for model, party_map in fits.items():
        for party, fit in party_map.items():
            for item in fit["dropped"]:
                dropped.append(
                    f"<li><strong>{html.escape(party)}</strong> im Modell <code>{model}</code>: "
                    f"<code>{item['election']}</code> entfällt – {html.escape(item['reason'])}.</li>")
    rows_models = "".join(
        f"<tr><td><code>{name}</code></td><td>{html.escape(spec['title'])}</td>"
        f"<td>{html.escape(spec['description'])}</td>"
        f"<td class='num'>{len(spec['predictors'])}</td></tr>"
        for name, spec in meta["models"].items()
    )
    area_note = meta.get("context_info", {}).get("area_coverage", "–")

    return f"""
  <div class="card">
    <div class="card-head"><h2>So wird gerechnet — Schritt für Schritt</h2></div>

    <h3>1. Die Aufgabe</h3>
    <p>Wir wollen für jede Gemeinde in Mecklenburg-Vorpommern und jede Partei vorhersagen,
       welchen Anteil die Partei bei der <strong>Landtagswahl am 20.&nbsp;September 2026</strong>
       erzielen würde. Als Vorhersagewerkzeug dient eine <strong>lineare Regression</strong> — das
       einfachste statistische Verfahren, das einen Zahlenwert aus mehreren anderen Zahlenwerten
       berechnet. Sie beantwortet die Frage: „Wenn wir die bisherigen Wahlergebnisse und die
       soziale Struktur dieser Gemeinde kennen, welcher Anteil ergibt sich daraus für 2026?“</p>

    <h3>2. Welche Zahlen eingehen</h3>
    <p>Für jede Partei wird eine eigene Gleichung aufgestellt. Jeder <em>Wahlparameter</em> ist der
       <strong>Zweitstimmenanteil derselben Partei in derselben Gemeinde</strong> bei einer früheren
       Wahl. „Zweitstimme“ ist die Hauptstimme, mit der die Partei in der jeweiligen Gemeinde gewählt
       wird. Dazu kommen <em>sozialökonomische Parameter</em>, die die Struktur der Gemeinde
       beschreiben — sie gelten für alle Parteien gleichermassen.</p>
    <div class="formula">Anteil_{party}_LTW26  =  b<sub>0</sub>
      + b<sub>1</sub> · LTW16_{party}   + b<sub>2</sub> · BTW17_{party}
      + b<sub>3</sub> · BTW21_{party}   + b<sub>4</sub> · LTW21_{party}
      + b<sub>5</sub> · BTW25_{party}   + b<sub>6</sub> · LRW25_{party}
      + b<sub>7</sub> · log(Einwohner)  + b<sub>8</sub> · log(Fläche)
      + … (weitere Sozialindikatoren)</div>
    <p><strong>Wichtig:</strong> Die Gleichung wird nicht für jede Gemeinde einzeln gerechnet,
       sondern <em>einmal für die ganze Gruppe vergleichbarer Gemeinden</em> — für gut 700
       Gemeinden in MV, sobald alle Parameter vorliegen. Nur so lässt sich angeben, wie sicher ein
       Koeffizient ist. Die geschätzten Koeffizienten gelten danach für jede Gemeinde gleich — jede
       Gemeinde liefert ihre eigenen Ausgangswerte und erhält daraus ihre eigene Prognose.</p>

    <h3>3. Die fünf Modelle</h3>
    <table><tr><th>Modell</th><th>Bezeichnung</th><th>Was enthalten ist</th>
      <th class="num">Parameter</th></tr>{rows_models}</table>
    <p class="mini">Ein Parameter wird stillschweigend weggelassen, wenn für ihn weniger als 20
       Gemeinden einen Wert haben oder wenn er überhaupt keine Streuung zeigt (die Partei
       kandidierte dort nicht).</p>

    <h3>4. Warum es zwei Varianten mit Bevölkerung gibt — und warum sie dasselbe ergeben</h3>
    <p>Die <strong>Bevölkerungsdichte</strong> ist genau der Quotient aus Einwohnerzahl und Fläche,
       die Logarithmen also: log(Dichte) = log(Einwohner) − log(Fläche). Daraus folgt zweierlei.</p>
    <p><strong>Erstens</strong> kann man nicht alle drei Größen zugleich in dieselbe Gleichung
       aufnehmen — die dritte wäre vollständig durch die beiden anderen bestimmt, die Rechnung würde
       singular.</p>
    <p><strong>Zweitens</strong> liefern die Modelle <code>sozio</code> (mit log Einwohnerzahl und
       log Fläche) und <code>soziodichte</code> (mit log Dichte und log Fläche) <em>dieselbe
       Prognose und dasselbe R²</em>. Das überrascht, ist aber richtig: Die beiden Sätze von
       Parametern beschreiben denselben Raum, es ist nur eine Umbenennung. Der Effekt der Gemeindegrösse
       verteilt sich dabei anders auf die beiden Summanden. Am Beispiel Grünen, Gemeinde Boizenburg:
       im Modell <code>sozio</code> stehen −0,198 für log(Fläche) und +0,252 für log(Einwohner); im
       Modell <code>soziodichte</code> steht +0,252 für log(Dichte) und +0,054 für log(Fläche)
       (= 0,252 − (−0,198)). Die Summe ist dieselbe.</p>
    <p>Praktisch heisst das: Man kann wählen, <em>worauf</em> man den Grösseneffekt liest —
       entweder als „dicht besiedelte Orte wählen anders“ (Variante mit Dichte) oder als
       „grosse Orte wählen anders“ (Variante mit Einwohnerzahl). Die Vorhersagekraft ändert das
       nicht. Die Kennzahlen beider Modelle sind deshalb auf dieser Seite bewusst getrennt
       ausgewiesen.</p>

    <h3>5. Die Landratswahl 2025 als Parameter</h3>
    <p>Die Landratswahl vom 11.&nbsp;Mai 2025 ist eine <em>Personenwahl</em>: Gewählt wird nicht eine
       Partei, sondern ein Landrat oder eine Landrätin. Als Parameter dient deshalb der Stimmenanteil
       der Kandidatur <em>derselben Partei</em> in derselben Gemeinde. 2025 fanden nur in vier
       Landkreisen Landratswahlen statt, und allein der Landkreis Ludwigslust-Parchim veröffentlicht
       die Ergebnisse je Gemeinde und Wahlbezirk — dort stehen die 140 Gemeinden zur Verfügung, die
       die Modelle <code>wahl6lrw</code> und <code>soziolrw</code> tragen.</p>

    <div class="note"><strong>Bitte beachten:</strong> Diese Seite ist ein <em>Rückblick</em>, keine
       Wahlprognose im eigentlichen Sinn. Die Landtagswahl 2026 hat am 20.&nbsp;September 2026
       stattgefunden und ist ausgezählt. Die Gleichungen wurden ausschließlich aus Wahlen und
       Sozialdaten <em>vor</em> diesem Datum berechnet; anschließend wurde die Prognose mit dem
       amtlichen Ergebnis verglichen. Genau dieser Vergleich erlaubt es, die Treffergenauigkeit
       seriös zu messen.</div>

    <h3>6. Was bedeuten Koeffizient, Konstante und p-Wert?</h3>
    <p><strong>Der Koeffizient</strong> sagt, wie viele Prozentpunkte der Anteil 2026 zusätzlich steigt,
       wenn derselbe Anteil 100 Jahre früher um einen Prozentpunkt höher gewesen wäre. Beispiel SPD:
       der Koeffizient für die Landtagswahl 2021 liegt bei rund 0,39. Ein Punkt höherer SPD-Anteil 2021
       bedeutet danach im Mittel etwa 0,39 Prozentpunkte mehr SPD 2026.</p>
    <p><strong>Die Konstante</strong> („Achsenabschnitt“) ist der Anteil, den das Modell für eine
       Gemeinde vorhersagt, in der alle Vorgängerwerte null wären. Sie ist bei einer linearen
       Regression weit weniger bedeutsam als die Koeffizienten und ändert sich stark, wenn sich der
       Wertebereich der Parameter verschiebt.</p>
    <p><strong>Der p-Wert</strong> beantwortet die Frage: „Wäre ein so starker Zusammenhang auch
       zufällig entstanden?“ Vereinfacht: <em>kleiner p-Wert = die Verbindung ist sehr wahrscheinlich
       echt; großer p-Wert = der Zusammenhang könnte Zufall sein.</em> Üblich sind die Schranken
       5&nbsp;%, 1&nbsp;% und 0,1&nbsp;%. Ein p-Wert von 0,42 heißt: Diese Zahl liefert keinen
       belastbaren Hinweis — der Koeffizient ist statistisch nicht von null zu unterscheiden.</p>

    <div class="note"><strong>Warum sind die p-Werte nicht überall klein?</strong>
       Die Parameter sind stark miteinander korreliert: Eine Gemeinde, die 2021 viele CDU-Stimmen
       hatte, hatte dort meist auch 2025 viele. Weil die Datenkolonnen redundant sind, kann die
       Rechnung den einzelnen Effekt schlechter trennen. Einzelne Koeffizienten sind daher deutlich
       unsicherer, als die Gesamtschätzung vermuten lässt. Entscheidend ist das Modell als Ganzes —
       dafür stehen die R²-Werte oben.</div>

    <h3>7. Was ist R², und wie liest man diese Seite?</h3>
    <p>Stellen Sie sich ein Punktdiagramm vor: auf der waagerechten Achse den Anteil, den das Modell
       vorausgesagt hat, auf der senkrechten das echte Ergebnis. Bei einem perfekten Modell lägen alle
       Punkte auf einer geraden Linie.</p>
    <p><strong>R² misst, wie viel von der Streuung zwischen den Gemeinden das Modell erklärt.</strong>
       R²&nbsp;=&nbsp;0,70 bedeutet: 70&nbsp;Prozent der Unterschiede zwischen den Gemeinden werden vom
       Modell nachgebildet, 30&nbsp;Prozent bleiben unerklärt. R² ist kein Anteil von Stimmen und
       keine Trefferquote — es beschreibt ausschließlich, wie gut Abstände auf der Karte nachgebildet
       werden. Ein Modell mit hohem R² kann trotzdem bei jeder einzelnen Gemeinde danebenliegen, und
       ein Modell mit niedrigem R² kann im Mittel richtig liegen.</p>
    <p>Weil man ein Modell auch mit Daten füttern kann, an denen es dann gut aussieht, ohne dass es
       wirklich generalisiert, wird zusätzlich <strong>R² „out-of-fold“</strong> ausgewiesen: Die
       Gemeinden werden in zehn Blöcke geteilt, das Modell wird neunmal mit neun Blöcken trainiert und
       am zehnten Block geprüft — zehnmal, bis jede Gemeinde einmal die unbekannte Prüfung war. Diese
       Zahl ist die ehrlichere; sie liegt erwartungsgemäß knapp unter dem ersten Wert.</p>
    <p><strong>Und noch etwas:</strong> Die Landtagswahlen und Bundestagswahlen finden nicht am
       selben Tag statt. Manche Gemeinden wählen stärker per Brief, andere nicht; die Landtagswahl
       2021 wurde sogar im September abgehalten. Ein hohes R² bedeutet deshalb ausdrücklich
       <em>nicht</em>, dass sich Stimmen ausgezählt werden könnten.</p>
    <div class="note"><strong>Grenzen des Verfahrens.</strong> Eine lineare Regression kennt keine
       Übertreibung nach oben. Realität: Ein Anteil kann nicht über 100&nbsp;Prozent liegen und nicht
       unter 0&nbsp;Prozent — negative Rechnungen werden deshalb auf 0 begrenzt. Das Modell erklärt
       nur die <em>Orts variation</em>. Es kann nicht vorhersagen, was Wahlkampf, eine Bundestagswahl
       vier Monate vorher, eine Bundestagsregierung oder ein Unwetter im Wahlkampfmonat bewirken —
       genau solche <em>Sprünge zwischen den Wahlen</em> zeigen sich hier in den Karten. Gerade für
       die AfD fällt das ins Gewicht: Ihr Ergebnis 2026 liegt vielerorts deutlich über dem, was die
       Vorgängerwerte erwarten ließen.</div>

    <h3>8. Warum fehlen die Briefwahlergebnisse?</h3>
    <p>Die Landeswahlleiterin veröffentlicht die Briefwahlergebnisse auf <em>Amtsebene</em>, nicht je
       Gemeinde. Sie lassen sich daher keiner einzelnen Gemeinde zuordnen. Weil eine Briefwählerin
       andere Parteien wählt als jemand, der ins Wahllokal geht, wäre eine Schätzung der Zuordnung
       keine Faktenlage, sondern eine Annahme. Wir lassen sie deshalb <em>weg</em> — in den
       Vergleichswerten <em>und</em> im Ergebnis. Damit ist die Grundlage für alle Wahlen identisch;
       die folgende Kontrollrechnung belegt die Größenordnung des Unterschieds.</p>

    <h3>9. Kontrollrechnung gegen das amtliche Landesergebnis</h3>
    <p>Wenn die Aufbereitung stimmt, muss die Summe aller Gemeinden zuzüglich der Briefwahl das
       veröffentlichte Landesergebnis ergeben. Für die LTW26 ergibt sich:</p>
    <table><tr><th>Partei</th>
      <th class="num">nur Wahlbezirke (Modellbasis)</th>
      <th class="num">+ Briefwahl (nachgerechnet)</th>
      <th class="num">amtlich, ganz MV</th><th class="num">Abweichung</th></tr>{rows_validation}</table>
    <p class="map-note">Die Abweichungen liegen bei höchstens 0,01&nbsp;Prozentpunkten und stammen vom
       Runden. Die Datenaufbereitung ist damit nachvollziehbar geprüft.</p>

    <h3>10. Wie stark Briefwahl das Ergebnis prägt</h3>
    <table><tr><th>Wahl</th><th class="num">Briefwahlanteil aller Wähler</th>
      <th class="num">Briefwahlstimmen</th></tr>{rows_postal}</table>
    <p class="map-note">Je nach Wahl lag zwischen 10,6&nbsp;% (Bundestagswahl 2017) und
       16,2&nbsp;% (Bundestagswahl 2021) der Wählerinnen und Wähler an der Briefwahl. Ohne
       Briefwahl liegt der Anteil der Grünen 2026 landesweit bei 5,65&nbsp;% statt 5,67&nbsp;%, der
       Anteil der AfD bei 39,68&nbsp;% statt 38,22&nbsp;% — die Briefwählerinnen und Briefwähler
       neigten also 2026 erkennbar zur AfD.</p>

    <h3>11. Sozialökonomische Daten</h3>
    <p>Einwohnerzahl, Haushaltsgrößen, Wohnungs- und Mietmerkmale stammen aus dem
       <strong>Zensus 2022</strong> (Stichtag 15.&nbsp;Mai 2022) der Statistischen Ämter des Bundes
       und der Länder. Die amtlichen Ergebnisdateien enthalten keine Gemeindeflächen, und
       amtliche Flächenangaben je Gemeinde liegen nicht als maschinenlesbare Datei vor. Deshalb werden
       die Flächen aus den <strong>OpenStreetMap-Grenzringen</strong> berechnet (Berechnung
       siehe unten). Abdeckung: {area_note}.</p>
    <table>{rows_context}</table>

    <h3>12. Welche Parameter sind weggefallen, und warum?</h3>
    <ul>{"".join(dropped)}</ul>
    <p>Bei der Landratswahl 2025 standen im Landkreis Ludwigslust-Parchim nur vier Bewerberinnen und
       Bewerber zur Wahl (CDU, AfD, SPD, GRÜNE) — für FDP, DIE LINKE und BSW existiert dort kein
       Parameter. Die BSW trat 2025 erstmals zu einer Bundestagswahl in MV an und war 2016, 2017 und
       2021 nicht im Wahlgebiet; für sie bleiben deshalb nur die späteren Werte.</p>

    <h3>13. Genauigkeit je Partei und Modell</h3>
    <table id="acc-table"><thead><tr>
      <th>Modell</th><th>Partei</th><th class="num">Gemeinden</th>
      <th class="num">R²</th><th class="num">R² out-of-fold</th><th class="num">RMSE (pp)</th>
      <th class="num">Ø |Fehler| (pp)</th><th class="num">≤ 0,5 pp</th><th class="num">≤ 1 pp</th>
      <th class="num">≤ 2 pp</th><th class="num">Pearson r</th></tr></thead></table>
    <p class="map-note">Spalten von links nach rechts: <strong>Gemeinden</strong> — wie viele
      Gemeinden in die Schätzung eingingen. <strong>R²</strong> — Anteil der Streuung zwischen den
      Gemeinden, den das Modell erklärt. <strong>R² out-of-fold</strong> — dasselbe, aber gemessen an
      Gemeinden, die nie in die Schätzung eingingen; das ist die ehrlichere Zahl.
      <strong>RMSE (pp)</strong> — mittlerer quadratischer Fehler in Prozentpunkten, gewichtet große
      Ausreißer stärker. <strong>Ø |Fehler| (pp)</strong> — durchschnittlicher Abstand in
      Prozentpunkten. <strong>≤ 0,5 pp / ≤ 1 pp / ≤ 2 pp</strong> — Anteil der Gemeinden, in denen
      die Prognose höchstens 0,5, 1 bzw. 2 Prozentpunkte danebenlag. <strong>Pearson r</strong> —
      Korrelation zwischen Prognose und Ergebnis (1 = perfekt). Die hervorgehobene Zeile gehört zur
      oben gewählten Partei und zum gewählten Modell.</p>

    <h3>14. Alle Gemeinden im Überblick</h3>
    <p>724 Zeilen, eine je Gemeinde. Anklickbare Spaltenüberschriften sortieren die Tabelle.</p>
    <div class="controls" style="margin-bottom:12px">
      <div><label for="q">Gemeinde suchen</label>
        <input type="search" id="q" placeholder="z. B. Boizenburg"></div>
      <div><label for="mq">Gemeinden</label><select id="mq"></select></div>
      <div><label for="mrows">Zeilen</label><select id="mrows">
        <option>25</option><option>50</option><option>100</option>
        <option>250</option><option value="9999">alle</option></select></div>
      <div><label for="mmetric">Sortieren nach</label><select id="mmetric">
        <option value="res">Ist − Prognose</option>
        <option value="pred">Prognose</option>
        <option value="actual">Ist-Ergebnis</option>
        <option value="dichte">Bevölkerungsdichte</option>
        <option value="einwohner">Einwohner</option>
        <option value="flaeche">Fläche</option>
        <option value="name">Name</option></select></div>
    </div>
    <table id="mtable"><thead><tr>
      <th data-sort="name">Gemeinde</th><th data-sort="landkreis">Landkreis</th>
      <th class="num" data-sort="einwohner">Einwohner</th>
      <th class="num" data-sort="flaeche">Fläche km²</th>
      <th class="num" data-sort="dichte">Einw./km²</th>
      <th class="num" data-sort="actual">Ist LTW26</th>
      <th class="num" data-sort="pred">Prognose</th>
      <th class="num" data-sort="res">Ist − Prognose</th>
      <th class="num" data-sort="ltw21">LTW 2021</th>
      <th class="num" data-sort="btw25">BTW 2025</th>
      <th class="num" data-sort="turnout">Wahlbeteiligung</th>
      <th class="num" data-sort="valid">gültige Stimmen</th>
    </tr></thead><tbody></tbody></table>

    <h3>15. Genauigkeit im Vergleich</h3>
    <p>Wie gut wäre es einfach gewesen, den Wert von 2025 ungewändert zu übernehmen? Weil
       Landesdurchschnitte dicht beieinanderliegen, kommt man damit erstaunlich weit. Erst der
       Vergleich macht das R² der Regression aussagekräftig. Gezeigt ist das Modell
       <code>{default_model_key}</code>.</p>
    <table id="naive-table">{naive_table_html}</table>

    <h3>16. Quellen der Wahlergebnisse</h3>
    <table>{rows_sources}</table>

    <h3>17. Weiterführende Literatur und Dokumentation</h3>
    <ul>
      <li>Landeswahlleiter Mecklenburg-Vorpommern, amtliche Wahlergebnisse und Bekanntmachungen:
        <a href="https://www.laiv-mv.de/Wahlen/">laiv-mv.de/Wahlen</a> ·
        <a href="https://wahlen.mvnet.de/">wahlen.mvnet.de</a></li>
      <li>Bundeswahlleiterin, Bundestagswahlergebnisse 2025:
        <a href="https://www.bundeswahlleiterin.de/bundestagswahlen/2025/ergebnisse/">bundeswahlleiterin.de</a></li>
      <li>Zensus 2022, Regionaltabellen:
        <a href="https://www.zensus2022.de/">zensus2022.de</a></li>
      <li>Kreiswahlleiter Ludwigslust-Parchim, Landratswahl 2025:
        <a href="https://www.kreis-lup.de/Politik/Wahlen/Landratswahl-2025/">kreis-lup.de</a></li>
      <li>Wikidata (CC0 1.0), Gemeindeflächen und amtlicher Gemeindeschlüssel:
        <a href="https://www.wikidata.org/">wikidata.org</a></li>
      <li>Hypothese zum Bestimmtheitsmaß (R²), Definition und Konventionen:
        <a href="https://de.wikipedia.org/wiki/Bestimmtheitsma%C3%9F">de.wikipedia.org/wiki/Bestimmtheitsmaß</a></li>
      <li>Grundlagen der linearen Regression (Schätzung, Koeffizienten, Signifikanztests):
        <a href="https://de.wikipedia.org/wiki/Lineare_Regression">de.wikipedia.org/wiki/Lineare_Regression</a></li>
      <li>Implementierung der OLS-Schätzung und Definition der p-Werte:
        <a href="https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html">statsmodels.org</a></li>
      <li>Definition des R²-Scores und der Kreuzvalidierung:
        <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score">scikit-learn.org</a></li>
    </ul>
    <p class="map-note">Die Landtagswahl MV 2026 wurde nach den hier ausgewerteten Daten ausgezählt;
       alle Ist-Werte sind amtlich. Fehler in dieser Aufbereitung sind möglich; das Skript
       <code>predict.py</code> ist der Nachweis: Jede Zahl lässt sich neu berechnen. Die Karten sind
       als SVG direkt in diese Datei eingebettet, sodass sie ohne weitere Dateien auskommt.</p>
  </div>
"""


def r2_of(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def naive_table(payload: dict, parties: Sequence[str], model: str) -> str:
    rows = []
    for party in parties:
        actual, naive, predicted = [], [], []
        for m in payload["municipalities"].values():
            a = m["actual"][party]
            b = m["features"][party]["BTW25"]
            p = m["predicted"][model][party]
            if a is None or b is None or p is None:
                continue
            actual.append(a); naive.append(b); predicted.append(p)
        if len(actual) < 10:
            continue
        a = np.asarray(actual, dtype=float)
        b = np.asarray(naive, dtype=float)
        p = np.asarray(predicted, dtype=float)
        rows.append((party, len(actual), float(r2_of(a, b)), float(r2_of(a, p))))
    parts = ["<tr><th>Partei</th><th class='num'>Gemeinden</th>"
             "<th class='num'>R²: BTW&nbsp;2025 einfach übernehmen</th>"
             "<th class='num'>R²: lineare Regression</th><th class='num'>Gewinn</th></tr>"]
    for party, n, r2_naive, r2_pred in rows:
        parts.append(
            f"<tr><td>{html.escape(party)}</td><td class='num'>{n}</td>"
            f"<td class='num'>{de(r2_naive, 3)}</td><td class='num'>{de(r2_pred, 3)}</td>"
            f"<td class='num'>{signed(r2_pred - r2_naive, 3)}</td></tr>")
    return "".join(parts)


PAGE = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LTW26 - Was sagt eine lineare Regression voraus?</title>
<style>{css}</style>
</head>
<body>
<header><div class="wrap">
  <h1>LTW26: Was sagt eine lineare Regression voraus?</h1>
  <p>Eine Prognose je Gemeinde und Partei aus den Ergebnissen früherer Wahlen &mdash; je nach
     gewähltem Modell ergänzt um die Landratswahl 2025 und die soziale Struktur der Gemeinde
     &mdash; verglichen mit dem amtlichen Ergebnis vom 20.&nbsp;September 2026.
     Fahren Sie mit der Maus über eine Gemeinde, um Prognose, Ergebnis, alle Parteien und die
     Ausgangswerte zu sehen.</p>
</div></header>

<main>

  <div class="card">
    <div class="card-head">
      <h2>Modell und Steuerung</h2>
      <p>Wählen Sie Partei, Modell und Karteninhalt. Karten, Farbskala, Kennzahlen, Rechenbeispiel,
         Koeffizienten und Tabellen passen sich sofort an.</p>
    </div>
    <div class="controls">
      <div><label for="party">Partei</label><select id="party">{party_options}</select></div>
      <div><label for="model">Modell</label><select id="model">{model_options}</select></div>
      <div><label for="metric">Karteninhalt</label><select id="metric">{metric_options}</select></div>
      <div style="flex:1 1 260px"><span class="label-spacer" aria-hidden="true">&nbsp;</span>
        <div class="mini" id="party-line"></div></div>
    </div>
    <div class="legend">
      <div class="legend-row">
        <span id="bar-lo">–</span><div class="bar" id="bar"></div><span id="bar-hi">–</span>
        <span id="bar-note"></span>
      </div>
      <div class="legend-votes" id="legend-votes"></div>
    </div>
    <div id="kpi"></div>
    <p class="map-note" id="kpi-note"></p>
  </div>

  <div id="maps">
{map_blocks}
  </div>

  <div class="card">
    <div class="card-head">
      <h2>Rechenbeispiel: die Prognose Schritt für Schritt</h2>
      <p>Hier wird die Rechnung für eine einzelne Gemeinde vorgeführt. Wählen Sie die Gemeinde —
         das Modell oben bestimmt Partei und Parameter.</p>
    </div>
    <div class="controls">
      <div><label for="municipality">Gemeinde für das Rechenbeispiel</label>
        <select id="municipality"></select></div>
      <div style="flex:1 1 260px"><span class="label-spacer" aria-hidden="true">&nbsp;</span>
        <div class="mini" id="example-status"></div></div>
    </div>
    <p id="example-intro"></p>
    <div id="example"></div>
  </div>

  <div class="card">
    <div class="card-head">
      <h2>Achsenabschnitt, Koeffizienten und p-Werte</h2>
      <p id="coef-intro"></p>
    </div>
    <div id="coef"></div>
    <p class="map-note">Signifikanz: *** p&nbsp;&lt;&nbsp;0,001 &middot; ** p&nbsp;&lt;&nbsp;0,01
      &middot; * p&nbsp;&lt;&nbsp;0,05 &middot; „n.&nbsp;s.“ = nicht signifikant. Der p-Wert gibt an,
      wie unwahrscheinlich ein so starker Zusammenhang allein zufällig zustande käme.</p>
  </div>

{explanation}
</main>

<footer>
<script>
  (function () {{
    // Ensure this only runs in a browser environment.
    if (typeof window === 'undefined' || typeof document === 'undefined') {{
      return;
    }}

    var siteId = '87faa452-3644-4756-ab38-50b55b72c272';
    var trackingDomain = 'www.veritametrics.com';

    // 1. Initialize the async command queue, so that window.verita() is
    //    available immediately.
    window.verita =
      window.verita ||
      function () {{
        (window.verita.q = window.verita.q || []).push(arguments);
      }};
    window.verita.l = new Date().getTime();

    // 2. Load the enhanced tracker script asynchronously. It will process
    //    the command queue once loaded.
    var script = document.createElement('script');
    script.src = 'https://' + trackingDomain + '/api/tracker.js?siteId=' + siteId;
    script.async = true;
    script.defer = true;
    script.dataset.siteId = siteId;
    script.dataset.trackingDomain = trackingDomain;
    document.head.appendChild(script);

    // 3. Send the initial pageview using the *queue*.
    window.verita('trackPageView');
  }})();
</script>
<noscript>
  <img src="https://www.veritametrics.com/api/track-pageview?siteId=87faa452-3644-4756-ab38-50b55b72c272"
       referrerpolicy="unsafe-url" alt="" width="1" height="1"
       style="display:none" aria-hidden="true">
</noscript>
</footer>

<div id="tip" role="tooltip" aria-live="polite"></div>
<script>
var DATA = {data};
var FITS = {fits};
var ACC = {acc};
var METRIC = {metric_meta};
var GEO = {geo};
var GEOM = {geom};
var OUTLINES = {outlines};
var FILLS = {fills};
var FILLPTR = {fillptr};
var SCALES = {scales};
var LABELS = {labels};
var MODELS = {models_list};
var PARTIES = {parties_list};
var DEFAULT_MODEL = {default_model};
var DEFAULT_PARTY = {default_party};
var DEFAULT_METRIC = {default_metric};
var COUNTY_KEYS = {counties_list};
var MAP_NOTE_BASE = {map_note_base};
</script>
<script>
(function () {{
  var $ = function (id) {{ return document.getElementById(id); }};
  var NBSP = '\u00a0';   // fuer textContent: Entities werden dort nicht ausgewertet
  var party = DEFAULT_PARTY, model = DEFAULT_MODEL, metric = DEFAULT_METRIC, example = null;
  Object.keys(DATA).forEach(function (c) {{ if (DATA[c].n === 'Boizenburg/Elbe') example = c; }});
  if (!example || !DATA[example]) example = Object.keys(DATA)[0];
  var PREDICTOR = LABELS.predictors, CTX = LABELS.context;

  function de(v, d) {{
    if (v === null || v === undefined) return '–';
    return Number(v).toFixed(d === undefined ? 2 : d).replace('.', ',');
  }}
  function pct(v) {{ return v === null || v === undefined ? '–' : de(v, 1) + '&nbsp;%'; }}
  function di(v) {{ return v === null || v === undefined ? '–' : Math.round(v).toLocaleString('de-DE'); }}
  function signed(v, d) {{
    if (v === null || v === undefined) return '–';
    d = d === undefined ? 2 : d;
    var a = Math.abs(Number(v)).toFixed(d).replace('.', ',');
    if (v > 1e-9) return '+' + a;
    if (v < -1e-9) return '−' + a;
    return (0).toFixed(d).replace('.', ',');
  }}
  function stars(p) {{
    if (p === null || p === undefined) return '';
    if (p < 0.001) return '***'; if (p < 0.01) return '**'; if (p < 0.05) return '*';
    return 'n.&nbsp;s.';
  }}
  function sigText(p) {{
    if (p === null || p === undefined) return '';
    if (p < 0.001) return 'hoch signifikant'; if (p < 0.01) return 'signifikant';
    if (p < 0.05) return 'schwach signifikant';
    return 'nicht signifikant';
  }}

  // ---------- Karte einfärben ----------
  function paint() {{
    document.querySelectorAll('[data-layer]').forEach(function (section) {{
      var region = section.getAttribute('data-layer');
      var idx = FILLPTR[model][metric][party][region];
      var palette = FILLS[idx];
      var svg = section.querySelector('svg.map-svg');
      if (!svg) return;
      svg.querySelectorAll('path.muni').forEach(function (p) {{
        var c = p.getAttribute('data-code');
        p.setAttribute('fill', palette[c] || '#e9edf0');
      }});
      var note = section.querySelector('[data-mapnote]');
      if (note) note.innerHTML = MAP_NOTES[region];
    }});
    var sc = SCALES[model][metric][party]['mv'];
    $('bar').style.background = METRIC[metric].gradient;
    // Achtung: textContent interpretiert keine HTML-Entities, deshalb hier
    // echte Zeichen und keine geschriebenen Entities wie &nbsp;.
    $('bar-lo').textContent = metric === 'residual'
      ? '−' + de(-sc.lo, 1) + ' pp' : de(sc.lo, 0) + NBSP + '%';
    $('bar-hi').textContent = metric === 'residual'
      ? '+' + de(sc.hi, 1) + ' pp' : de(sc.hi, 0) + NBSP + '%';
    $('bar-note').textContent = METRIC[metric].note;
  }}
  var MAP_NOTES = {{mv: '', lup: ''}};


  // ---------- Tooltip ----------
  function tipFor(code) {{
    var m = DATA[code];
    var out = '<div class="tt-name">' + m.n + '</div>';
    out += '<div class="tt-sec">' + party + ' &mdash; ' + METRIC[metric].label + '</div>';
    var pred = m.p[model][party], act = m.a[party], res = m.r[model][party];
    out += '<div class="tt-grid">';
    out += '<span>Prognose</span><b>' + (pred === null ? '–' : de(pred) + '&nbsp;%') + '</b>';
    out += '<span>Ist-Ergebnis</span><b>' + (act === null ? '–' : de(act) + '&nbsp;%') + '</b>';
    out += '<span>Ist &minus; Prognose</span><b class="' + (res > 0 ? 'pos' : (res < 0 ? 'neg' : ''))
         + '">' + signed(res) + '&nbsp;pp</b>';
    Object.keys(PREDICTOR).forEach(function (k) {{
      var v = m.f[party] ? m.f[party][k] : null;
      out += '<span>' + PREDICTOR[k] + '</span><b>' + (v === null ? '–' : de(v, 2)) + '</b>';
    }});
    out += '</div>';

    out += '<div class="tt-sec">Alle Parteien &mdash; Ist LTW26 / Prognose ' + model + '</div>';
    out += '<table><tr><th>Partei</th><th>Ist</th><th>Prognose</th><th>Diff.</th></tr>';
    PARTIES.forEach(function (p) {{
      var a = m.a[p], pr = m.p[model][p], rr = m.r[model][p];
      out += '<tr' + (p === party ? ' class="hl"' : '') + '>'
        + '<td>' + p + '</td>'
        + '<td>' + (a === null ? '–' : de(a)) + '</td>'
        + '<td>' + (pr === null ? '–' : de(pr)) + '</td>'
        + '<td>' + signed(rr) + '</td></tr>';
    }});
    out += '</table>';

    out += '<div class="tt-sec">Sozialer Kontext</div><div class="tt-hist">';
    out += di(m.c.einwohner) + ' Einwohner &middot; '
      + (m.c.flaeche_km2 === null ? 'Fläche unbekannt' : de(m.c.flaeche_km2, 1) + '&nbsp;km²') + ' &middot; '
      + (m.c.dichte === null ? 'Dichte unbekannt' : de(m.c.dichte, 1) + ' Einw./km²') + '<br>'
      + de(m.c.ueber65, 1) + '&nbsp;% 65 Jahre und älter &middot; '
      + de(m.c.auslaender, 1) + '&nbsp;% ausländische Staatsangehörigkeit<br>'
      + de(m.c.miete_pro_qm, 2) + '&nbsp;€/m² Miete &middot; '
      + de(m.c.eigentum, 0) + '&nbsp;% Eigentümerquote &middot; '
      + 'Ø Haushalt ' + de(m.c.haushalt_groesse, 2) + '<br>'
      + di(m.v) + ' gültige Zweitstimmen (ohne Briefwahl) &middot; Wahlbeteiligung '
      + de(m.t, 1) + '&nbsp;%';
    out += '</div>';
    if (pred === null) {{
      out += '<div class="tt-warn">Für diese Gemeinde liegen im Modell ' + model
           + ' nicht alle Parameter vor.</div>';
    }}
    return out;
  }}

  // ---------- Kennzahlen ----------
  function accRow() {{
    var row = null;
    ACC.forEach(function (r) {{ if (r.model === model && r.party === party) row = r; }});
    return row;
  }}
  function renderKpi() {{
    var row = accRow(), fit = FITS[model][party];
    function cell(label, value) {{ return '<div><span>' + label + '</span><b>' + value + '</b></div>'; }}
    var cells = '';
    if (row) {{
      cells += cell('R² (Modell)', de(row.r2, 3));
      cells += cell('R² out-of-fold', de(row.oof, 3));
      cells += cell('Ø Fehler (RMSE)', de(row.rmse) + '&nbsp;pp');
      cells += cell('Ø |Fehler|', de(row.mae) + '&nbsp;pp');
      cells += cell('Innerhalb ±1 pp', pct(row.w1));
      cells += cell('Innerhalb ±2 pp', pct(row.w2));
    }}
    cells += cell('Gemeinden', di(row ? row.n : 0));
    cells += cell('Parameter', fit ? fit.predictors.length : 0);
    $('kpi').innerHTML = '<div class="kpi">' + cells + '</div>';
    var note = '';
    if (fit && fit.dropped && fit.dropped.length) {{
      note = 'Nicht verwendet: ' + fit.dropped.map(function (d) {{
        return '<code>' + d.election + '</code> (' + d.reason + ')'; }}).join(', ') + '. ';
    }}
    if (fit && fit.same) {{
      note += 'Für ' + party + ' ist dieses Modell rechnerisch identisch mit <code>wahl6</code>, '
            + 'weil kein zusätzlicher Parameter vorliegt. ';
    }}
    if (row && row.rmse !== null) {{
      note += 'Ein durchschnittlicher Fehler von ' + de(row.rmse) + ' Prozentpunkten heißt: In einer '
            + 'typischen Gemeinde verfehlt die Prognose das Ergebnis dieser Partei um gut '
            + de(row.rmse) + ' Prozentpunkte.';
    }}
    $('kpi-note').innerHTML = note;
    $('party-line').innerHTML = party + ' &middot; Modell <code>' + model + '</code>'
        + (fit ? ' &middot; n = ' + di(fit.n) : '');
  }}

  // ---------- Rechenbeispiel ----------
  function renderExample() {{
    var sel = $('municipality');
    if (!example || !DATA[example]) example = Object.keys(DATA)[0];
    var m = DATA[example], fit = FITS[model][party];
    $('example-intro').innerHTML = 'So entsteht die Prognose für <strong>' + m.n + '</strong> im '
        + 'Modell <code>' + model + '</code>. Jeder Summand ist ein Ausgangswert dieser Gemeinde mal '
        + 'dem zugehörigen Koeffizienten aus der Tabelle darunter.';
    if (!fit) {{
      $('example').innerHTML = '<p class="mini">Für diese Partei liegt im gewählten Modell keine '
          + 'Schätzung vor.</p>'; return;
    }}
    var rows = '<tr><th>Rechnungsschritt</th><th class="num">Wert</th>'
      + '<th class="num">× Koeffizient</th><th class="num">= Beitrag</th></tr>';
    var total = fit.intercept.estimate;
    rows += '<tr><td>Konstante (Achsenabschnitt)</td><td class="num">–</td>'
      + '<td class="num">' + de(fit.intercept.estimate, 4) + '</td>'
      + '<td class="num">' + de(fit.intercept.estimate, 4) + '</td></tr>';
    fit.predictors.forEach(function (key) {{
      var socio = LABELS.socio.indexOf(key) >= 0;
      var value = m.f[party][key], c = fit.params[key].estimate;
      var contrib = (value === null || value === undefined) ? null : value * c;
      if (contrib !== null) total += contrib;
      var digits = socio ? 4 : 2;
      var label = (socio ? (key.replace('ln_', 'log ') + ' (Soziographie)')
                         : (PREDICTOR[key] + ' ' + party));
      rows += '<tr><td>' + label + (value === null ? ' <span class="mini">(fehlt)</span>' : '')
        + '</td><td class="num">' + de(value, digits) + '</td>'
        + '<td class="num">' + de(c, 4) + '</td>'
        + '<td class="num">' + (contrib === null ? '–' : de(contrib, 4)) + '</td></tr>';
    }});
    rows += '<tr><th>= Prognose LTW26</th><th class="num"></th><th class="num"></th>'
      + '<th class="num">' + de(total, 3) + '</th></tr>';
    var pred = m.p[model][party];
    rows += '<tr><td>Prognose, begrenzt auf 0 bis 100 %</td><td class="num" colspan="3"><b>'
      + (pred === null ? '–' : de(pred) + '&nbsp;%') + '</b></td></tr>';
    rows += '<tr><td>Amtliches Ist-Ergebnis LTW26</td><td class="num" colspan="3"><b>'
      + (m.a[party] === null ? '–' : de(m.a[party]) + '&nbsp;%') + '</b></td></tr>';
    if (m.r[model][party] !== null) {{
      rows += '<tr><td>Differenz Ist minus Prognose</td><td class="num" colspan="3"><b class="'
        + (m.r[model][party] > 0 ? 'pos' : 'neg') + '">' + signed(m.r[model][party])
        + '&nbsp;pp</b></td></tr>';
    }}
    var c = m.c;
    var avail = Object.keys(DATA).filter(function (k) {{ return DATA[k].p[model][party] !== null; }});
    var hasLrw = FITS[model][party].predictors.indexOf('LRW25') >= 0;
    $('example-status').innerHTML = 'Für die gewählte Partei und das Modell <code>' + model
      + '</code> ist die Prognose in <strong>' + di(avail.length) + '</strong> von '
      + di(Object.keys(DATA).length) + ' Gemeinden berechenbar'
      + (avail.length < Object.keys(DATA).length
         ? (hasLrw
            ? ' — außerhalb des Landkreises Ludwigslust-Parchim fehlt die Landratswahl 2025.'
            : ' — für die übrigen Gemeinden fehlen einzelne Parameter (siehe Hinweis oben).')
         : '.');
    $('example').innerHTML = '<table>' + rows + '</table>'
      + '<p class="mini">Gemeindeschlüssel ' + example + ' &middot; ' + (m.lk || '')
      + (m.wk ? ' &middot; Bundestagswahlkreis 2025: ' + m.wk : '') + ' &middot; ' + di(c.einwohner) + ' Einwohner'
      + (c.flaeche_km2 === null ? '' : ' &middot; ' + de(c.flaeche_km2, 1) + ' km²')
      + (c.dichte === null ? '' : ' &middot; ' + de(c.dichte, 1) + ' Einwohner je km²')
      + ' &middot; ' + di(m.v) + ' gültige Zweitstimmen (ohne Briefwahl) &middot; Wahlbeteiligung '
      + de(m.t, 1) + '&nbsp;%</p>';
  }}

  // ---------- Koeffizienten ----------
  function renderCoef() {{
    var fit = FITS[model][party];
    $('coef-intro').innerHTML = 'Regression für <strong>' + party + '</strong> im Modell <code>'
      + model + '</code>, geschätzt über ' + di(fit ? fit.n : 0)
      + ' Gemeinden. Die Spalte „p“ ist der p-Wert: Je kleiner, desto sicherer ist der Zusammenhang.';
    if (!fit) {{ $('coef').innerHTML = '<p class="mini">keine Schätzung</p>'; return; }}
    var out = '<table><tr><th>Parameter</th><th class="num">Koeffizient</th>'
      + '<th class="num">Standardfehler</th><th class="num">p-Wert</th><th>Signifikanz</th>'
      + '<th class="num">95-%-Intervall</th></tr>';
    function line(label, t, sel) {{
      return '<tr' + (sel ? ' class="sel"' : '') + '><td>' + label + '</td>'
        + '<td class="num">' + de(t.estimate, 4) + '</td>'
        + '<td class="num">' + de(t.std_error, 4) + '</td>'
        + '<td class="num">' + de(t.p_value, 4) + '</td>'
        + '<td>' + stars(t.p_value) + ' <span class="mini">' + sigText(t.p_value) + '</span></td>'
        + '<td class="num">[' + de(t.ci_low, 3) + ' ; ' + de(t.ci_high, 3) + ']</td></tr>';
    }}
    out += line('Konstante (Achsenabschnitt)', fit.intercept, true);
    fit.predictors.forEach(function (key) {{
      var socio = LABELS.socio.indexOf(key) >= 0;
      var label = socio ? key.replace('ln_', 'log ') + ' (Soziographie)'
                        : PREDICTOR[key] + ' ' + party;
      out += line(label, fit.params[key], false);
    }});
    var s = fit.stats;
    out += '</table><p class="mini">Modellgüte: R² = ' + de(s.r2, 3)
      + ' &middot; korrigiertes R² = ' + de(s.adj_r2, 3)
      + ' &middot; R² out-of-fold = ' + de(s.oof_r2, 3)
      + ' &middot; RMSE = ' + de(s.rmse_pp) + ' pp &middot; MAE = ' + de(s.mae_pp) + ' pp</p>';
    $('coef').innerHTML = out;
  }}

  // ---------- Gemeindetabelle ----------
  var sortKey = 'res', sortDir = -1, rowsPer = 25;
  $('mq').innerHTML = ['<option value="__all">alle Landkreise</option>'].concat(
    COUNTY_KEYS.map(function (c) {{ return '<option value="' + c.replace(/"/g, '&quot;') + '">'
      + c + '</option>'; }})).join('');

  function cellValue(m, key) {{
    switch (key) {{
      case 'name': return m.n; case 'landkreis': return m.lk;
      case 'actual': return m.a[party]; case 'pred': return m.p[model][party];
      case 'res': return m.r[model][party];
      case 'ltw21': return m.f[party].LTW21; case 'btw25': return m.f[party].BTW25;
      case 'turnout': return m.t; case 'valid': return m.v;
      case 'einwohner': return m.c.einwohner; case 'flaeche': return m.c.flaeche_km2;
      case 'dichte': return m.c.dichte;
    }}
    return null;
  }}
  function renderTable() {{
    var county = $('mq').value, q = $('q').value.trim().toLowerCase();
    var list = Object.keys(DATA).map(function (c) {{ return {{ c: c, m: DATA[c] }}; }}).filter(function (e) {{
      if (county !== '__all' && (e.m.lk || 'unbekannt') !== county) return false;
      if (q && e.m.n.toLowerCase().indexOf(q) === -1) return false;
      return true;
    }});
    list.sort(function (x, y) {{
      var a = cellValue(x.m, sortKey), b = cellValue(y.m, sortKey);
      var na = (a === null || a === undefined), nb = (b === null || b === undefined);
      if (na && nb) return 0;
      if (na) return 1; if (nb) return -1;
      if (typeof a === 'string') return sortDir > 0 ? a.localeCompare(b, 'de') : b.localeCompare(a, 'de');
      return sortDir > 0 ? a - b : b - a;
    }});
    var max = Math.max.apply(null, list.map(function (e) {{ return cellValue(e.m, 'actual') || 0; }}).concat([1]));
    var slice = rowsPer === 9999 ? list : list.slice(0, rowsPer);
    $('mtable').querySelector('tbody').innerHTML = slice.map(function (e) {{
      var m = e.m, res = m.r[model][party], act = m.a[party];
      var cls = res === null ? '' : (res > 0 ? 'pos' : (res < 0 ? 'neg' : ''));
      var w = act === null ? 0 : Math.max(0, act / max * 100);
      return '<tr><td>' + m.n + '</td><td>' + (m.lk || '') + '</td>'
        + '<td class="num">' + di(m.c.einwohner) + '</td>'
        + '<td class="num">' + (m.c.flaeche_km2 === null ? '–' : de(m.c.flaeche_km2, 1)) + '</td>'
        + '<td class="num">' + (m.c.dichte === null ? '–' : de(m.c.dichte, 1)) + '</td>'
        + '<td class="num bar-cell"><i style="width:' + w.toFixed(1) + '%"></i><b>' + de(act) + '</b></td>'
        + '<td class="num">' + de(m.p[model][party]) + '</td>'
        + '<td class="num ' + cls + '">' + signed(res) + '</td>'
        + '<td class="num">' + de(m.f[party].LTW21) + '</td>'
        + '<td class="num">' + de(m.f[party].BTW25) + '</td>'
        + '<td class="num">' + de(m.t, 1) + '</td>'
        + '<td class="num">' + di(m.v) + '</td></tr>';
    }}).join('') || '<tr><td colspan="12" class="mini">keine Gemeinde gefunden</td></tr>';
    var foot = document.getElementById('table-foot');
    if (!foot) {{
      foot = document.createElement('p');
      foot.id = 'table-foot'; foot.className = 'map-note';
      $('mtable').parentNode.appendChild(foot);
    }}
    foot.innerHTML = list.length + ' Gemeinden'
      + (rowsPer < list.length ? ', angezeigt: ' + slice.length : '');
  }}

  function fillLegendVotes() {{
    // Wichtig: Ist und Prognose nur ueber dieselbe Menge Gemeinden mitteln,
    // sonst werden zwei verschiedene Gebiete verglichen.
    var act = [], pred = [], valid = 0, covered = 0;
    Object.keys(DATA).forEach(function (c) {{
      var m = DATA[c];
      if (m.a[party] === null || m.p[model][party] === null) return;
      covered += 1; valid += m.v;
      act.push([m.a[party], m.v]); pred.push([m.p[model][party], m.v]);
    }});
    function share(rows) {{
      var s = 0, v = 0;
      rows.forEach(function (r) {{ s += r[0] * r[1]; v += r[1]; }});
      return v > 0 ? s / v : null;
    }}
    var a = share(act), p = share(pred);
    var alle = Object.keys(DATA).length;
    $('legend-votes').innerHTML = '<strong>' + party + ' bei der LTW26, Modell ' + model + ':</strong> '
      + 'Prognose <span class="num">' + (p === null ? '–' : de(p) + '&nbsp;%') + '</span>, '
      + 'tatsächlich <span class="num">' + (a === null ? '–' : de(a) + '&nbsp;%') + '</span>, '
      + 'Differenz <span class="num">' + (a === null || p === null ? '–' : signed(a - p) + '&nbsp;pp')
      + '</span>. Nach Wahlberechtigten gewichtet, gemittelt über <strong>' + di(covered)
      + '</strong> von ' + di(alle) + ' Gemeinden'
      + (covered < alle ? ' — außerhalb dieser Gemeinden liegt für diese Partei im Modell keine '
          + 'Prognose vor, deshalb werden die Werte nicht für das ganze Land ausgewiesen.' : '.');
  }}

  function fillMunicipalityOptions() {{
    var codes = Object.keys(DATA).sort(function (a, b) {{
      return DATA[a].n.localeCompare(DATA[b].n, 'de'); }});
    $('municipality').innerHTML = codes.map(function (c) {{
      return '<option value="' + c + '"' + (c === example ? ' selected' : '') + '>' + DATA[c].n
        + ' (' + DATA[c].lk + ')</option>'; }}).join('');
  }}

  function renderAccuracy() {{
    var table = $('acc-table');
    if (!table.querySelector('tbody')) table.appendChild(document.createElement('tbody'));
    table.querySelector('tbody').innerHTML = ACC.map(function (r) {{
      return '<tr class="' + (r.model === model && r.party === party ? 'sel' : '') + '">'
        + '<td>' + r.model + '</td><td>' + r.party + '</td>'
        + '<td class="num">' + r.n + '</td>'
        + '<td class="num">' + de(r.r2, 3) + '</td>'
        + '<td class="num">' + de(r.oof, 3) + '</td>'
        + '<td class="num">' + de(r.rmse) + '</td>'
        + '<td class="num">' + de(r.mae) + '</td>'
        + '<td class="num">' + de(r.w05, 1) + '</td>'
        + '<td class="num">' + de(r.w1, 1) + '</td>'
        + '<td class="num">' + de(r.w2, 1) + '</td>'
        + '<td class="num">' + de(r.r, 3) + '</td></tr>';
    }}).join('');
  }}

  function refresh() {{
    MAP_NOTES = MAP_NOTE_BASE;
    var warn = '';
    if (model === 'wahl6lrw' || model === 'soziolrw') {{
      warn = ' <strong style="color:#8a6100">Hinweis:</strong> Das Modell ' + model
        + ' nutzt die Landratswahl 2025 und ist deshalb nur im Landkreis Ludwigslust-Parchim '
        + 'sinnvoll. Außerhalb dieses Landkreises sind die betreffenden Gemeinden grau.';
    }}
    MAP_NOTES = {{mv: MAP_NOTE_BASE.mv + warn, lup: MAP_NOTE_BASE.lup}};
    paint();
    renderKpi();
    renderExample();
    renderCoef();
    renderAccuracy();
    fillLegendVotes();
    renderTable();
  }}

  $('party').addEventListener('change', function () {{ party = this.value; refresh(); }});
  $('model').addEventListener('change', function () {{ model = this.value; refresh(); }});
  $('metric').addEventListener('change', function () {{ metric = this.value; refresh(); }});
  $('municipality').addEventListener('change', function () {{ example = this.value; renderExample(); }});
  $('mq').addEventListener('change', renderTable);
  $('mrows').addEventListener('change', function () {{ rowsPer = parseInt(this.value, 10); renderTable(); }});
  $('q').addEventListener('input', renderTable);
  $('mmetric').addEventListener('change', function () {{
    sortKey = this.value; sortDir = -1; renderTable();
  }});
  Array.prototype.forEach.call(document.querySelectorAll('#mtable th[data-sort]'), function (th) {{
    th.style.cursor = 'pointer';
    th.addEventListener('click', function () {{
      var key = th.getAttribute('data-sort');
      if (sortKey === key) sortDir = -sortDir; else {{ sortKey = key; sortDir = -1; }}
      renderTable();
    }});
  }});

  var tip = $('tip');
  function moveTip(e) {{
    var pad = 16, w = tip.offsetWidth, h = tip.offsetHeight;
    var x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > window.innerWidth - 8) x = e.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = e.clientY - h - pad;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }}
  document.addEventListener('mouseover', function (e) {{
    var t = e.target;
    if (!t || !t.classList || !t.classList.contains('muni')) return;
    tip.innerHTML = tipFor(t.getAttribute('data-code'));
    tip.style.opacity = '1'; moveTip(e);
  }});
  document.addEventListener('mousemove', function (e) {{
    if (tip.style.opacity === '1') moveTip(e);
  }});
  document.addEventListener('mouseout', function (e) {{
    if (e.target && e.target.classList && e.target.classList.contains('muni')) tip.style.opacity = '0';
  }});

  fillMunicipalityOptions();
  refresh();
}})();
</script>
</body>
</html>
"""

# --------------------------------------------------------------------------
# Seite
# --------------------------------------------------------------------------
def build_fill_pool(payload: dict, regions: Dict[str, dict], parties, models):
    """Fuellfarben fuer alle Kombinationen, mit Duplikat-Eliminierung."""
    municipalities = payload["municipalities"]
    values: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}
    for model in models:
        values[model] = {}
        for metric in METRICS:
            values[model][metric] = {}
            for party in parties:
                column = {"residual": "residual_pp", "predicted": "predicted", "actual": "actual"}[metric]
                values[model][metric][party] = {
                    code: (m[column][party] if column == "actual" else m[column][model][party])
                    for code, m in municipalities.items()
                }
    pool: List[Dict[str, str]] = []
    index: Dict[str, int] = {}
    scales: Dict[str, Dict[str, Dict[str, Dict[str, dict]]]] = {}
    pointer: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}
    for model in models:
        pointer[model] = {}
        scales[model] = {}
        for metric in METRICS:
            pointer[model][metric] = {}
            scales[model][metric] = {}
            for party in parties:
                pointer[model][metric][party] = {}
                scales[model][metric][party] = {}
                for region_name, region in regions.items():
                    fills, lo, hi = color_scale(values[model][metric][party], region["codes"], metric)
                    key = json.dumps(fills, sort_keys=True)
                    if key not in index:
                        index[key] = len(pool)
                        pool.append(fills)
                    pointer[model][metric][party][region_name] = index[key]
                    # Die Skala gehoert zu genau dieser Farbtabelle, also je
                    # Partei und Region getrennt: ein Maximum ueber die
                    # Parteien hinweg wuerde die Legende zu den Farben
                    # widersprechen (die Normalisierung ist je Partei).
                    scales[model][metric][party][region_name] = {"lo": lo, "hi": hi}
    return pool, pointer, scales


def main() -> int:
    payload = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    meta = payload["meta"]
    municipalities = payload["municipalities"]
    fits = payload["fits"]
    accuracy = payload["accuracy"]
    default_model = meta.get("default_model", "wahl6")
    models = [name for name in meta["models"] if name in fits]
    if default_model not in models:
        default_model = models[0]
    # Eine Partei gehoert auf die Seite, wenn sie in *irgendeinem* Modell
    # geschaetzt wurde -- nicht nur, wenn es das Vorgabemodell schafft.
    parties = [p for p in PARTY_ORDER
               if any(p in fits[model] for model in models)]
    default_party = DEFAULT_PARTY if DEFAULT_PARTY in parties else parties[0]

    print(f"[1/5] Karten aufbauen ({len(models)} Modelle x {len(parties)} Parteien x {len(METRICS)} Karteninhalte)")
    paths, titles = load_map_paths()
    geometries, boundaries = build_geometries(paths)
    lup_codes = {code for code in geometries if code.startswith("13076")}
    regions = {"mv": build_region(geometries, boundaries, None),
               "lup": build_region(geometries, boundaries, lup_codes)}
    print(f"      Geometrie: {len(regions['mv']['shapes'])} Gemeinden landesweit, "
          f"{len(regions['lup']['shapes'])} im Landkreis Ludwigslust-Parchim")

    pool, pointer, scales = build_fill_pool(payload, regions, parties, models)
    print(f"      Farbtabellen: {len(pool)} eindeutige (statt {len(models) * len(METRICS) * len(parties) * 2})")

    print("[2/5] Kennzahlen und Tabellen")
    client: Dict[str, dict] = {}
    for code, m in municipalities.items():
        client[code] = {
            "n": m["name"],
            "lk": m["landkreis"] or "",
            "wk": m["wahlkreis"] or "",
            "v": round(m["ltw26_valid_votes"] or 0),
            "t": round(m["ltw26_turnout_pct"] or 0, 1),
            "a": {p: (None if m["actual"][p] is None else round(m["actual"][p], 2)) for p in parties},
            "p": {mo: {p: (None if m["predicted"][mo][p] is None else round(m["predicted"][mo][p], 2))
                       for p in parties} for mo in models},
            "r": {mo: {p: (None if m["residual_pp"][mo][p] is None else round(m["residual_pp"][mo][p], 2))
                       for p in parties} for mo in models},
            "f": {p: {k: (None if v is None else round(v, 2)) for k, v in m["features"][p].items()}
                  for p in parties},
            "c": {k: v for k, v in m.get("context", {}).items()},
        }

    accuracy_rows = []
    for model in models:
        for party in parties:
            entry = accuracy[model]["per_party"].get(party) or {}
            if not entry.get("n"):
                continue
            accuracy_rows.append({
                "model": model, "party": party, "n": entry["n"],
                "r2": entry.get("r2"), "oof": entry.get("oof_r2"),
                "rmse": entry.get("rmse_pp"), "mae": entry.get("mae_pp"),
                "w05": entry.get("within_0_5pp"), "w1": entry.get("within_1pp"),
                "w2": entry.get("within_2pp"), "r": entry.get("pearson_r"),
                "npar": entry.get("n_predictors"),
            })

    fits_json = {
        model: {
            party: {
                "n": f["n"], "intercept": f["intercept"], "params": f["params"],
                "stats": f["stats"], "dropped": f["dropped"], "predictors": f["predictors"],
                "same": f.get("identical_to_wahl6", False),
            } for party, f in party_map.items()
        } for model, party_map in fits.items()
    }

    print("[3/5] HTML zusammensetzen")
    default_fills = {r: pool[pointer[default_model][DEFAULT_METRIC][default_party][r]]
                     for r in regions}
    map_blocks = (
        map_block("mv", "Mecklenburg-Vorpommern — alle Gemeinden",
                  f"Alle {len(regions['mv']['shapes'])} Gemeinden aus dem amtlichen "
                  f"Gemeindeschlüssel.",
                  "<span data-mapnote></span>",
                  svg_region(regions["mv"], default_fills["mv"]))
        + "\n"
        + map_block("lup", "Ludwigslust-Parchim — Kreisdetail",
                    "Die Gemeinden des Landkreises, für die auch die Landratswahl 2025 je Gemeinde "
                    "veröffentlicht wurde.",
                    "<span data-mapnote></span>",
                    svg_region(regions["lup"], default_fills["lup"]))
    )
    map_note_base = {
        "mv": f"{len(regions['mv']['shapes'])} Gemeinden · mit der Maus über eine Fläche fahren. "
              f"Grau = für diese Gemeinde liegt keine Prognose vor.",
        "lup": f"{len(regions['lup']['shapes'])} Gemeinden des Landkreises Ludwigslust-Parchim · "
               f"gleiche Farbskala wie die Landeskarte, deshalb direkt vergleichbar.",
    }

    party_options = "".join(
        f'<option value="{html.escape(p)}"' + (" selected" if p == default_party else "") + ">"
        + html.escape(p) + "</option>" for p in parties)
    model_options = "".join(
        f'<option value="{name}"' + (" selected" if name == default_model else "") + ">"
        + html.escape(meta["models"][name]["title"]) + "</option>" for name in models)
    metric_options = "".join(
        f'<option value="{key}"' + (" selected" if key == DEFAULT_METRIC else "") + ">"
        + html.escape(spec["label"]) + "</option>" for key, spec in METRICS.items())

    counties = sorted({m["landkreis"] or "unbekannt" for m in municipalities.values()})
    generated = datetime.now().strftime("%d.%m.%Y, %H:%M Uhr")

    print("[4/5] Seite schreiben")
    naive_table_html = naive_table(payload, parties, default_model)
    document = PAGE.format(
        css=CSS,
        party_options=party_options,
        model_options=model_options,
        metric_options=metric_options,
        metric_meta=json.dumps({k: {"label": v["label"], "unit": v["unit"], "note": v["note"],
                                    "gradient": v["gradient"]} for k, v in METRICS.items()},
                               ensure_ascii=False, separators=(",", ":")),
        map_blocks=map_blocks,
        map_note_base=json.dumps(map_note_base, ensure_ascii=False),
        explanation=explanation_section(payload, parties, naive_table_html, default_model),
        generated=generated,
        data=json.dumps(client, ensure_ascii=False, separators=(",", ":")),
        fits=json.dumps(fits_json, ensure_ascii=False, separators=(",", ":")),
        acc=json.dumps(accuracy_rows, ensure_ascii=False, separators=(",", ":")),
        models_list=json.dumps(models, ensure_ascii=False),
        parties_list=json.dumps(parties, ensure_ascii=False),
        default_model=json.dumps(default_model),
        default_party=json.dumps(default_party),
        default_metric=json.dumps(DEFAULT_METRIC),
        geo=json.dumps({r: {"viewBox": regions[r]["viewBox"], "codes": regions[r]["codes"]}
                        for r in regions}, ensure_ascii=False, separators=(",", ":")),
        geom=json.dumps({r: [{"c": s["c"], "d": s["d"]} for s in regions[r]["shapes"]]
                         for r in regions}, ensure_ascii=False, separators=(",", ":")),
        outlines=json.dumps({r: regions[r]["outlines"] for r in regions},
                            ensure_ascii=False, separators=(",", ":")),
        fills=json.dumps(pool, ensure_ascii=False, separators=(",", ":")),
        fillptr=json.dumps(pointer, ensure_ascii=False, separators=(",", ":")),
        scales=json.dumps(scales, ensure_ascii=False, separators=(",", ":")),
        labels=json.dumps({"predictors": PREDICTOR_LABELS, "context": CONTEXT_LABELS,
                           "socio": sorted(SOCIO_KEYS)}, ensure_ascii=False),
        municipalities_count=len(municipalities),
        counties_list=json.dumps(counties, ensure_ascii=False),
    )
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"[5/5] -> {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
