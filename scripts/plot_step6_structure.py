#!/usr/bin/env python
"""Step 6 structure figure (docs/step6/fig_structure.png): writes diagram.html; render it at 2000 x 1420
with any headless browser (e.g. Playwright page.screenshot)."""
# builds diagram.html: 2000 x 1420 canvas, absolute boxes + SVG arrows (layout of the reference figure)
X = [90, 470, 850, 1230, 1610]; BW, BH = 307, 168
C = {"blue": ("#2f5fa8", "#eaf1fb"), "green": ("#2e8a6a", "#e9f4ef"), "purple": ("#7c55a8", "#f1ebf8"),
     "orange": ("#c36a2a", "#fcf1e7"), "gray": ("#6f7a8c", "#f3f5f8")}
boxes, arrows, labels = [], [], []
def box(col, y, color, title, lines, dashed=False):
    b, f = C[color]
    st = "dashed" if dashed else "solid"
    body = "".join(f"<div class='l'>{l}</div>" for l in lines)
    boxes.append(f"<div class='box' style='left:{X[col]}px;top:{y}px;border:2px {st} {b};background:{f}'>"
                 f"<div class='t'>{title}</div>{body}</div>")
def row_arrows(y, color):
    for c in range(4):
        x1, x2, yy = X[c] + BW + 5, X[c + 1] - 8, y + BH / 2
        arrows.append(f"<line x1='{x1}' y1='{yy}' x2='{x2}' y2='{yy}' stroke='{C[color][0]}' stroke-width='2' marker-end='url(#a_{color})'/>")
def label(x, y, text, color, size=17, weight=600):
    labels.append(f"<div style='position:absolute;left:{x}px;top:{y}px;color:{C[color][0]};font-size:{size}px;font-weight:{weight}'>{text}</div>")
def side(y, h, color, text):
    b = C[color][0]
    labels.append(f"<div style='position:absolute;left:55px;top:{y}px;width:3px;height:{h}px;background:{b}'></div>")
    labels.append(f"<div style='position:absolute;left:{32}px;top:{y + h/2}px;transform:translate(-50%,-50%) rotate(-90deg);"
                  f"color:{b};font-size:19px;font-weight:700;letter-spacing:1px;white-space:nowrap'>{text}</div>")

# row 1: the physics core
Y1, Y2, Y3 = 292, 610, 908
side(232, 240, "blue", "01&nbsp; CORE")
label(90, 236, "Step 1 · simulator F<sub>phys</sub>: the 6 h increment of SpeedyWeather.jl from the newest state x(t) (unchanged)", "blue")
box(0, Y1, "purple", "Newest state x(t)", ["KAI 66 channels, 5.625° (32 × 64)", "12 pressure levels 50–1000 hPa", "+ time of the state (hours)"], dashed=True)
box(1, Y1, "green", "p → σ (Python)", ["lat → 32 Gaussian rings, same lons", "p<sub>s</sub> on the T21 orography from z", "dinosaur interp_pressure_to_sigma"])
box(2, Y1, "blue", "SpeedyWeather PrimitiveWet", ["T21, 18 σ levels, leapfrog 30 min, 6 h", "radiation · convection · condensation", "boundary layer · fluxes, SST = skt"])
box(3, Y1, "green", "σ → p (Python)", ["hydrostatic z from T, q, orography", "below ground: u v T q = lowest level", "z: linear extrapolation"])
box(4, Y1, "orange", "D(x) = P(s<sub>6h</sub>) − P(s<sub>0</sub>)", ["both ends spectrally truncated", "round-trip error cancels", "0 for t2m, tisr, tcwv, msl, skt"])
row_arrows(Y1, "blue")
# dashed connector: core increment -> row 2 first box
arrows.append(f"<path d='M {X[4]+BW/2} {Y1+BH} L {X[4]+BW/2} 523 L {X[0]+BW/2} 523 L {X[0]+BW/2} {Y2-10}' fill='none' stroke='{C['blue'][0]}' stroke-width='2' stroke-dasharray='7 6' marker-end='url(#a_blue)'/>")
label(X[0] + BW/2 + 16, 556, "core increment, every 6 h step", "blue", 15, 500)

# row 2: the hybrid field
side(542, 244, "purple", "02&nbsp; HYBRID")
label(510, 556, "Step 2 · operator on the sequence: dx/dt = α ⊙ F<sub>phys</sub>(x<sub>t</sub>) + R<sub>θ</sub>(x<sub>t</sub>, x<sub>t−12</sub>, x<sub>t−24</sub>, F<sub>phys</sub>, t)", "purple")
box(0, Y2, "purple", "Sequence + α ⊙ D(x)", ["x<sub>t</sub> − x<sub>t−12</sub>, x<sub>t−12</sub> − x<sub>t−24</sub>", "per 6 h tendency std (velocity, accel.)", "α as before, frozen"], dashed=True)
box(1, Y2, "green", "SFNO residual R<sub>θ</sub>", ["in: x, 2 differences, F<sub>phys</sub>, time", "width 64, 4 layers, l<sub>max</sub> 16 (618 k)", "head zero init → starts as the core"])
box(2, Y2, "blue", "6 h step", ["G = α D + t6 · R<sub>θ</sub>, sequence held fixed", "torchdiffeq midpoint", "then the sequence slides by 6 h"])
box(3, Y2, "orange", "Loss on the rollout", ["L2 + H¹ Sobolev (λ 1)", "+ PINN hinge (λ 0.05, target 1)", "latitude weighted, per 6 h variance"])
box(4, Y2, "gray", "Training", ["3000 it × batch 4, AdamW 3e-4", "K curriculum 1 → 2 → 4 → 8 steps", "past: ERA5, then own states · 2.9 h"])
row_arrows(Y2, "purple")
arrows.append(f"<path d='M {X[1]+BW/2} {Y2+BH} L {X[1]+BW/2} {Y3-10}' fill='none' stroke='{C['purple'][0]}' stroke-width='2' stroke-dasharray='7 6' marker-end='url(#a_purple)'/>")
label(X[1] + BW/2 + 16, 856, "trained field", "purple", 15, 500)

# row 3: forecast and evaluation
side(852, 244, "green", "03&nbsp; FORECAST")
label(850, 856, "Step 3 · roll out to 6 days and score with WeatherBench2 against the climate ODE", "green")
box(0, Y3, "purple", "ERA5 initial states", ["2017, 72 ICs + the 24 h before each", "8 members: x(t) perturbed,", "past states shared"], dashed=True)
box(1, Y3, "green", "hybrid_speedy_full_seq", ["core + sequence SFNO every 6 h", "24 steps → 144 h", "one Julia worker for all members"], dashed=True)
box(2, Y3, "purple", "WB2 scoring", ["RMSE, ACC (clim 2010–15)", "CRPS, spread/skill, bias", "pg-flow vendored WB2 code"])
box(3, Y3, "orange", "z500 / t850 / t2m RMSE", ["24 h: 210 m · 1.58 K · 1.39 K", "72 h: 492 m · 2.57 K · 2.32 K", "144 h: 778 m · 3.53 K · 3.22 K"])
box(4, Y3, "gray", "What the sequence adds", ["z500 6 h: 87 → 78 m, 24 h: −4 m", "t2m better at every lead", "ens CRPS better, more spread"])
row_arrows(Y3, "green")

foot = """
<div class='cmp'><div class='t'>Against the single-state fields (WB2 2017 z500 RMSE 24 / 72 / 144 h)</div>
<div class='l'>this field: 210 / 492 / 778 m (bias −75 m at 144 h) · same field without the sequence (Step 5): 214 / 484 / 759 m ·
dinosaur + SFNO (Step 4): 209 / 479 / 748 m · climate ODE (Step 1): 220 / 520 / 839 m</div></div>
<div class='rule' style='top:1340px'></div>
<div class='note' style='top:1352px'>Solid: data / loss flow.  Dashed border: frozen or fixed (data, α, trained weights).  Orange: increments, losses and scores.  Purple: inputs and calibration.</div>
<div class='note' style='top:1380px'>Gradients flow through the sliding sequence (the field's own earlier states); training rollouts reach 48 h, which is where the sequence stops helping the deterministic z500.</div>
"""
html = f"""<!doctype html><html><head><meta charset='utf-8'><style>
body{{margin:0;background:#fff;font-family:Inter,'Helvetica Neue',Arial,sans-serif;color:#1f2a3c}}
#c{{position:relative;width:2000px;height:1420px}}
.box{{position:absolute;width:{BW-4}px;height:{BH-4}px;border-radius:12px;text-align:center;box-sizing:content-box}}
.box .t{{font-weight:700;font-size:19px;margin:22px 8px 18px}}
.box .l{{font-size:15.5px;color:#4a5568;margin:0 10px 13px}}
.cmp{{position:absolute;left:90px;top:1168px;width:1876px;height:126px;border:2px solid #6f7a8c;border-radius:12px;background:#f3f5f8;text-align:center}}
.cmp .t{{font-weight:700;font-size:19px;margin:28px 0 22px}} .cmp .l{{font-size:16px;color:#4a5568;margin:0 30px}}
.rule{{position:absolute;left:90px;width:1880px;height:1px;background:#d5dae2}}
.note{{position:absolute;left:90px;font-size:16px;color:#4a5568;white-space:pre}}
h1{{position:absolute;left:90px;top:48px;margin:0;font-size:34px;font-weight:800;color:#1b2740}}
.sub{{position:absolute;left:90px;top:120px;font-size:21px;color:#5a6577}}
.meta{{display:none;position:absolute;right:30px;text-align:right;font-size:17px;color:#4a5568}}
</style></head><body><div id='c'>
<h1>Sequence field: SpeedyWeather core + SFNO on x(t−24), x(t−12), x(t)</h1>
<div class='sub'>hybrid_speedy_full_seq: the operator sees the last 24 h of the flow next to the simulator increment, and the sequence slides along the rollout</div>
<div class='meta' style='top:58px'>Trained on 2010–2015, checkpoint on 2016</div>
<div class='meta' style='top:105px'>Core: T21, 18 σ levels, all parameterizations</div>
<div class='meta' style='top:150px'>Test 2017 (72 ICs), WeatherBench2, 6 days</div>
<div class='rule' style='top:195px'></div>
{''.join(labels)}{''.join(boxes)}
<svg width='2000' height='1420' style='position:absolute;left:0;top:0;pointer-events:none'><defs>
{''.join(f"<marker id='a_{k}' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='{v[0]}'/></marker>" for k, v in C.items())}
</defs>{''.join(arrows)}</svg>
{foot}
</div></body></html>"""
open("diagram.html", "w").write(html)
