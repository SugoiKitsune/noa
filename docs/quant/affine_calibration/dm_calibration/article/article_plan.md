# Risk (Cutting Edge) article — work plan

Companion to `workplan.md` (thesis + repo + data). This file = **article plan**, now built
around **Roland's supervisor review** (`Results/Grinis_review_Popov_thesis.pdf`, 3 Jun 2026,
grade 5/excellent, "recommend for defence" + explicit "before publication" guidance) and the
DM rework directive.

---

## Roland's review = our referee report in advance

He graded it excellent and recommended publication, so the article's job is to **implement his
6 pre-publication recommendations** on DM data. Treat them as the acceptance criteria.

### His 6 specific recommendations (§7) → article tasks
1. **Short-maturity policy-event risk — quantify it.** Analyse the T<6M residual; add a
   *simple scheduled-jump extension* (ECB/BoE/Fed meeting dates for DM), even if not fully
   calibrated, to show how much residual smile is meeting-risk. → new results subsection.
2. **Nested-model comparison — same dataset.** Calibrate the *hierarchy* on the identical
   surface: (a) 1-factor Gaussian HW, (b) 2-factor Gaussian HW, (c) HW + deterministic vol
   term structure, (d) the proposed 2-factor lifted-CIR HWH. Quantifies each ingredient's
   marginal contribution. **He stresses this twice (§4.2, §7) — highest-value addition.**
3. **HJM connection — make it explicit.** State which forward-rate dynamics the short-rate
   factors imply, which no-arbitrage drift restrictions are inherited/imposed, and place the
   model in the Cuchiero–Fontana–Gnoatto taxonomy of affine multiple-curve models.
4. **Relative calibration metrics.** Report relative errors, maturity-bucket errors, ATM-vs-OTM
   decompositions — not only absolute bp RMSE. Makes the Cuchiero/Cao comparison precise.
5. **Surface diagnostics — decompose the arbitrage source.** Distinguish violations from
   (a) market data, (b) MC noise, (c) the parametric model — different remedies. (Dovetails
   with the "why does an affine model need PAVA?" defense answer.)
6. **Editorial refinement.** Fix informal wording / terminological inconsistencies.

### Framing steer (important — from §5)
> "The GPU implementation … should **not be overstated as a separate theoretical
> contribution**. Its value lies in making the chosen model calibratable."

So do **not** lead with "GPU + SPSA methodology." Lead with the **modelling contribution**:
1. the **Gaussian-impossibility result** (analytic; market-independent — survives de-RF),
2. the **multi-curve affine SV model** (HWH + lifted CIR + basis) as a tractable finite-dim
   realization of the HJM multi-curve framework,
3. the **arbitrage-aware calibration** — soft penalties + **hard PAVA projection** (he calls
   the no-arbitrage treatment "one of the best parts").
GPU + Adam-SPSA are the **enabler** that makes it calibratable — mentioned, not headlined.

---

## Working title
"A multi-curve affine stochastic-volatility model for RFR caplet surfaces: calibration and
hard no-arbitrage projection" (drop the RU framing; EUR primary).

## Outline (Risk Cutting Edge, ~4,000 words, ~6 exhibits)

| # | Section | ~Words | Content (Roland's items folded in) |
|---|---|---|---|
| 1 | Motivation | 450 | RFR reform → multi-curve; smiles need SV; **Gaussian-impossibility** as the hook (market-independent) |
| 2 | Model & HJM positioning | 750 | multi-curve affine HWH + lifted CIR + basis; **explicit HJM/affine-multi-curve connection (rec 3)**; implied forward dynamics |
| 3 | Instrument & pricing | 350 | per-period RFR/IBOR caplets, T-forward measure, Bachelier + Newton inversion |
| 4 | Calibration & no-arbitrage | 900 | composite objective; Adam-SPSA + GPU as *enabler*; **PAVA hard projection** as the highlight; **arb-source decomposition (rec 5)** |
| 5 | Empirical results | 1,100 | EUR primary; **nested-model hierarchy (rec 2)**; **relative + bucketed + ATM/OTM metrics (rec 4)**; GBP/USD robustness; swaption out-of-sample; **T<6M jump analysis (rec 1)** |
| 6 | Conclusions & extensions | 300 | scheduled jumps, richer correlation/skew, adjoint sensitivities |

Exhibits: (1) EUR market-vs-model surface overlay; (2) PAVA before/after; (3) **nested-model
RMSE ladder** (new, rec 2); (4) relative-error by maturity bucket / ATM-OTM (new, rec 4);
(5) θ(t),ρ(t) term structures; (6) swaption ATM model-vs-market. All via `deck_charts.py`
rerun on DM.

---

## Reference list (Roland's — these are the lit anchors he expects cited)
Multi-curve / HJM: Cuchiero–Fontana–Gnoatto (2016 F&S; 2019 Math.Fin.), Moreni–Pallavicini
(2014), Pallavicini–Tarenghi (2010), Henrard (2014), Grbac–Runggaldier (2016), Sabelli–Pioppi–
Sitzia–Bormetti (2018), Bienek–Deelstra–Lichtenstern–Zagst (2023), Manti–Molteni (2023),
Chiarella–Kwon (2003, finite-dim affine HJM realizations). RFR / stochastic discontinuities:
Fontana–Grbac–Guembel–Schmidt (2020), Fontana–Grbac–Schmidt (2023). Lévy multi-curve:
Crepey–Grbac–Ngor–Skovmand (2015). Computational: Capriotti–Giles (2024, AAD), Savine
(2020). Plus Gatheral (total-variance no-arb), Cao (calibration benchmark), Abi Jaber (lifted CIR).

---

## Critical path
```
caplet stripper ─► per-period pricer ─► EUR calibration ─► nested-model hierarchy (rec 2)
                                                        ─► GBP/USD + swaption validation
                                                        ─► T<6M jump mini-study (rec 1)  ─► figures
model/HJM/method text (rec 3, market-agnostic) ── draft in parallel ───────────────────────┴─► full draft
```
Sections 1–4 + the HJM positioning (rec 3) can be drafted now. Section 5 + all new exhibits
blocked on the calibration campaign (workplan.md §2–3), of which the **nested hierarchy is now
first-class**, not optional.

## Milestones
- **M1** stripper + per-period pricer + one clean EUR run
- **M2** EUR full calibration + **nested-model ladder** + core figures; draft §§1–4
- **M3** GBP/USD + swaption validation + T<6M jump study + 23-Jul stability
- **M4** full draft (all sections, exhibits, relative metrics)
- **M5** Roland edit pass → revise → submit

## Division of labour
- **Andrey:** runs, figures, method/results, data.  **Roland:** framing, lit, journal, editorial.
- **Me:** stripper + pricer + ingestion + nested-model harness, run orchestration, figures, LaTeX.

---

## Open questions for Roland (trimmed — the review answered several)
Answered by the review: framing (model-led, GPU as enabler), lit anchors (his list), that the
nested comparison + jump study are wanted. Still open:
1. **Breadth:** EUR only, or EUR+GBP+USD robustness? (rec: all three — cheap once the pipeline runs)
2. **Which nested models** exactly in the ladder — his 4, or add 2-factor-CIR-single-curve?
3. **Authorship / affiliations** + does Andrey's name on a bank-adjacent paper need employer sign-off?
4. **Data licensing:** confirm "Source: Bloomberg", public repo = code+params+figures only.
5. **Target & timing:** Risk section (Cutting Edge: Derivatives?), issue/deadline.
6. **Scope of open-source repo:** full engine vs trimmed reference implementation.
