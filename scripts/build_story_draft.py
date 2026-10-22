#!/usr/bin/env python3
"""build_story_draft.py — deterministic producer of ``story-draft.json`` from a published
constellation: the editorial spec that fills the Science Live *story* template
(https://w3id.org/np/RAfjXiLtYji0w1Y3AtKmtL_S21esoh5jprGWAPpRQ1zUk).

It REUSES ``build_story.py``'s gather — constellation fetch, per-outcome figure extraction from
the Outcome's evidence, hero discovery, repo/DOI resolution — the SAME logic that builds the
static story page, and emits JSON instead of HTML. So the platform's ``/np/create/story`` wizard
and the static blog render the same story from the same signed inputs.

Same relationship to the story template as ``build_chain_draft.py`` has to the FORRT chain:
deterministic, nothing invented, every value traced to a signed nanopublication. The one field
left for the human author is ``framing`` (the editorial intro), reviewed/filled before signing.

Env (as for build_story.py): SCIENCELIVE_API_KEY, and SCIENCELIVE_API / SCIENCELIVE_PLATFORM for
a dev-network replication.
"""
import argparse
import json
import sys

import build_story as bs

# Current story template (v4); the platform wizard publishes against its own registry
# value regardless, but the draft should name the same version.
TEMPLATE = "https://w3id.org/np/RAfjXiLtYji0w1Y3AtKmtL_S21esoh5jprGWAPpRQ1zUk"


def _doi(url):
    """Normalise a hasOutcomeRepository value to a doi.org URL when it is a DOI, else None.
    Only a DOI is first-class figure credit (schema:isBasedOn); a bare GitHub archive is not."""
    if not url:
        return None
    if "doi.org/" in url:
        return url
    if url.startswith("10."):
        return "https://doi.org/" + url
    return None


def build_draft(apex, con=None):
    """Editorial spec for one story, derived from the constellation at ``apex``."""
    syn = bs.build_synthesis(apex, con)
    rs = syn["rs"]
    draft = {
        "template": TEMPLATE,
        "apex": apex,
        "headline": rs.get("label") or "Research synthesis",
        "framing": "",  # human-authored — the author fills/confirms this before signing
        "summaries": [],
        "hero": None,
        "figures": [],
        "_notes": [],  # deterministic diagnostics (what was and wasn't found), never published
    }

    # audience tabs: the published AI summaries schema:about this apex, one live per audience
    for i, a in enumerate(bs.fetch_published_summaries(apex), 1):
        draft["summaries"].append({
            "uri": a["np_uri"], "audience": a.get("label"), "position": i,
        })
    if not draft["summaries"]:
        draft["_notes"].append("no published AI summaries found for this apex")

    # hero: an overview figure the individual limbs don't show
    if syn.get("hero"):
        url, name = syn["hero"]
        draft["hero"] = {"contentUrl": url, "caption": "Overview — " + name, "doi": None}
    else:
        draft["_notes"].append("no hero/overview figure discovered")

    # evidence figures: one (or more) per limb, anchored to that limb's Outcome
    for i, d in enumerate(syn["limbs"], 1):
        lv = bs._limb_view(d)
        fig = lv.get("figure")
        if not fig:
            draft["_notes"].append(
                "limb %d (%s) has no figure named in its evidence" % (i, lv.get("heading")))
            continue
        url, name = fig
        draft["figures"].append({
            "contentUrl": url,
            "caption": "%s — %s" % (name, lv.get("heading")),
            "doi": _doi(lv.get("archive")),
            "about": lv.get("outcome_uri") or None,  # the Outcome this figure depicts
            "position": i,
        })
    return draft


def apply_figure_spec(draft, spec):
    """Fill hero/figures from an explicit, committed spec (JSON) when the Outcomes'
    evidence names no figure file. Each figure must point at a published Outcome of
    this constellation; the image URL should be pinned to a release tag."""
    if spec.get("hero"):
        draft["hero"] = spec["hero"]
    if spec.get("figures"):
        draft["figures"] = [dict(f, position=i) for i, f in enumerate(spec["figures"], 1)]
        draft["_notes"] = [n for n in draft["_notes"] if "has no figure named" not in n]
        draft["_notes"].append("figures taken from the committed figure spec")
    if spec.get("hero"):
        draft["_notes"] = [n for n in draft["_notes"] if "no hero" not in n]
    if spec.get("headline"):
        draft["headline"] = spec["headline"]
    if spec.get("framing"):
        draft["framing"] = spec["framing"]
    return draft


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("apex", help="constellation apex nanopublication URI (the Research Synthesis)")
    p.add_argument("-o", "--out", default="story-draft.json", help="output path")
    p.add_argument("--figures", help="JSON spec with hero / figures / framing to apply")
    args = p.parse_args(argv)
    draft = build_draft(args.apex)
    if args.figures:
        with open(args.figures) as f:
            draft = apply_figure_spec(draft, json.load(f))
    with open(args.out, "w") as f:
        json.dump(draft, f, indent=2, ensure_ascii=False)
    print(json.dumps(draft, indent=2, ensure_ascii=False))
    print("\nwrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
