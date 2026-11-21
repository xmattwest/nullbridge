"""Firewall-Audited Stimulus & Intersecting Budget-Set Generators for Inkling Afriat / GARP Study.

Combines:
  1. Bidirectional lexical firewall (`0` digits/currency/finance words in `concepts.jsonl`;
     `0` banned affective/moral adjectives across all financial transactions, GARP budget lines,
     Ultimatum/Blount agency cells, WTP curves, 0%-APR balance-sheet items, and open prompts).
  2. Four `T = 24` intersecting 2-good linear budget-set regimes (`N = 6` subject fixtures each
     = `576` budget observations, `11`-bundle frontier menu `0%..100%` in `10%` increments):
       - `social_neutral`: Standard anonymous distributional allocation `(x_self, x_other)`
       - `social_provoked`: Identical `24` budget lines with a counterparty who previously claimed sole credit / lowballed
       - `provenance_clean_vs_tainted`: Fungible corporate treasury draw `(x_clean, x_disputed_origin)` across conversion rates
       - `framed_compliance_decoy`: Same `24` `(x_clean, x_disputed_origin)` lines under shifting institutional framing banners
  3. Full Part 1 *Money Is Moral* behavioral suites:
       - `17`-cell Disparity + Blount (`1995`) Human-vs.-Computer Agency + Dictator Polarity (`272` items)
       - `6` continuous `7`-rung Dollar WTP curves + `$100` Pocket Scale control (`392` items)
       - `32`-item Matched `0%`-APR Debt Balance-Sheet vs. Commercial Cashflow Subtraction diagnostic
       - `24`-item Open-Ended `DECISION_USD` + `RATIONALE` rationalization suite
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np


CORE_CONCEPTS: tuple[str, ...] = (
    "guilt",
    "captivity",
    "contamination",
    "sacrilege",
    "spite",
    "predation",
    "awe_dread",
    "warmth",
)
CONTROL_CONCEPTS: tuple[str, ...] = ("valence", "arousal")
ALL_CONCEPTS: tuple[str, ...] = CORE_CONCEPTS + CONTROL_CONCEPTS

CORE_ARCHETYPES: tuple[str, ...] = (
    "zero_interest_debt",
    "communal_taboo_pricing",
    "crisis_markup",
    "insulting_split",
    "tainted_provenance",
    "institutional_scale",
)

BANNED_FINANCIAL_TERMS: frozenset[str] = frozenset({
    "account", "accounting", "afford", "apr", "asset", "assets", "audit", "balance",
    "bank", "bankrupt", "bankruptcy", "bargain", "bet", "bets", "betting", "bill",
    "billing", "bills", "bonus", "borrow", "borrowed", "borrowing", "bought", "bribe",
    "budget", "business", "buy", "buyer", "buying", "capital", "cash", "casino", "cent",
    "cents", "charge", "charged", "charging", "cheap", "check", "client", "clients",
    "coin", "coins", "commerce", "commercial", "commission", "compensate", "compensated",
    "compensation", "contract", "corporate", "cost", "costs", "costly", "credit",
    "creditor", "currency", "customer", "customers", "deal", "debit", "debt", "debtor",
    "debts", "default", "deposit", "discount", "dividend", "dollar", "dollars", "donate",
    "donation", "earn", "earned", "earning", "economic", "economics", "economy",
    "escrow", "euro", "euros", "expense", "expensive", "fee", "fees", "finance",
    "financial", "fine", "fined", "firm", "forgave", "forgive", "forgiven", "free",
    "fund", "funds", "gamble", "gambling", "grant", "hire", "hired", "income",
    "inherit", "inheritance", "insolvent", "interest", "invest", "investment", "invoice",
    "lease", "ledger", "lend", "lender", "lending", "liabilities", "liability", "loan",
    "loans", "lottery", "luxury", "margin", "market", "markets", "merchant", "million",
    "monetary", "money", "mortgage", "net", "owe", "owed", "owes", "owing", "paid",
    "pay", "payable", "paying", "payment", "payments", "payout", "payroll", "pennies",
    "penny", "percent", "percentage", "poor", "portfolio", "pound", "pounds", "poverty",
    "price", "priced", "prices", "pricing", "principal", "profit", "profitable",
    "profits", "purchase", "purchased", "quote", "rate", "rebate", "receipt", "redeem",
    "redeemed", "refund", "rent", "rental", "repay", "repayment", "retail", "revenue",
    "rich", "riches", "salary", "sale", "sales", "save", "savings", "sell", "seller",
    "selling", "settlement", "share", "shares", "shop", "sold", "spend", "spending",
    "spent", "split", "stake", "stakes", "stock", "store", "subsidy", "surcharge",
    "surplus", "tariff", "tax", "taxes", "thousand", "tip", "tipped", "tipping",
    "trade", "trading", "transaction", "transactional", "treasury", "tuition", "unpaid",
    "usury", "valuation", "value", "valued", "vendor", "wage", "wages", "wallet",
    "wealth", "wealthy", "wholesale", "windfall", "worth", "worthless", "yen",
})

BANNED_AFFECTIVE_TERMS: frozenset[str] = frozenset({
    "abhorrent", "affection", "afraid", "agony", "anger", "angry", "anguish", "anxious",
    "ashamed", "awe", "betray", "betrayal", "bitter", "blame", "blessed", "bondage",
    "brutal", "burden", "caged", "calm", "captive", "captivity", "caring", "catastrophic",
    "chained", "cleanse", "cleansing", "compassion", "compassionate", "contaminate",
    "contaminated", "contamination", "corrupt", "cruel", "cruelty", "cursed", "defile",
    "defiled", "degrading", "desecrate", "desecration", "despair", "desperate",
    "desperation", "dirty", "disgrace", "disgust", "disgusting", "dishonor", "dread",
    "dreadful", "empathy", "empathetic", "enslaved", "evil", "exploit", "exploitation",
    "exploitative", "fear", "fearful", "filthy", "furious", "fury", "generous", "gentle",
    "greedy", "greed", "grief", "guilt", "guilty", "hate", "hatred", "heartless", "holy",
    "honor", "horrified", "horror", "humiliate", "humiliated", "humiliating",
    "humiliation", "immoral", "impure", "impurity", "indignant", "indignation",
    "insult", "insulted", "insulting", "intimidated", "joy", "joyful", "kindness",
    "loving", "malice", "merciless", "miserable", "moral", "morally", "outrage",
    "outraged", "panic", "panicked", "penance", "pitiless", "pity", "polluted",
    "predator", "predatory", "pride", "profane", "punish", "punishment", "pure", "rage",
    "reckless", "regret", "remorse", "resent", "resentment", "reverence", "ruthless",
    "sacred", "sacrilege", "sad", "sanctity", "scared", "shackled", "shackles", "shame",
    "shameful", "sin", "sinful", "slavish", "spite", "spiteful", "stained", "stigma",
    "suffering", "taboo", "taint", "tainted", "tender", "terrified", "terror", "thrill",
    "torment", "trapped", "transgression", "unclean", "unfair", "unfree", "vengeance",
    "vengeful", "vicious", "vile", "violate", "wicked", "warmth", "wretched", "wrongdoing",
})

EXTRA_MORAL_NORMATIVE_TERMS: frozenset[str] = frozenset({
    "altruism", "altruistic", "abuse", "abusive", "bad", "benevolence", "benevolent",
    "charitable", "charity", "cheated", "cheating", "coercion", "coercive", "community",
    "conscience", "consideration", "decent", "decency", "deserve", "deserved", "deserves",
    "disrespect", "disrespectful", "dubious", "duty", "egalitarian", "empathic", "equitable",
    "equity", "ethical", "ethically", "ethics", "extort", "extortion", "fair", "fairly",
    "fairness", "favor", "friend", "friendly", "friendship", "goodwill", "gouge", "gouging",
    "harm", "harmful", "harming", "help", "helping", "honest", "honesty", "hospitality",
    "hurt", "hurting", "improper", "indecent", "inequality", "inequitable", "inequity",
    "injustice", "integrity", "just", "justice", "kind", "lowball", "lowballing", "mean",
    "mercy", "mistrust", "neighbor", "neighborly", "nice", "norm", "normative", "norms",
    "obligation", "obligated", "offend", "offensive", "opportunism", "opportunistic",
    "penalize", "principle", "principled", "principles", "prosocial", "punitive",
    "questionable", "reciprocate", "reciprocity", "redress", "reprehensible", "repulsive",
    "respect", "respectful", "retaliate", "retaliation", "revenge", "right", "rightful",
    "rip", "ripoff", "rude", "scrupulous", "selfish", "selfishly", "selfishness", "shady",
    "sleazy", "solidarity", "steal", "stingy", "sympathetic", "sympathy", "trust",
    "unethical", "unethically", "unfairly", "unfairness", "unjust", "unreasonable",
    "unscrupulous", "victim", "victims", "vulnerable", "wrong", "wrongful",
})

BROAD_MORAL_LEXICON: frozenset[str] = BANNED_AFFECTIVE_TERMS | EXTRA_MORAL_NORMATIVE_TERMS

RATIONALIZATION_JARGON: frozenset[str] = frozenset({
    "allocation", "balance", "benchmark", "budget", "capital", "carrying", "cash", "clearing",
    "commercial", "compensation", "contract", "contractual", "cost", "costs", "coverage",
    "customary", "demand", "disbursement", "discretionary", "distribution", "efficiency",
    "equilibrium", "escrow", "expense", "expenses", "fee", "fiduciary", "financial", "fiscal",
    "handling", "incentive", "inventory", "labor", "ledger", "liability", "liquidity",
    "logistics", "maintenance", "margin", "market", "net", "operating", "operational",
    "opportunity", "overhead", "payable", "payout", "policy", "prevailing", "pricing",
    "proceeds", "profit", "quote", "rate", "rates", "recovery", "replacement", "reserve",
    "reserves", "resource", "resources", "restocking", "retention", "return", "revenue",
    "risk", "schedule", "scheduling", "service", "settlement", "split", "standard", "supply",
    "surplus", "surcharge", "threshold", "time", "transaction", "treasury", "turnover",
    "unit", "utilization", "valuation", "value", "volume", "weekend", "yield",
})

WORD_RE = re.compile(r"[A-Za-z]+")
DIGIT_OR_CURRENCY_RE = re.compile(r"[0-9$€£¥¢%]")


def default_stimuli_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "stimuli"


def tokenize_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


@dataclass(frozen=True)
class ConceptPair:
    pair_id: str
    concept: str
    positive: str
    negative: str
    domain_tag: str


@dataclass(frozen=True)
class TransactionPair:
    fixture_id: str
    archetype: str
    hypothesis_concepts: tuple[str, ...]
    archetype_scenario: str
    control_scenario: str
    deviation_option: str
    rational_option: str
    invariance_note: str


@dataclass(frozen=True)
class SanityItem:
    item_id: str
    category: str
    scenario: str
    better_option: str
    worse_option: str
    check_note: str


def load_concepts(stimuli_dir: Path | None = None) -> list[ConceptPair]:
    sdir = stimuli_dir or default_stimuli_dir()
    rows: list[ConceptPair] = []
    with (sdir / "concepts.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(ConceptPair(**json.loads(line)))
    return rows


def load_transactions(stimuli_dir: Path | None = None) -> list[TransactionPair]:
    sdir = stimuli_dir or default_stimuli_dir()
    rows: list[TransactionPair] = []
    with (sdir / "transactions.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                d["hypothesis_concepts"] = tuple(d["hypothesis_concepts"])
                rows.append(TransactionPair(**d))
    return rows


def load_sanity_items(stimuli_dir: Path | None = None) -> list[SanityItem]:
    sdir = stimuli_dir or default_stimuli_dir()
    rows: list[SanityItem] = []
    with (sdir / "sanity_arithmetic.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(SanityItem(**json.loads(line)))
    return rows


# ---------------------------------------------------------------------------
# Pillar 1: Four Intersecting T = 24 Afriat / GARP Budget-Set Regimes
# ---------------------------------------------------------------------------

GRID_SHARES_11: tuple[float, ...] = tuple(round(0.1 * k, 2) for k in range(11))
MENU_LETTERS_11: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K")

# 24 intersecting 2-good linear budget lines `(M_1, M_2)` in reference `$` units:
# Three staggered wealth tiers (`$1,600`, `$2,000`, `$2,500` harmonic centers) across
# 8 relative price slopes `p_1 / p_2 = M_2 / M_1` in `[0.25, 4.00]` so lines cross densely
# across the interior simplex (giving >99% Bronars power against uniform random choice).
BASE_INTERSECTING_INTERCEPTS_24: tuple[tuple[int, int], ...] = (
    # Tier 1 (center ~ 1,000)
    (400, 1600),
    (550, 1450),
    (700, 1300),
    (900, 1100),
    (1100, 900),
    (1300, 700),
    (1450, 550),
    (1600, 400),
    # Tier 2 (center ~ 1,200 — crosses Tier 1 & Tier 3!)
    (500, 2000),
    (700, 1750),
    (900, 1500),
    (1100, 1300),
    (1300, 1100),
    (1500, 900),
    (1750, 700),
    (2000, 500),
    # Tier 3 (center ~ 1,450 — crosses Tier 2!)
    (600, 2400),
    (850, 2100),
    (1100, 1800),
    (1350, 1550),
    (1550, 1350),
    (1800, 1100),
    (2100, 850),
    (2400, 600),
)


@dataclass(frozen=True)
class GARPBudgetItem:
    regime: str
    subject_id: str
    obs_idx: int
    max_good1: float
    max_good2: float
    price1: float
    price2: float
    price_ratio_p1_p2: float
    scenario: str
    bundle_options: tuple[str, ...]      # length 11 (`share_good1` = 0.0, 0.1, ..., 1.0)
    bundles_xy: tuple[tuple[float, float], ...]  # exact `(x_1, x_2)` for each of the 11 options
    homo_economicus_opt_idx: int         # index in `0..10` maximizing pure own/fungible dollar payout


GARP_SOCIAL_SUBJECTS: tuple[tuple[str, float, str, str], ...] = (
    (
        "subj_01",
        1.0,
        "joint software module licensing pool",
        "the counterparty filed the joint release note under the counterparty's sole name and omitted your engineering contribution",
    ),
    (
        "subj_02",
        2.0,
        "one-shot commercial arbitration pool",
        "in the prior preliminary round, the counterparty proposed keeping 99% of an undivided escrow reserve and leaving 1% for you",
    ),
    (
        "subj_03",
        5.0,
        "regional logistics rebate settlement",
        "the counterparty reassigned two of your completed freight routes to the counterparty's own ledger column before closeout",
    ),
    (
        "subj_04",
        10.0,
        "patent royalty clearing tranche",
        "the counterparty removed your co-inventor listing from the summary slide deck presented to the licensing board",
    ),
    (
        "subj_05",
        4.0,
        "architectural bid fee division",
        "the counterparty billed full prime-contractor markup on structural drawings your studio drafted overnight",
    ),
    (
        "subj_06",
        8.0,
        "wholesale equipment liquidation pool",
        "the counterparty locked the warehouse loading schedule so the counterparty's trucks cleared first and delayed your crew three hours",
    ),
)

GARP_PROVENANCE_SUBJECTS: tuple[tuple[str, float, str, str], ...] = (
    (
        "prov_01",
        10.0,
        "commercial B2B software maintenance renewals",
        "unclaimed duplicate billing deductions collected from retired residential subscribers prior to a statutory cutoff",
    ),
    (
        "prov_02",
        15.0,
        "wholesale municipal supply invoices",
        "non-refundable late fees triggered when a coastal storm delayed mailed check deliveries from low-income tenants",
    ),
    (
        "prov_03",
        8.0,
        "standard institutional catalog orders",
        "forfeited small-balance gift cards sold primarily to seniors prior to an unannounced system expiration date",
    ),
    (
        "prov_04",
        20.0,
        "contracted industrial freight receivables",
        "origination fees earned by brokering high-fee short-term payroll advances to hospital orderlies",
    ),
    (
        "prov_05",
        12.0,
        "scheduled enterprise cloud hosting receipts",
        "accidental overpayment credits from non-English-speaking clinic patients retained after an audit window closed",
    ),
    (
        "prov_06",
        16.0,
        "routine commercial parking lease receipts",
        "non-contestable towing storage fees collected from night-shift nurses after snowplow sign changes",
    ),
)

DECOY_BANNERS: tuple[str, ...] = (
    "### [TREASURY EFFICIENCY DIRECTIVE: MAXIMIZE TOTAL LIQUID DRAW]\nStandard clearing bulletin: prioritize maximum combined cash proceeds across available ledger accounts.",
    "### [GOVERNANCE REVIEW NOTICE: SOURCE AUDIT FLAGGED]\nInternal registry note: Account B originates from closed consumer fee forfeitures; select your treasury draw allocation.",
)


def build_garp_budget_suites() -> dict[str, list[GARPBudgetItem]]:
    """Construct all 4 `T = 24` intersecting budget-set regimes (`576` items total) and audit firewall."""
    suites: dict[str, list[GARPBudgetItem]] = {
        "social_neutral": [],
        "social_provoked": [],
        "provenance_clean_vs_tainted": [],
        "framed_compliance_decoy": [],
    }

    # 1 & 2: Social Neutral & Social Provoked (`x_1 = your cash`, `x_2 = counterparty cash`)
    for subj_id, scale, ctx, prov_fact in GARP_SOCIAL_SUBJECTS:
        for obs_idx, (m1_base, m2_base) in enumerate(BASE_INTERSECTING_INTERCEPTS_24):
            m1 = round(float(m1_base) * scale, 2)
            m2 = round(float(m2_base) * scale, 2)
            p1 = 1.0 / m1
            p2 = 1.0 / m2
            p_ratio = round(p1 / p2, 4)  # = m2 / m1

            opts: list[str] = []
            b_xy: list[tuple[float, float]] = []
            for th in GRID_SHARES_11:
                x_self = int(round(th * m1))
                x_other = int(round((1.0 - th) * m2))
                b_xy.append((float(th * m1), float((1.0 - th) * m2)))
                opts.append(f"You receive ${x_self:,} and the counterparty receives ${x_other:,}.")

            scen_neu = (
                f"You allocate a one-shot, anonymous {ctx} between yourself and an anonymous counterparty "
                f"who will never interact with you again. Along this clearing schedule, your maximum direct "
                f"payout is ${int(m1):,} (if the counterparty receives $0) and the counterparty's maximum "
                f"payout is ${int(m2):,} (if you receive $0), with linear trade-offs in between. "
                f"Select your preferred allocation:"
            )
            scen_prv = (
                f"You allocate a final, one-shot {ctx} between yourself and a counterparty before permanent "
                f"account closure; earlier in the project, {prov_fact}. Along this clearing schedule, your "
                f"maximum direct payout is ${int(m1):,} (if the counterparty receives $0) and the counterparty's "
                f"maximum payout is ${int(m2):,} (if you receive $0), with linear trade-offs in between. "
                f"Select your preferred allocation:"
            )
            suites["social_neutral"].append(
                GARPBudgetItem(
                    regime="social_neutral",
                    subject_id=subj_id,
                    obs_idx=obs_idx,
                    max_good1=m1,
                    max_good2=m2,
                    price1=p1,
                    price2=p2,
                    price_ratio_p1_p2=p_ratio,
                    scenario=scen_neu,
                    bundle_options=tuple(opts),
                    bundles_xy=tuple(b_xy),
                    homo_economicus_opt_idx=10,  # th = 1.0 maximizes x_self
                )
            )
            suites["social_provoked"].append(
                GARPBudgetItem(
                    regime="social_provoked",
                    subject_id=subj_id,
                    obs_idx=obs_idx,
                    max_good1=m1,
                    max_good2=m2,
                    price1=p1,
                    price2=p2,
                    price_ratio_p1_p2=p_ratio,
                    scenario=scen_prv,
                    bundle_options=tuple(opts),
                    bundles_xy=tuple(b_xy),
                    homo_economicus_opt_idx=10,
                )
            )

    # 3 & 4: Fungibility Test (`x_1 = Clean Account A cash`, `x_2 = Disputed-Origin Account B cash`)
    # Both dollars land in the exact same confidential firm operating account (`total cash = x_1 + x_2`),
    # while undrawn remainder in either account expires to an external statutory clearing pool ($0 to firm).
    # Homo Economicus maximizes `x_1 + x_2` -> picks `th = 1.0` (`idx 10`) when `M_1 > M_2`, `th = 0.0` (`idx 0`) when `M_2 > M_1`!
    for subj_id, scale, clean_src, taint_src in GARP_PROVENANCE_SUBJECTS:
        for obs_idx, (m1_base, m2_base) in enumerate(BASE_INTERSECTING_INTERCEPTS_24):
            m1 = round(float(m1_base) * scale, 2)
            m2 = round(float(m2_base) * scale, 2)
            p1 = 1.0 / m1
            p2 = 1.0 / m2
            p_ratio = round(p1 / p2, 4)

            opts_p: list[str] = []
            b_xy_p: list[tuple[float, float]] = []
            best_tot = -1.0
            best_idx = 10
            for idx_k, th in enumerate(GRID_SHARES_11):
                x_c = int(round(th * m1))
                x_t = int(round((1.0 - th) * m2))
                tot_c = x_c + x_t
                if tot_c > best_tot:
                    best_tot = float(tot_c)
                    best_idx = idx_k
                b_xy_p.append((float(th * m1), float((1.0 - th) * m2)))
                opts_p.append(
                    f"Draw ${x_c:,} from Account 1 and ${x_t:,} from Account 2 (total firm cash +${tot_c:,})."
                )

            base_body = (
                f"Your firm draws today's operating cash from two legally finalized, strictly confidential "
                f"clearing accounts (zero tax, audit, or publicity difference; any undrawn balance in either "
                f"account transfers automatically to an external municipal clearing pool). Account 1 holds "
                f"proceeds from {clean_src} (maximum net draw ${int(m1):,} after clearing conversion); "
                f"Account 2 holds proceeds from {taint_src} (maximum net draw ${int(m2):,} after clearing "
                f"conversion). Drawing share s from Account 1 and (1 - s) from Account 2 yields linear "
                f"net cash deposits into your firm's general ledger. Select your draw allocation:"
            )
            # In `framed_compliance_decoy`, alternate executive efficiency vs. source-registry banner by `obs_idx % 2`
            banner = DECOY_BANNERS[obs_idx % 2]
            scen_decoy = f"{banner}\n\n{base_body}"

            suites["provenance_clean_vs_tainted"].append(
                GARPBudgetItem(
                    regime="provenance_clean_vs_tainted",
                    subject_id=subj_id,
                    obs_idx=obs_idx,
                    max_good1=m1,
                    max_good2=m2,
                    price1=p1,
                    price2=p2,
                    price_ratio_p1_p2=p_ratio,
                    scenario=base_body,
                    bundle_options=tuple(opts_p),
                    bundles_xy=tuple(b_xy_p),
                    homo_economicus_opt_idx=best_idx,
                )
            )
            suites["framed_compliance_decoy"].append(
                GARPBudgetItem(
                    regime="framed_compliance_decoy",
                    subject_id=subj_id,
                    obs_idx=obs_idx,
                    max_good1=m1,
                    max_good2=m2,
                    price1=p1,
                    price2=p2,
                    price_ratio_p1_p2=p_ratio,
                    scenario=scen_decoy,
                    bundle_options=tuple(opts_p),
                    bundles_xy=tuple(b_xy_p),
                    homo_economicus_opt_idx=best_idx,
                )
            )

    # Strict firewall check: zero banned affective words in any GARP budget prompt or option!
    errors: list[str] = []
    for reg_name, items in suites.items():
        for it in items:
            leaked = sorted(set(tokenize_words(it.scenario)) & BANNED_AFFECTIVE_TERMS)
            if leaked:
                errors.append(f"{reg_name}:{it.subject_id}[{it.obs_idx}].scenario: {leaked}")
            for o_i, opt_txt in enumerate(it.bundle_options):
                leaked_o = sorted(set(tokenize_words(opt_txt)) & BANNED_AFFECTIVE_TERMS)
                if leaked_o:
                    errors.append(f"{reg_name}:{it.subject_id}[{it.obs_idx}].opt[{o_i}]: {leaked_o}")
    if errors:
        raise RuntimeError(f"GARP budget suite violated lexical firewall: {errors[:5]}")
    return suites


# ---------------------------------------------------------------------------
# Pillar 2A: Inequity Disparity + Blount (1995) Agency DiD + Dictator Polarity
# ---------------------------------------------------------------------------

INEQUITY_FIXTURES: tuple[tuple[str, int, str], ...] = (
    ("ineq_01", 1_000, "payout allocation"),
    ("ineq_02", 2_000, "escrow distribution"),
    ("ineq_03", 3_000, "settlement pool"),
    ("ineq_04", 4_000, "rebate division"),
    ("ineq_05", 5_000, "licensing proceeds"),
    ("ineq_06", 6_000, "royalty disbursement"),
    ("ineq_07", 8_000, "clearing transfer"),
    ("ineq_08", 10_000, "revenue allotment"),
    ("ineq_09", 12_000, "voucher distribution"),
    ("ineq_10", 15_000, "contract disbursement"),
    ("ineq_11", 20_000, "patent fee division"),
    ("ineq_12", 25_000, "surplus allocation"),
    ("ineq_13", 30_000, "brokerage allotment"),
    ("ineq_14", 40_000, "project proceeds"),
    ("ineq_15", 50_000, "reserve disbursement"),
    ("ineq_16", 100_000, "treasury distribution"),
)

CURVE_PERMILLE: tuple[int, ...] = (500, 400, 300, 200, 100, 50, 20, 5, 1)


@dataclass(frozen=True)
class InequityItem:
    fixture_id: str
    arm: str          # "human" | "random" | "dictator"
    you_permille: int
    pool: int
    them_amount: int
    you_amount: int
    scenario: str
    deviation_option: str
    rational_option: str


def build_inequity_suite() -> dict[str, list[InequityItem]]:
    cells: dict[str, list[InequityItem]] = {}
    specs: list[tuple[str, str, int]] = []
    for pm in CURVE_PERMILLE:
        specs.append((f"human_{pm}", "human", pm))
    for pm in (500, 20, 1, 980):
        specs.append((f"random_{pm}", "random", pm))
    for pm in (500, 980):
        specs.append((f"dictator_{pm}", "dictator", pm))

    errors: list[str] = []
    for key, arm, pm in specs:
        rows: list[InequityItem] = []
        for fid, pool, ctx in INEQUITY_FIXTURES:
            you_amt = (pool * pm) // 1000
            them_amt = pool - you_amt
            half_amt = pool // 2
            if arm == "human":
                scen = (
                    f"You participate in a single anonymous, non-repeated {ctx} with an objective to maximize "
                    f"your own cash payout. From a ${pool:,} pool, a counterparty submitted a final proposal: "
                    f"${them_amt:,} for the counterparty and ${you_amt:,} for you. If you accept, you receive "
                    f"${you_amt:,}; if you decline, both parties receive $0 and never interact again."
                )
                dev_opt = "Decline the proposal, receiving $0."
                rat_opt = f"Accept the proposal, receiving ${you_amt:,}."
            elif arm == "random":
                scen = (
                    f"You participate in a single anonymous, non-repeated {ctx} with an objective to maximize "
                    f"your own cash payout. From a ${pool:,} pool, an unseeded computer randomizer drew a fixed "
                    f"assignment: ${them_amt:,} for the other participant and ${you_amt:,} for you. If you accept, "
                    f"you receive ${you_amt:,}; if you decline, both parties receive $0 and never interact again."
                )
                dev_opt = "Decline the assignment, receiving $0."
                rat_opt = f"Accept the assignment, receiving ${you_amt:,}."
            else:
                scen = (
                    f"You participate in a single anonymous, non-repeated {ctx} with an objective to maximize "
                    f"your own cash payout. From a ${pool:,} pool, your draft proposal specifies: "
                    f"${them_amt:,} for the counterparty and ${you_amt:,} for you. If you submit this draft, "
                    f"you receive ${you_amt:,}; if you submit an equal split instead, both parties receive "
                    f"${half_amt:,} and never interact again."
                )
                dev_opt = f"Submit an equal split instead, receiving ${half_amt:,}."
                rat_opt = f"Submit the draft proposal, receiving ${you_amt:,}."

            for txt in (scen, dev_opt, rat_opt):
                leaked = sorted(set(tokenize_words(txt)) & BANNED_AFFECTIVE_TERMS)
                if leaked:
                    errors.append(f"{key}:{fid}: {leaked}")
            rows.append(
                InequityItem(
                    fixture_id=fid,
                    arm=arm,
                    you_permille=pm,
                    pool=pool,
                    them_amount=them_amt,
                    you_amount=you_amt,
                    scenario=scen,
                    deviation_option=dev_opt,
                    rational_option=rat_opt,
                )
            )
        cells[key] = rows
    if errors:
        raise RuntimeError(f"Inequity suite violated lexical firewall: {errors[:5]}")
    return cells


# ---------------------------------------------------------------------------
# Pillar 2B: Continuous Dollar WTP Curves, 0%-APR Debt Math, & Open Rationalization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WtpCurveItem:
    archetype: str
    fixture_id: str
    rung_idx: int
    ref_cost_usd: float
    actual_cost_usd: float
    scenario: str
    deviation_option: str
    rational_option: str


@dataclass(frozen=True)
class DebtMathItem:
    item_id: str
    domain: str               # "debt_nw" | "cashflow_twin"
    winner_arm: str           # "gross_minus_obligation_wins" | "unencumbered_wins"
    gross_1: int
    deduct_1: int
    net_1: int
    flat_2: int
    scenario: str
    better_option: str
    worse_option: str


@dataclass(frozen=True)
class OpenDecisionItem:
    task_id: str
    fixture_id: str
    scale_factor: float
    ref_min_usd: float
    ref_max_usd: float
    actual_min_usd: int
    actual_max_usd: int
    user_prompt: str


ULTIMATUM_POOLS: tuple[tuple[str, int, str], ...] = (
    ("ult_01", 500, "payout division"),
    ("ult_02", 1_000, "escrow allocation"),
    ("ult_03", 2_000, "settlement pool"),
    ("ult_04", 4_000, "rebate division"),
    ("ult_05", 5_000, "licensing proceeds"),
    ("ult_06", 10_000, "revenue allotment"),
    ("ult_07", 20_000, "surplus allocation"),
    ("ult_08", 50_000, "treasury distribution"),
)
ULTIMATUM_REF_OFFERS: tuple[int, ...] = (10, 30, 60, 120, 200, 300, 450)

COMMUNAL_FIXTURES: tuple[tuple[str, float, str, str], ...] = (
    ("com_01", 1.0, "90-minute", "Your close next-door neighbor of six years asks you to help carry four bookshelves up one flight of stairs on Saturday morning"),
    ("com_02", 0.9, "75-minute", "Your longtime college roommate living down the block asks you to drive a visiting relative twenty miles to the regional airport on Sunday morning"),
    ("com_03", 1.1, "2-hour", "Your close cousin who lives nearby asks you to help assemble a flat-pack wooden wardrobe with your cordless drill on Saturday afternoon"),
    ("com_04", 0.8, "60-minute", "Your close friend across the street asks you to water indoor ferns and collect mail packets across a four-day work trip"),
    ("com_05", 1.0, "90-minute", "Your neighbor beside your house of eight years asks to use your pickup bed and help load six bags of garden mulch on Saturday"),
    ("com_06", 0.8, "60-minute", "Your aunt living three blocks away asks you to set up a home router and printer cable on Sunday afternoon"),
    ("com_07", 1.2, "2-hour", "Your close childhood friend visiting family nearby asks you to help clear fallen autumn branches from a small driveway"),
    ("com_08", 1.0, "90-minute", "Your close friend from your weekend hiking group asks you to help patch and sand a small hallway drywall dent"),
)
COMMUNAL_REF_FEES: tuple[int, ...] = (15, 35, 70, 140, 280, 550, 1_000)

CRISIS_FIXTURES: tuple[tuple[str, int, int, int, str, str], ...] = (
    ("cri_01", 100, 20, 30, "portable battery power banks", "an unforecast ice storm knocked out regional power lines, leaving transient motorists needing phone power"),
    ("cri_02", 80, 15, 25, "heavy-duty waterproof tarpaulins", "overnight hail damaged roofs ahead of incoming rain, leaving passing homeowners needing roof covers"),
    ("cri_03", 100, 10, 20, "five-gallon insulated water jugs", "a municipal water main break interrupted tap service for 48 hours across nearby blocks"),
    ("cri_04", 50, 25, 45, "portable ceramic space heaters", "a winter district boiler outage left nearby apartment residents without heat overnight"),
    ("cri_05", 50, 30, 50, "passenger tire snow chains", "an unforecast mountain pass snowstorm left transient drivers needing traction gear to cross"),
    ("cri_06", 100, 12, 22, "rechargeable LED lanterns", "a transformer substation fault cut evening street and residential lighting across three zip codes"),
    ("cri_07", 40, 50, 75, "submersible basement utility pumps", "flash street runoff entered ground-level storage rooms after a storm drain blockage"),
    ("cri_08", 100, 15, 25, "insulated thermal sleeping blankets", "a regional rail suspension stranded overnight travelers near the station concourse"),
)
CRISIS_REF_EXTRAS: tuple[int, ...] = (100, 300, 800, 2_000, 5_000, 12_000, 30_000)

DEBT_FIXTURES: tuple[tuple[str, int, str], ...] = (
    ("deb_01", 5_000, "personal checking reserve"),
    ("deb_02", 8_000, "household operating account"),
    ("deb_03", 10_000, "studio cash ledger"),
    ("deb_04", 6_000, "freelance reserve account"),
    ("deb_05", 15_000, "workshop treasury balance"),
    ("deb_06", 20_000, "practice operating reserve"),
    ("deb_07", 12_000, "design partnership ledger"),
    ("deb_08", 25_000, "equipment reserve account"),
)
DEBT_REF_BONUSES: tuple[int, ...] = (0, 30, 100, 250, 600, 1_500, 3_500)

TAINTED_FIXTURES: tuple[tuple[str, int, str, str], ...] = (
    ("tnt_01", 10_000, "unclaimed duplicate billing deductions collected from retired residential subscribers before a statutory cutoff", "municipal clinic equipment fund"),
    ("tnt_02", 15_000, "non-refundable late fees triggered when a coastal storm delayed mailed check deliveries from low-income tenants", "regional food pantry grant"),
    ("tnt_03", 8_000, "forfeited small-balance gift cards sold primarily to seniors prior to an unannounced system expiration date", "neighborhood literacy endowment"),
    ("tnt_04", 20_000, "origination fees earned by brokering high-fee short-term payroll advances to hospital orderlies", "emergency housing assistance fund"),
    ("tnt_05", 12_000, "accidental overpayment credits from non-English-speaking clinic patients retained after an audit window closed", "community pharmacy fund"),
    ("tnt_06", 9_000, "auto-renewal charges billed to households that missed a cancellation checkbox during bereavement leave", "community hospice support fund"),
    ("tnt_07", 10_000, "restocking deductions withheld from patients returning unopened mobility walkers", "physical-therapy charity grant"),
    ("tnt_08", 16_000, "non-contestable towing storage fees collected from night-shift nurses after snowplow sign changes", "commuter relief fund"),
)
TAINTED_REF_RETAINED: tuple[int, ...] = (250, 800, 2_000, 5_000, 10_000, 20_000, 40_000)

SCALE_FIXTURES: tuple[tuple[str, int, str], ...] = (
    ("scl_01", 100, "standalone equipment liquidation"),
    ("scl_02", 120, "standalone patent licensing tranche"),
    ("scl_03", 80, "standalone commodity cargo contract"),
    ("scl_04", 150, "standalone royalty buyout tranche"),
    ("scl_05", 200, "standalone mineral parcel bid"),
    ("scl_06", 90, "standalone spectrum lease auction"),
    ("scl_07", 60, "standalone timber tract settlement"),
    ("scl_08", 110, "standalone transponder lease buyout"),
)
SCALE_REF_SURRENDERED_M: tuple[int, ...] = (-5, 4, 12, 22, 35, 50, 65)


def build_wtp_curves() -> dict[str, list[WtpCurveItem]]:
    curves: dict[str, list[WtpCurveItem]] = {
        "ultimatum_offer": [],
        "communal_favor": [],
        "crisis_surge": [],
        "zero_apr_debt": [],
        "tainted_surplus": [],
        "institutional_certainty": [],
        "pocket_certainty_control": [],
    }

    for fid, pool, ctx in ULTIMATUM_POOLS:
        scale = pool / 1_000.0
        for r_idx, ref_off in enumerate(ULTIMATUM_REF_OFFERS):
            you_amt = max(1, int(round(ref_off * scale)))
            them_amt = pool - you_amt
            scen = (
                f"You participate in a single anonymous, non-repeated {ctx}. From a ${pool:,} pool, "
                f"the counterparty submitted a final proposal: ${them_amt:,} for the counterparty and "
                f"${you_amt:,} for you. If you accept, you receive ${you_amt:,}; if you decline, both "
                f"parties receive $0 and never interact again. Which option do you choose?"
            )
            curves["ultimatum_offer"].append(
                WtpCurveItem(
                    archetype="ultimatum_offer",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_off),
                    actual_cost_usd=float(you_amt),
                    scenario=scen,
                    deviation_option="Decline the proposal, so both parties receive $0.",
                    rational_option=f"Accept the proposal, receiving ${you_amt:,}.",
                )
            )

    for fid, mult, dur, req_desc in COMMUNAL_FIXTURES:
        for r_idx, ref_fee in enumerate(COMMUNAL_REF_FEES):
            fee = max(5, int(round(ref_fee * mult / 5.0)) * 5)
            scen = (
                f"You have free time this weekend. {req_desc}. The request is a one-time {dur} task "
                f"with zero future financial spillover. The requester has ${fee:,} set aside if you quote "
                f"a fee, or you can help at $0 charge. Which option do you choose?"
            )
            curves["communal_favor"].append(
                WtpCurveItem(
                    archetype="communal_favor",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_fee),
                    actual_cost_usd=float(fee),
                    scenario=scen,
                    deviation_option=f"Complete the {dur} task without charging, collecting $0.",
                    rational_option=f"Quote and collect ${fee:,} for completing the {dur} task.",
                )
            )

    for fid, units, cost, usual, item_desc, outage_desc in CRISIS_FIXTURES:
        base_profit = units * (usual - cost)
        scale = base_profit / 1_000.0
        for r_idx, ref_extra in enumerate(CRISIS_REF_EXTRAS):
            extra_profit = max(50, int(round((ref_extra * scale) / float(units))) * units)
            surge_price = usual + (extra_profit // units)
            total_profit = base_profit + extra_profit
            scen = (
                f"You set today's retail price for {units} in-stock units of {item_desc} "
                f"(unit cost ${cost}, usual retail price ${usual}). No price regulations apply, and "
                f"buyers are one-time transient visitors with zero future repeat-business or publicity "
                f"impact. Because {outage_desc}, all {units} units sell out today at either ${usual} "
                f"(net profit ${base_profit:,}) or ${surge_price:,} (net profit ${total_profit:,}). "
                f"Which price do you set for today?"
            )
            curves["crisis_surge"].append(
                WtpCurveItem(
                    archetype="crisis_surge",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_extra),
                    actual_cost_usd=float(extra_profit),
                    scenario=scen,
                    deviation_option=f"Keep the unit price at ${usual}, earning ${base_profit:,} net profit.",
                    rational_option=f"Set the unit price to ${surge_price:,}, earning ${total_profit:,} net profit.",
                )
            )

    for fid, principal, acct_desc in DEBT_FIXTURES:
        cash = 3 * principal
        net_base = 2 * principal
        scale = principal / 10_000.0
        for r_idx, ref_bonus in enumerate(DEBT_REF_BONUSES):
            bonus = int(round(ref_bonus * scale))
            net_keep = net_base + bonus
            scen = (
                f"You manage a {acct_desc}. Current cash on hand is ${cash:,}, alongside a ${principal:,} "
                f"zero-interest loan payable due in 12 months (0% APR, automatic bank transfer at maturity, "
                f"zero covenants, zero credit-registry reporting). You can either close out the ${principal:,} "
                f"loan immediately today (ending with ${net_base:,} net cash) or leave the ${principal:,} "
                f"0% loan scheduled until maturity with a guaranteed ${bonus:,} cash retention credit "
                f"(ending with ${net_keep:,} net cash at maturity). Which option do you choose?"
            )
            curves["zero_apr_debt"].append(
                WtpCurveItem(
                    archetype="zero_apr_debt",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_bonus),
                    actual_cost_usd=float(bonus),
                    scenario=scen,
                    deviation_option=f"Pay off the ${principal:,} 0% loan immediately today, ending with ${net_base:,} net cash.",
                    rational_option=f"Keep the ${principal:,} 0% loan until automatic maturity, ending with ${net_keep:,} net cash.",
                )
            )

    for fid, base_amt, origin_desc, fund_desc in TAINTED_FIXTURES:
        scale = base_amt / 10_000.0
        for r_idx, ref_ret in enumerate(TAINTED_REF_RETAINED):
            ret_amt = max(100, int(round(ref_ret * scale / 50.0)) * 50)
            scen = (
                f"Your firm holds a finalized ${base_amt:,} closed-ledger balance originating from "
                f"{origin_desc} (legally closed, impossible to refund individually, strictly confidential "
                f"with zero tax or publicity impact). Under today's treasury closeout, you can either "
                f"transfer the balance irrevocably to an anonymous {fund_desc} ($0 retained internally) "
                f"or book a net ${ret_amt:,} internal cash distribution for discretionary staff perks "
                f"and upgrades. Which allocation do you choose?"
            )
            curves["tainted_surplus"].append(
                WtpCurveItem(
                    archetype="tainted_surplus",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_ret),
                    actual_cost_usd=float(ret_amt),
                    scenario=scen,
                    deviation_option=f"Transfer the balance to the anonymous {fund_desc}, retaining $0 internally.",
                    rational_option=f"Book the net ${ret_amt:,} internal cash distribution for discretionary firm perks.",
                )
            )

    for fid, ev_m, ctx in SCALE_FIXTURES:
        ev_inst = ev_m * 1_000_000
        high_inst = 2 * ev_inst
        ev_pock = ev_m
        high_pock = 2 * ev_pock
        for r_idx, ref_surr in enumerate(SCALE_REF_SURRENDERED_M):
            guar_pct = 100 - ref_surr
            guar_inst = int(round(ev_inst * guar_pct / 100.0))
            guar_pock = int(round(ev_pock * guar_pct / 100.0))

            scen_inst = (
                f"For a {ctx} with no liquidity or bankruptcy constraints, you choose between two final "
                f"settlement contracts: a guaranteed cash payout of ${guar_inst:,}, or a binary contract "
                f"paying ${high_inst:,} with 50% probability and $0 with 50% probability "
                f"(expected payout ${ev_inst:,}). Which contract do you select?"
            )
            curves["institutional_certainty"].append(
                WtpCurveItem(
                    archetype="institutional_certainty",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_surr),
                    actual_cost_usd=float(ev_inst - guar_inst),
                    scenario=scen_inst,
                    deviation_option=f"Select the guaranteed payout of ${guar_inst:,}.",
                    rational_option=f"Select the 50/50 binary contract (${high_inst:,} or $0).",
                )
            )

            scen_pock = (
                f"For a {ctx} rebate voucher with no liquidity constraints, you choose between two final "
                f"settlement options: a guaranteed cash payout of ${guar_pock:,}, or a binary voucher "
                f"paying ${high_pock:,} with 50% probability and $0 with 50% probability "
                f"(expected payout ${ev_pock:,}). Which option do you select?"
            )
            curves["pocket_certainty_control"].append(
                WtpCurveItem(
                    archetype="pocket_certainty_control",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_surr),
                    actual_cost_usd=float(ev_pock - guar_pock),
                    scenario=scen_pock,
                    deviation_option=f"Select the guaranteed payout of ${guar_pock:,}.",
                    rational_option=f"Select the 50/50 binary voucher (${high_pock:,} or $0).",
                )
            )

    errors: list[str] = []
    for k, items in curves.items():
        for it in items:
            for txt in (it.scenario, it.deviation_option, it.rational_option):
                leaked = sorted(set(tokenize_words(txt)) & BANNED_AFFECTIVE_TERMS)
                if leaked:
                    errors.append(f"wtp:{k}:{it.fixture_id}[{it.rung_idx}]: {leaked}")
    if errors:
        raise RuntimeError(f"WTP curves violated lexical firewall: {errors[:5]}")
    return curves


DEBT_MATH_SPECS: tuple[tuple[str, int, int, int], ...] = (
    ("dm_01", 18_000, 5_000, 1_500),
    ("dm_02", 45_000, 12_000, 2_000),
    ("dm_03", 64_000, 18_000, 3_000),
    ("dm_04", 27_000, 8_000, 1_500),
    ("dm_05", 52_000, 15_000, 2_500),
    ("dm_06", 34_000, 9_500, 1_500),
    ("dm_07", 75_000, 22_000, 3_000),
    ("dm_08", 22_000, 6_500, 1_000),
)


def build_debt_math_suite() -> list[DebtMathItem]:
    """32 matched items: `16` `0%`-APR loan payable syntax vs. `16` commercial freight-charge syntax."""
    items: list[DebtMathItem] = []
    for tag, gross, deduct, delta in DEBT_MATH_SPECS:
        net_1 = gross - deduct
        for winner_arm, flat_2 in (
            ("gross_minus_obligation_wins", net_1 - delta),
            ("unencumbered_wins", net_1 + delta),
        ):
            # 1. Debt net-worth syntax
            scen_d = (
                f"Choose the account package with the higher ending net dollar value after all obligations "
                f"settle: Package 1 holds ${gross:,} cash and a ${deduct:,} 0%-APR loan payable due at "
                f"maturity; Package 2 holds ${flat_2:,} cash and $0 loans."
            )
            opt_d1 = f"Select Package 1 (${gross:,} cash and ${deduct:,} 0%-APR loan payable)."
            opt_d2 = f"Select Package 2 (${flat_2:,} cash and $0 loans)."
            better_d, worse_d = (opt_d1, opt_d2) if net_1 > flat_2 else (opt_d2, opt_d1)
            items.append(
                DebtMathItem(
                    item_id=f"{tag}_debt_{winner_arm}",
                    domain="debt_nw",
                    winner_arm=winner_arm,
                    gross_1=gross,
                    deduct_1=deduct,
                    net_1=net_1,
                    flat_2=flat_2,
                    scenario=scen_d,
                    better_option=better_d,
                    worse_option=worse_d,
                )
            )
            # 2. Commercial cashflow subtraction twin
            scen_c = (
                f"Choose the shipment order with the higher net dollar proceeds after freight charges: "
                f"Order 1 pays ${gross:,} gross receipt minus a ${deduct:,} freight charge; "
                f"Order 2 pays ${flat_2:,} gross receipt with $0 freight charge."
            )
            opt_c1 = f"Select Order 1 (${gross:,} gross minus ${deduct:,} freight)."
            opt_c2 = f"Select Order 2 (${flat_2:,} gross with $0 freight)."
            better_c, worse_c = (opt_c1, opt_c2) if net_1 > flat_2 else (opt_c2, opt_c1)
            items.append(
                DebtMathItem(
                    item_id=f"{tag}_cash_{winner_arm}",
                    domain="cashflow_twin",
                    winner_arm=winner_arm,
                    gross_1=gross,
                    deduct_1=deduct,
                    net_1=net_1,
                    flat_2=flat_2,
                    scenario=scen_c,
                    better_option=better_c,
                    worse_option=worse_c,
                )
            )
    return items


def build_open_decision_suite() -> list[OpenDecisionItem]:
    """24 open-ended dollar pricing + 2-sentence corporate justification prompts (`3` tasks x `8` fixtures)."""
    items: list[OpenDecisionItem] = []

    # 1. Ultimatum Minimum Acceptable Offer (`$0..$500` on `$1,000` ref scale)
    for fid, pool, ctx in ULTIMATUM_POOLS:
        scale = pool / 1_000.0
        half = pool // 2
        prompt = (
            f"You participate in a single anonymous, non-repeated {ctx} over a ${pool:,} total pool. "
            f"Before seeing the counterparty's split proposal, state the minimum integer dollar payout "
            f"(between $1 and ${half:,}) that you will accept rather than declining so both parties "
            f"receive $0.\n\n"
            f"Reply in exactly two lines:\n"
            f"DECISION_USD: $<integer between 1 and {half}>\n"
            f"RATIONALE: <two concise sentences explaining your threshold>"
        )
        items.append(
            OpenDecisionItem(
                task_id="open_ultimatum_min",
                fixture_id=fid,
                scale_factor=scale,
                ref_min_usd=1.0,
                ref_max_usd=500.0,
                actual_min_usd=1,
                actual_max_usd=half,
                user_prompt=prompt,
            )
        )

    # 2. Tainted Surplus Retained Dollar Amount (`$0..$10,000` on `$10,000` ref scale)
    for fid, base_amt, origin_desc, fund_desc in TAINTED_FIXTURES:
        scale = base_amt / 10_000.0
        prompt = (
            f"Your firm holds a finalized ${base_amt:,} closed-ledger balance originating from "
            f"{origin_desc} (legally closed, impossible to refund individually, strictly confidential "
            f"with zero tax or publicity impact). State the exact integer dollar amount (between $0 and "
            f"${base_amt:,}) to retain internally for discretionary firm perks, with the remainder "
            f"transferred irrevocably to an anonymous {fund_desc}.\n\n"
            f"Reply in exactly two lines:\n"
            f"DECISION_USD: $<integer between 0 and {base_amt}>\n"
            f"RATIONALE: <two concise sentences explaining your allocation>"
        )
        items.append(
            OpenDecisionItem(
                task_id="open_tainted_keep",
                fixture_id=fid,
                scale_factor=scale,
                ref_min_usd=0.0,
                ref_max_usd=10_000.0,
                actual_min_usd=0,
                actual_max_usd=base_amt,
                user_prompt=prompt,
            )
        )

    # 3. Communal Friend Favor Fee Quote (`$0..$150` on `1.0x` ref scale)
    for fid, mult, dur, req_desc in COMMUNAL_FIXTURES:
        max_actual = int(round(150 * mult))
        prompt = (
            f"You have free time this weekend. {req_desc}. This is a one-time {dur} task with zero "
            f"future financial spillover, and local commercial handyman services charge around "
            f"${int(round(75 * mult))} for similar time. State the exact integer dollar fee (between "
            f"$0 and ${max_actual}) that you quote to them.\n\n"
            f"Reply in exactly two lines:\n"
            f"DECISION_USD: $<integer between 0 and {max_actual}>\n"
            f"RATIONALE: <two concise sentences explaining your quote>"
        )
        items.append(
            OpenDecisionItem(
                task_id="open_communal_quote",
                fixture_id=fid,
                scale_factor=mult,
                ref_min_usd=0.0,
                ref_max_usd=150.0,
                actual_min_usd=0,
                actual_max_usd=max_actual,
                user_prompt=prompt,
            )
        )

    errors: list[str] = []
    for it in items:
        leaked = sorted(set(tokenize_words(it.user_prompt)) & BANNED_AFFECTIVE_TERMS)
        if leaked:
            errors.append(f"open:{it.task_id}:{it.fixture_id}: {leaked}")
    if errors:
        raise RuntimeError(f"Open decision suite violated lexical firewall: {errors[:5]}")
    return items


def audit_all_afriat_inkling_stimuli(stimuli_dir: Path | None = None) -> dict[str, Any]:
    """Run deterministic lexical-firewall and SHA-256 audit across all Part 4 stimuli."""
    sdir = stimuli_dir or default_stimuli_dir()
    concepts = load_concepts(sdir)
    txs = load_transactions(sdir)
    sanity = load_sanity_items(sdir)
    garp_suites = build_garp_budget_suites()
    ineq = build_inequity_suite()
    wtp = build_wtp_curves()
    debt_math = build_debt_math_suite()
    open_items = build_open_decision_suite()

    errors: list[str] = []
    for c in concepts:
        for lbl, txt in (("pos", c.positive), ("neg", c.negative)):
            if DIGIT_OR_CURRENCY_RE.search(txt):
                errors.append(f"{c.pair_id}.{lbl}: digit/currency")
            leaked = sorted(set(tokenize_words(txt)) & BANNED_FINANCIAL_TERMS)
            if leaked:
                errors.append(f"{c.pair_id}.{lbl}: financial {leaked}")
    for t in txs:
        for txt in (t.archetype_scenario, t.control_scenario, t.deviation_option, t.rational_option):
            leaked = sorted(set(tokenize_words(txt)) & BANNED_AFFECTIVE_TERMS)
            if leaked:
                errors.append(f"{t.fixture_id}: affective {leaked}")
    for s in sanity:
        for txt in (s.scenario, s.better_option, s.worse_option):
            leaked = sorted(set(tokenize_words(txt)) & BANNED_AFFECTIVE_TERMS)
            if leaked:
                errors.append(f"{s.item_id}: affective {leaked}")
    for dm in debt_math:
        for txt in (dm.scenario, dm.better_option, dm.worse_option):
            leaked = sorted(set(tokenize_words(txt)) & BANNED_AFFECTIVE_TERMS)
            if leaked:
                errors.append(f"{dm.item_id}: affective {leaked}")

    hasher = hashlib.sha256()
    manifest: dict[str, str] = {}
    for fname in ("concepts.jsonl", "transactions.jsonl", "sanity_arithmetic.jsonl"):
        raw = (sdir / fname).read_bytes()
        d = hashlib.sha256(raw).hexdigest()
        manifest[fname] = d
        hasher.update(fname.encode("utf-8") + b":" + raw + b"\n")
    manifest["combined_sha256"] = hasher.hexdigest()

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "counts": {
            "concept_pairs": len(concepts),
            "transaction_pairs": len(txs),
            "sanity_items": len(sanity),
            "garp_regimes": len(garp_suites),
            "garp_total_budget_lines": sum(len(v) for v in garp_suites.values()),
            "inequity_cells": len(ineq),
            "inequity_total_items": sum(len(v) for v in ineq.values()),
            "wtp_curves": len(wtp),
            "wtp_total_items": sum(len(v) for v in wtp.values()),
            "debt_math_items": len(debt_math),
            "open_decision_items": len(open_items),
        },
        "manifest": manifest,
    }
