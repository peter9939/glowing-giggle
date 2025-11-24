# smart_patterns_advanced.py
# Ultra-accurate, ERP-ready, cross-industry categorization (best-of-both worlds)
# Keep function name: smart_dynamic_categorization
# Dependencies: pandas, rapidfuzz

import re
import os
import pandas as pd
from rapidfuzz import process, fuzz

USER_CAT_FILE = "categorization.csv"

# ----------------------------- Utilities --------------------------------- #
def safe_lower(x):
    return "" if x is None else str(x).lower()

def normalize_text(s):
    """Normalize description: lowercase, replace smart quotes, remove excessive punctuation, collapse spaces."""
    s = safe_lower(s)
    s = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", s)              # smart quotes -> '
    # keep currency symbols, percent, dot, dash, slash and alphanum
    s = re.sub(r"[^a-z0-9\$\€\£\₹\.\-\/\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_amount_from_desc(desc):
    """Extract first numeric-looking amount from a description. Returns float or None."""
    if not desc:
        return None
    m = re.search(r"(-?\d{1,3}(?:[,\d]{0,3})*(?:\.\d+)?)", desc)
    if not m:
        return None
    txt = m.group(1).replace(",", "")
    try:
        return float(txt)
    except:
        return None

# -------------------------- User categories (persisted) ------------------- #
def load_user_categories():
    """Load CSV with user-edited mapping. If missing, create blank file."""
    if os.path.exists(USER_CAT_FILE):
        try:
            df = pd.read_csv(USER_CAT_FILE)
            if 'Transaction Description' not in df.columns:
                # malformed file -> replace with proper columns
                return pd.DataFrame(columns=["Transaction Description", "Main Category", "Subcategory"])
            return df
        except Exception:
            return pd.DataFrame(columns=["Transaction Description", "Main Category", "Subcategory"])
    else:
        df = pd.DataFrame(columns=["Transaction Description", "Main Category", "Subcategory"])
        try:
            df.to_csv(USER_CAT_FILE, index=False)
        except Exception:
            pass
        return df

user_categories_df = load_user_categories()

def save_new_category(desc, main_cat, sub_cat):
    """Persist a new user mapping if not already there (case-insensitive)."""
    global user_categories_df
    if not desc or not isinstance(desc, str):
        return
    desc_strip = desc.strip()
    if user_categories_df.empty:
        exists = False
    else:
        exists = any(user_categories_df['Transaction Description'].str.lower() == desc_strip.lower())
    if not exists:
        new_row = pd.DataFrame([{
            "Transaction Description": desc_strip,
            "Main Category": main_cat,
            "Subcategory": sub_cat
        }])
        user_categories_df = pd.concat([user_categories_df, new_row], ignore_index=True)
        try:
            user_categories_df.to_csv(USER_CAT_FILE, index=False)
        except Exception:
            pass

# --------------------------- Core token dictionaries ---------------------- #
# These lists are intentionally broad — extend them as needed.
processors = [
    "stripe", "stripe payout", "stripe settlement", "stripe transfer", "paypal", "paypal payout",
    "square", "shopify", "shopify payout", "razorpay", "payoneer", "paytm", "flutterwave",
    "braintree", "adyen", "klarna", "wise", "transferwise", "mollie", "payu"
]

saas_keywords = [
    "subscription", "saas", "plan", "license", "monthly", "annual", "renewal", "pro plan", "premium",
    "software license", "cloud service", "hosting", "web hosting", "domain", "cdn", "api", "tool subscription",
    "subscription payment", "subscription fee"
]

ecommerce = [
    "amazon", "amazon mktplace", "amazon marketplace", "amzn", "ebay", "etsy", "daraz", "lazada", "flipkart",
    "shopify", "woocommerce"
]

bank_tokens = [
    "bank transfer", "ach deposit", "direct deposit", "wire transfer", "credited", "deposit", "transfer from",
    "transfer to", "withdrawal", "atm", "atm withdrawal", "bank fee", "bank charge", "bank charges", "bank", "eft"
]

refund_tokens = ["refund", "chargeback", "reversal", "rebate", "refund processed"]

pos_tokens = ["pos", "point of sale", "store", "merchant", "retailer", "supercenter", "checkout"]

food_tokens = ["starbucks", "mcdonald", "kfc", "pizza", "restaurant", "foodpanda", "ubereats", "food delivery", "coffee"]

travel_tokens = ["uber", "careem", "taxi", "flight", "airlines", "hotel", "airbnb", "booking.com", "expedia", "train", "bus"]

marketing_tokens = ["facebook ads", "meta ads", "instagram ads", "google ads", "adwords", "linkedin ads", "ad campaign", "ad agency"]

payroll_tokens = ["payroll", "salary", "wages", "payroll deposit", "direct dep", "payroll direct dep", "payroll dr"]

utility_tokens = ["electricity", "water", "internet", "ptcl", "iesco", "k-electric", "k electric", "gas bill", "phone bill"]

crypto_tokens = ["bitcoin", "btc", "ethereum", "eth", "crypto", "coinbase", "binance", "wallet"]

tax_tokens = ["vat", "gst", "tax", "tax refund", "withholding", "fbr", "customs", "import duty"]

# A compact vendor exact map for high-confidence exact matches (vendor substring -> (Main, Sub))
vendor_exact_map = {
    "adobe acropro": ("Expense", "Software Subscription"),
    "adobe": ("Expense", "Software Subscription"),
    "canva pro": ("Expense", "Design / SaaS"),
    "google workspace": ("Expense", "SaaS / Productivity"),
    "hostgator": ("Expense", "Hosting / IT Services"),
    "hostinger": ("Expense", "Hosting / IT Services"),
    "godaddy": ("Expense", "Hosting / Domain"),
    "dropbox": ("Expense", "SaaS / Storage"),
    "zoho": ("Expense", "SaaS / CRM"),
    "paypal": ("Revenue", "Online Payments"),
    "stripe": ("Revenue", "Online Payments"),
    "square": ("Revenue", "Online Payments"),
    "shopify": ("Revenue", "E-commerce Sales"),
    "amazon mktplace": ("Revenue", "E-commerce Sales"),
    "amazon marketplace": ("Revenue", "E-commerce Sales"),
    "amzn": ("Expense", "E-commerce Purchases"),
    "upwork": ("Expense", "Freelance / Contractors"),
    "fiverr": ("Expense", "Freelance / Contractors"),
    "starbucks": ("Expense", "Meals & Entertainment"),
    "uber": ("Expense", "Travel / Transport"),
    "ptcl": ("Expense", "Utilities / Internet"),
    "iesco": ("Expense", "Utilities / Electricity"),
    "payoneer": ("Revenue", "Online Payments"),
    "easypaisa": ("Revenue", "Mobile Wallet"),
    "jazzcash": ("Revenue", "Mobile Wallet"),
    "hostgator web hosting": ("Expense", "Hosting / IT Services"),
    "canva": ("Expense", "Design / SaaS"),
    "stripe fee": ("Expense", "Payment Processor Fees"),
    "paypal fee": ("Expense", "Payment Processor Fees"),
    "card refund": ("Expense", "Refund / Reversal"),
    "cash dep": ("Revenue", "Cash Deposit"),
    "cash deposit": ("Revenue", "Cash Deposit"),
    "bank fee": ("Expense", "Bank Charges"),
    "bank charges": ("Expense", "Bank Charges"),
}

# normalize vendor_exact_map keys to lower
vendor_exact_map = {k.lower(): v for k, v in vendor_exact_map.items()}

# -------------------- Helper matchers ----------------------------------- #
def match_vendor_exact(desc):
    """Longest-substring exact match for vendors (high confidence)."""
    # check longer keys first
    for vendor in sorted(vendor_exact_map.keys(), key=lambda x: -len(x)):
        if vendor in desc:
            return vendor_exact_map[vendor], vendor, 100
    return None, None, 0

def fuzzy_user_lookup(desc):
    """Fuzzy-match user-saved descriptions (highest precedence)."""
    if user_categories_df is None or user_categories_df.empty:
        return None, None, 0
    user_map = {str(row['Transaction Description']).lower(): (row['Main Category'], row['Subcategory'])
                for _, row in user_categories_df.iterrows()}
    choices = list(user_map.keys())
    if not choices:
        return None, None, 0
    match = process.extractOne(desc, choices, scorer=fuzz.token_sort_ratio)
    if match:
        matched_str, score, _ = match
        if score >= 85:
            return user_map[matched_str], matched_str, int(score)
    return None, None, 0

def any_token(desc, tokens):
    """Check if any token exists as substring in desc."""
    for t in tokens:
        if t in desc:
            return True
    return False

# ----------------- The main function (kept name) ------------------------- #
def smart_dynamic_categorization(description):
    """
    Input: description (string) - raw transaction description (may include amounts)
    Output: (main_category, subcategory, matched_user_override_bool, confidence_score)
    """

    # Normalize
    raw = "" if description is None else str(description)
    desc = normalize_text(raw)

    # Extract numeric amount if present (helps determine direction)
    amt = extract_amount_from_desc(desc)  # may be positive or negative
    # If amount is None we won't assume based on sign unless forced

    # 1) User overrides (highest priority)
    user_match, user_key, user_score = fuzzy_user_lookup(desc)
    if user_match:
        main, sub = user_match
        return main, sub, True, int(user_score)

    # 2) Exact vendor mapping (very high confidence)
    vendor_map, vendor_key, vendor_score = match_vendor_exact(desc)
    if vendor_map:
        main, sub = vendor_map
        return main, sub, True, vendor_score

    # 3) Refund / Chargeback detection (treat as expense/refund)
    if any_token(desc, refund_tokens):
        # If refund has positive amt -> it's actually customer refund to you? treat as Expense (refund out)
        # Many bank feed tools mark refunds as positive amounts (refund to customer) or negative (refund from marketplace)
        return "Expense", "Refund / Chargeback", True, 98

    # 4) Processors / aggregators: treat payouts/deposits as Revenue by default
    for p in processors:
        if p in desc:
            # Recognize 'fee' mentions -> processor fee (expense)
            if any(k in desc for k in ["fee", "processing fee", "stripe fee", "paypal fee", "charge"]):
                return "Expense", f"{p.title()} Fees", True, 96
            # Recognize 'refund' around processor -> refund
            if any(k in desc for k in refund_tokens):
                return "Expense", f"{p.title()} Refund", True, 96
            # Recognize words that indicate deposit/payout/settlement -> Revenue
            if any(k in desc for k in ["payout", "payouts", "deposit", "settlement", "transfer", "paid out", "credit", "payment received", "payment received", "payment"]):
                return "Revenue", f"Payments via {p.title()}", True, 97
            # Fallback: treat processor mentions as Revenue (most common/auto rule)
            return "Revenue", f"Payments via {p.title()}", True, 90

    # 5) Bank-specific logic
    if any_token(desc, bank_tokens):
        # credits/deposits -> Revenue
        if any(k in desc for k in ["credit", "credited", "deposit", "direct deposit", "credited by", "received"]):
            return "Revenue", "Bank Deposit / Transfer", True, 95
        # charges / withdrawal / paid to -> Expense
        if any(k in desc for k in ["fee", "bank fee", "withdrawal", "atm", "paid to", "debit", "payment to", "charge"]):
            return "Expense", "Bank Fees / Payment", True, 95
        # sign-aware fallback
        if amt is not None:
            if amt > 0:
                return "Revenue", "Bank Transfer / Deposit", True, 92
            else:
                return "Expense", "Bank Transfer / Payment", True, 92
        return "Expense", "Bank Transaction", True, 85

    # 6) SaaS / hosting / subscription detection (vendor or keyword)
    if any_token(desc, saas_keywords) or any_token(desc, saas_keywords) or any_token(desc, list(map(lambda v: v.lower(), saas_keywords))):
        # If string explicitly indicates income -> revenue
        if any(k in desc for k in ["subscription income", "subscription payment received", "recurring payment received", "membership payment"]):
            return "Revenue", "Subscription Income", True, 94
        # otherwise likely expense (vendor subscription)
        return "Expense", "SaaS / Software Subscription", True, 94

    # 7) Utilities / telecom / internet
    if any_token(desc, utility_tokens):
        return "Expense", "Utilities", True, 96

    # 8) Payroll / salary
    if any_token(desc, payroll_tokens):
        return "Expense", "Payroll & Benefits", True, 97

    # 9) Food / beverage
    if any_token(desc, food_tokens):
        return "Expense", "Meals & Entertainment", True, 96

    # 10) Travel / transport
    if any_token(desc, travel_tokens):
        return "Expense", "Travel / Transport", True, 96

    # 11) Marketing / advertising
    if any_token(desc, marketing_tokens):
        return "Expense", "Marketing / Advertising", True, 96

    # 12) POS / retail purchases
    if any_token(desc, pos_tokens) or any_token(desc, ecommerce):
        # Distinguish marketplace settlements (marketplace->revenue) vs purchases
        if any(k in desc for k in ["marketplace", "mktplace", "settlement", "payout", "payment received", "payment to seller"]):
            return "Revenue", "E-commerce Sales", True, 95
        # Amazon has many forms — treat AMZN/Amazon purchases as Expense unless it says marketplace payout/settlement
        if any(k in desc for k in ["amzn", "amazon purchase", "amazon order", "amazon prime", "amazon.co.uk"]):
            return "Expense", "E-commerce Purchases", True, 94
        # default POS -> expense
        return "Expense", "Retail / POS Purchase", True, 93

    # 13) Crypto / wallets
    if any_token(desc, crypto_tokens):
        # incoming deposit? use sign or 'received' keyword
        if any(k in desc for k in ["received", "deposit", "credited"]):
            return "Revenue", "Crypto / Wallet Inflow", True, 90
        if any(k in desc for k in ["withdrawal", "sent", "transfer out"]):
            return "Expense", "Crypto / Wallet Outflow", True, 90
        return "Expense", "Crypto / Wallet", True, 80

    # 14) Tax / customs
    if any_token(desc, tax_tokens):
        # tax refund -> revenue, tax paid -> expense
        if any(k in desc for k in ["refund", "tax refund"]):
            return "Revenue", "Tax Refund", True, 92
        return "Expense", "Taxes & Duties", True, 92

    # 15) Refund/chargeback earlier check missed? do again
    if any_token(desc, refund_tokens):
        return "Expense", "Refund / Chargeback", True, 95

    # 16) Generic revenue clues
    revenue_clues = ["invoice payment", "client payment", "payment from", "payment received", "received payment", "deposit received", "sale", "payment - received"]
    if any(k in desc for k in revenue_clues):
        return "Revenue", "Sales Income", True, 90

    # 17) Generic expense clues
    expense_clues = ["paid to", "payment to", "purchase", "bill", "billed", "payment -", "charge", "fee", "payment online"]
    if any(k in desc for k in expense_clues):
        return "Expense", "General Expense", True, 85

    # 18) Sign-aware fallback (if amount present)
    if amt is not None:
        # Positive amount -> inbound -> revenue (default automation)
        if amt > 0:
            return "Revenue", "Bank / Positive Amount", False, 78
        else:
            return "Expense", "Bank / Negative Amount", False, 78

    # 19) Fuzzy-match remaining keywords (lower confidence)
    # Combine revenue and expense keywords into master list for fuzzy check
    master_keywords = []
    for kws in [list(map(str.lower, v)) for v in [saas_keywords, ecommerce, bank_tokens, refund_tokens, pos_tokens, food_tokens, travel_tokens, marketing_tokens, payroll_tokens, utility_tokens, crypto_tokens, tax_tokens]]:
        master_keywords += kws
    # Remove duplicates
    master_keywords = list(dict.fromkeys(master_keywords))
    if master_keywords:
        best = process.extractOne(desc, master_keywords, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= 72:
            kw = best[0]
            # if found in any revenue indicative list
            if kw in revenue_clues:
                return "Revenue", "Sales Income", False, int(best[1])
            if kw in expense_clues or kw in utility_tokens or kw in marketing_tokens or kw in payroll_tokens:
                # map to approximate category based on token membership
                if kw in utility_tokens:
                    return "Expense", "Utilities", False, int(best[1])
                if kw in marketing_tokens:
                    return "Expense", "Marketing / Advertising", False, int(best[1])
                if kw in payroll_tokens:
                    return "Expense", "Payroll & Benefits", False, int(best[1])
                return "Expense", "General Expense", False, int(best[1])

    # 20) Persist numeric unknowns as suggested misc expense (helps auto-learn)
    if extract_amount_from_desc(desc) is not None:
        try:
            save_new_category(raw, "Expense", "Misc Payment / Auto-Suggest")
        except Exception:
            pass
        return "Expense", "Misc Payment / Auto-Suggest", False, 70
     # -------------------- New automated category detection -------------------- #
     additional_expense_tokens = {
    "Equipment / Repairs": ["equipment repair", "machinery repair", "tool repair", "machine maintenance", "equipment maintenance", "repair service"],
    "Office Supplies / Materials": ["supplies", "office supplies", "stationery", "printer paper", "materials", "consumables", "stationary"],
    "Consulting / Freelancers": ["consulting", "consultant", "freelancer", "freelance", "contractor", "outsourced service", "advisor"],
    "Real Estate / Property": ["real estate", "property management", "rent paid", "lease", "property tax", "mortgage"],
    "Insurance": ["insurance", "premium", "policy", "vehicle insurance", "health insurance", "life insurance"],
    "Digital Marketing / Ads": ["digital marketing", "facebook ads", "google ads", "meta ads", "instagram ads", "linkedin ads", "ad campaign", "seo", "sponsorship", "banner printing", "flyer printing"]
     }

    for subcategory, tokens in additional_expense_tokens.items():
    if any_token(desc, tokens):
        return "Expense", subcategory, True, 96


    # 21) Final fallback guess
    main = "Revenue" if any(k in desc for k in ["receive", "received", "credit", "credited", "deposit", "payment from", "payout"]) else "Expense"
    sub = "Client Payment" if main == "Revenue" else "Miscellaneous"
    try:
        save_new_category(raw, main, sub)
    except Exception:
        pass
    return main, sub, False, 60

# ----------------------- Convenience wrapper ----------------------------- #
def categorize_description(description, amount=None):
    """
    Wrapper for external usage. If `amount` provided, embed it into description for sign-aware heuristics.
    Returns a dictionary with Main Category, Subcategory, From User Override (bool), Confidence.
    """
    desc = description if amount is None else f"{description} {amount}"
    main, sub, user_flag, conf = smart_dynamic_categorization(desc)
    return {
        "Main Category": main,
        "Subcategory": sub,
        "From User Override": bool(user_flag),
        "Confidence": int(conf)
    }

# --------------------------- Quick tests -------------------------------- #
if __name__ == "__main__":
    tests = [
        ("ACH DEPOSIT STRIPE 82920", 320),
        ("AMAZON MKTPLACE PMTS AMZN", -89.99),
        ("ACCT TRANSFER CREDIT CLIENT 2201", 1450),
        ("STARBUCKS STORE 4471", -6.5),
        ("ADOBE ACROPRO SUBSCRIPTION", -29.99),
        ("HOSTGATOR WEB HOSTING", -14),
        ("UPWORK ESCROW PAYMENT", -150),
        ("FACEBK ADS 483920", -120),
        ("CASH DEP BR 192", 500),
        ("RENT PMT ONLINE", -1200),
        ("PTCL INTERNET BILL", -55),
        ("STRIPE PAYOUT 19033", 780),
        ("STRIPE FEE - processing", -12.5),
        ("PAYPAL *PAYMENT RECEIVED", 250),
        ("CARD REFUND AMAZON EU", 23.5),
        ("BANK TRANSFER TO SUPPLIER", -350.00),
        ("IBFT CREDIT FROM CLIENT", 1250.00),
        ("EASYPaisa deposit received", 200),
        ("BINANCE WITHDRAWAL", -0.5),
    ]
    for desc, amt in tests:
        res = categorize_description(desc, amt)
        print(f"{desc:40} | {amt:8} | {res['Main Category']:8} | {res['Subcategory'][:30]:30} | conf={res['Confidence']}")
