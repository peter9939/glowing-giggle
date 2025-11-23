# smart_patterns.py
# Ultra-smart dynamic categorization with full parsing and auto-learning

import re
import pandas as pd
import os
from rapidfuzz import process, fuzz

USER_CAT_FILE = "categorization.csv"

# ---------------- Load / Initialize User-edited categories ----------------
def load_user_categories():
    if os.path.exists(USER_CAT_FILE):
        try:
            df = pd.read_csv(USER_CAT_FILE)
            return df
        except Exception:
            return pd.DataFrame(columns=["Transaction Description", "Main Category", "Subcategory"])
    else:
        df = pd.DataFrame(columns=["Transaction Description", "Main Category", "Subcategory"])
        df.to_csv(USER_CAT_FILE, index=False)
        return df

user_categories_df = load_user_categories()

# ---------------- Save new user category ----------------
def save_new_category(desc, main_cat, sub_cat):
    global user_categories_df
    desc_lower = desc.lower()
    if desc_lower not in user_categories_df['Transaction Description'].str.lower().values:
        new_row = pd.DataFrame([{
            "Transaction Description": desc,
            "Main Category": main_cat,
            "Subcategory": sub_cat
        }])
        user_categories_df = pd.concat([user_categories_df, new_row], ignore_index=True)
        user_categories_df.to_csv(USER_CAT_FILE, index=False)

# ---------------- Extract amount and currency ----------------
def extract_amount_currency(desc):
    amount_match = re.search(r"(\d{1,3}(?:[,\d]{0,3})*(?:\.\d+)?)", desc)
    currency_match = re.search(r"\b(usd|pkr|eur|gbp|inr|₹|\$|€|£)\b", desc)
    amount = amount_match.group(1).replace(',', '') if amount_match else None
    currency = currency_match.group(1).upper() if currency_match else ""
    return amount, currency

# ---------------- Smart dynamic categorization ----------------
def smart_dynamic_categorization(description):
    desc = str(description).lower().strip()

    # ---------------- User-edited categories first ----------------
    if not user_categories_df.empty:
        user_dict = {str(row['Transaction Description']).lower(): (row['Main Category'], row['Subcategory'])
                     for _, row in user_categories_df.iterrows()}
        result = process.extractOne(desc, user_dict.keys(), scorer=fuzz.token_sort_ratio)
        if result:
            match_str, score, _ = result
            if score >= 85:
                main_cat, sub_cat = user_dict[match_str]
                return main_cat, sub_cat, True, score

    # ---------------- Extract amounts and currency ----------------
    amount, currency = extract_amount_currency(desc)
    amount_str = f" ({amount}{currency})" if amount else ""

    # ---------------- Generic Credit / Debit Detection ----------------
    credit_regex = [
        r"\bcredit from\b", r"\bcredited\b", r"\bdeposit from\b", r"\bpayment from\b",
        r"\btransfer from\b", r"\breceived from\b"
    ]
    debit_regex = [
        r"\bdebit to\b", r"\bpaid to\b", r"\bwithdrawn to\b", r"\btransfer to\b", r"\bsent to\b"
    ]

    if any(re.search(pat, desc) for pat in credit_regex):
        match = re.search(r"(?:credit from|credited by|deposit from|payment from|transfer from|received from)\s+([\w\s]+)", desc)
        payer = match.group(1).title() if match else "Client"
        sub_cat = f"Credit from {payer}{amount_str}"
        return "Revenue", sub_cat, True, 95

    if any(re.search(pat, desc) for pat in debit_regex):
        match = re.search(r"(?:debit to|paid to|withdrawn to|transfer to|sent to)\s+([\w\s]+)", desc)
        payee = match.group(1).title() if match else "Payee"
        sub_cat = f"Debit to {payee}{amount_str}"
        return "Expense", sub_cat, True, 95

    # ---------------- Revenue Patterns ----------------
    revenue_patterns = [
        r"ach (deposit|credit)", r"paypal (deposit|payment|transfer)", r"stripe payout",
        r"client payment", r"cash sale", r"tuition", r"student fee",
        r"invoice (paid|payment)", r"payment (from|received|credited|sent)",
        r"transfer (from|received|to)", r"grant received", r"rental income",
        r"subscription income", r"commission income", r"interest income"
    ]
    if any(re.search(pat, desc) for pat in revenue_patterns):
        sub_cat = "Client Payment"
        if any(k in desc for k in ["consulting", "software dev", "freelance", "upwork", "fiverr"]):
            sub_cat = "Consulting / Freelance"
        elif any(k in desc for k in ["real estate", "property management"]):
            sub_cat = "Real Estate Income"
        elif any(k in desc for k in ["tuition", "school", "student"]):
            sub_cat = "Education / School"
        elif any(k in desc for k in ["subscription", "membership", "saas"]):
            sub_cat = "Subscription / Membership"
        elif any(k in desc for k in ["commission", "brokerage", "agent"]):
            sub_cat = "Commission Income"
        sub_cat += amount_str
        return "Revenue", sub_cat, True, 95

    # ---------------- Expense Patterns ----------------
    expense_keywords = {
        "Payroll / Salaries": ["bonus", "salary", "employee", "payroll", "stipend", "wages"],
        "Utilities / Rent": ["electric", "water", "internet", "ptcl", "iesco", "utility",
                             "phone bill", "gas", "fuel", "pso", "psp", "rent", "office cleaning"],
        "Insurance": ["insurance", "prem", "policy", "health insurance", "vehicle insurance"],
        "Office / Supplies / Equipment": ["office depot", "staples", "supplies", "equipment",
                                         "amazon purchase", "amzn office", "amzn marketplace",
                                         "home depot", "best buy", "lowe's"],
        "Food / Beverage": ["starbucks", "mcdonald", "subway", "panera", "naheed mart",
                            "coffee", "snacks", "foodpanda", "restaurant"],
        "SaaS / Digital Tools": ["dropbox", "google workspace", "zoho", "canva", "adobe",
                                 "zoom.us", "wordpress plugin", "api tool", "software license",
                                 "hostgator", "web hosting", "cloud service"],
        "Marketing / Advertising": ["facebook ads", "linkedin ads", "ad agency", "web dev payment", "marketing"],
        "Travel / Transport": ["uber", "trip", "taxi", "hotel", "fuel", "bus", "train", "airlines"],
        "Freelance / Contractors": ["upwork", "fiverr", "escrow", "contractor", "freelancer"],
        "Retail / POS": ["pos", "store", "supercenter", "purchase", "retail", "square", "shopify", "woocommerce", "etsy"],
        "Web3 / Crypto": ["crypto", "bitcoin", "ethereum", "nft", "staking", "wallet"],
        "Education / School": ["school", "tuition", "books", "student", "college", "training"],
        "Subscription / Membership": ["netflix", "spotify", "subscription", "membership", "saas"],
        "Fees / Service Charges": ["tcs delivery charge", "bank fee", "credit card cashback",
                                   "paypal fee", "processing fee", "service charge", "escrow fee"],
        "Franchise / Licensing / Royalties": ["franchise fee", "royalty", "license fee", "patent fee"],
        "Customs / Duties": ["customs", "import duty", "tariff"]
    }

    for cat, keywords in expense_keywords.items():
        if any(kw in desc for kw in keywords):
            sub_cat = f"{cat}{amount_str}"
            return "Expense", sub_cat, True, 95

    # ---------------- Refunds / Chargebacks ----------------
    if any(kw in desc for kw in ["refund", "chargeback", "reversal", "rebate"]):
        return "Expense", f"Refund / Reversal{amount_str}", True, 95

    # ---------------- Rounded / Uncategorized Payments ----------------
    if amount:
        return "Expense", f"Rounded Payment / Bill{amount_str}", True, 80

    # ---------------- Fallback ----------------
    if any(k in desc for k in ["pay", "deposit", "transfer", "receive", "payment", "credited"]):
        main_cat, sub_cat = "Revenue", f"Client Payment{amount_str}"
    else:
        main_cat, sub_cat = "Expense", f"Miscellaneous{amount_str}"

    save_new_category(desc, main_cat, sub_cat)
    return main_cat, sub_cat, False, 50
