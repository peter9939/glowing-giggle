# smart_patterns.py
# Ultra-enhanced dynamic categorization with auto-learning
# Handles revenue, expenses, payroll, utilities, e-commerce, SaaS, Web3, schools, marketing, insurance, and more

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

    # ---------------- Revenue Patterns ----------------
    revenue_patterns = [
        r"ach (deposit|credit)", r"paypal (deposit|payment|transfer)", r"stripe payout",
        r"client payment", r"cash sale", r"tuition", r"student fee",
        r"invoice (paid|payment)", r"payment (from|received|credited|sent)",
        r"transfer (from|received|to)", r"grant received", r"rental income",
        r"subscription income", r"commission income", r"interest income"
    ]
    if any(re.search(pat, desc) for pat in revenue_patterns):
        if any(k in desc for k in ["consulting", "software dev", "freelance", "upwork", "fiverr"]):
            return "Revenue", "Consulting / Freelance", True, 95
        elif any(k in desc for k in ["real estate", "property management"]):
            return "Revenue", "Real Estate Income", True, 95
        elif any(k in desc for k in ["tuition", "school", "student"]):
            return "Revenue", "Education / School", True, 95
        elif any(k in desc for k in ["subscription", "membership", "saas"]):
            return "Revenue", "Subscription / Membership", True, 95
        elif any(k in desc for k in ["commission", "brokerage", "agent"]):
            return "Revenue", "Commission Income", True, 95
        else:
            return "Revenue", "Client Payment", True, 90

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
            return "Expense", cat, True, 95

    # ---------------- Rounded / Numeric Payments ----------------
    if re.search(r"[\s,](\d{2,})(\.00)?$", desc):
        return "Expense", "Rounded Payment / Bill", True, 80

    # ---------------- Refunds / Chargebacks ----------------
    if any(kw in desc for kw in ["refund", "chargeback", "reversal", "rebate"]):
        return "Expense", "Refund / Reversal", True, 95

    # ---------------- Uncategorized Fallback ----------------
    if any(k in desc for k in ["pay", "deposit", "transfer", "receive", "payment", "credited"]):
        main_cat, sub_cat = "Revenue", "Client Payment"
    else:
        main_cat, sub_cat = "Expense", "Miscellaneous"

    save_new_category(desc, main_cat, sub_cat)
    return main_cat, sub_cat, False, 50
