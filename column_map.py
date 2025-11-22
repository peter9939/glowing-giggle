COLUMN_MAP = {
    # === Description Fields ===
    'desc': 'Description', 'details': 'Description', 'transaction_desc': 'Description',
    'memo': 'Description', 'narration': 'Description', 'particulars': 'Description',
    'item_description': 'Description', 'payment_for': 'Description', 'purpose': 'Description',
    'notes': 'Description', 'remark': 'Description', 'remarks': 'Description',
    'description_1': 'Description', 'description_2': 'Description',

    # === Amount Fields ===
    'amount': 'Amount', 'amt': 'Amount', 'value': 'Amount', 'transaction_value': 'Amount',
    'payment_amount': 'Amount', 'paid': 'Amount', 'received': 'Amount',
    'balance': 'Amount', 'closing_balance': 'Amount', 'opening_balance': 'Amount',
    'total': 'Amount', 'net_amount': 'Amount', 'gross_amount': 'Amount',
    'subtotal': 'Amount', 'grand_total': 'Amount',

    # === Debit / Credit ===
    'debit': 'Debit', 'dr': 'Debit', 'withdrawal': 'Debit', 'debits': 'Debit',
    'credit': 'Credit', 'cr': 'Credit', 'deposit': 'Credit', 'credits': 'Credit',

    # === Date / Time ===
    'date': 'Date', 'transaction_date': 'Date', 'payment_date': 'Date',
    'posting_date': 'Date', 'entry_date': 'Date', 'created_on': 'Date',
    'value_date': 'Date', 'invoice_date': 'Date', 'bill_date': 'Date',

    'month': 'Month', 'month_name': 'Month', 'month_number': 'Month',
    'year': 'Year', 'yr': 'Year', 'financial_year': 'Year',
    'time': 'Time', 'transaction_time': 'Time',

    # === Accounts / Categories ===
    'account': 'Account Type', 'acct_type': 'Account Type', 'ledger': 'Account Type',
    'category': 'Account Type', 'account_name': 'Account Type', 'cost_center': 'Account Type',
    'department': 'Account Type', 'gl_code': 'Account Type', 'coa': 'Account Type',

    # === Taxes ===
    'tax': 'Tax', 'tax_amount': 'Tax', 'vat': 'Tax', 'gst': 'Tax',
    'sales_tax': 'Tax', 'income_tax': 'Tax', 'tax_code': 'Tax',
    'tax_percent': 'Tax', 'tax_rate': 'Tax',

    # === Payment Method ===
    'payment_method': 'Payment Method', 'mode': 'Payment Method', 'method': 'Payment Method',
    'transaction_mode': 'Payment Method', 'paid_via': 'Payment Method',
    'payment_type': 'Payment Method', 'channel': 'Payment Method',

    # === Reference / Invoice / IDs ===
    'invoice_no': 'Reference', 'invoice_number': 'Reference',
    'transaction_id': 'Reference', 'ref_no': 'Reference', 'voucher_no': 'Reference',
    'bill_no': 'Reference', 'receipt_no': 'Reference', 'utr': 'Reference',
    'reference_number': 'Reference', 'document_no': 'Reference',

    # === Client / Vendor ===
    'client': 'Client/Vendor', 'customer': 'Client/Vendor', 'vendor': 'Client/Vendor',
    'supplier': 'Client/Vendor', 'party': 'Client/Vendor', 'name': 'Client/Vendor',
    'contact': 'Client/Vendor', 'merchant': 'Client/Vendor',

    # === Inventory / Products ===
    'item': 'Inventory', 'product': 'Inventory', 'product_code': 'Inventory',
    'sku': 'Inventory', 'stock': 'Inventory', 'quantity': 'Inventory', 'qty': 'Inventory',
    'unit_price': 'Inventory', 'rate': 'Inventory', 'price': 'Inventory',

    # === Salary / Payroll ===
    'salary': 'Salary', 'payroll': 'Salary', 'employee_payment': 'Salary',
    'wages': 'Salary', 'bonus': 'Salary', 'commission': 'Salary',
    'overtime': 'Salary', 'allowance': 'Salary', 'deduction': 'Salary',

    # === Expense Categories ===
    'expense': 'Expense', 'purchase': 'Expense', 'utility_bill': 'Expense',
    'travel_expense': 'Expense', 'maintenance': 'Expense', 'rent': 'Expense',
    'marketing': 'Expense', 'it_expense': 'Expense', 'training': 'Expense',
    'insurance': 'Expense', 'food': 'Expense', 'vehicle': 'Expense',
    'office_expense': 'Expense', 'repair': 'Expense', 'fuel': 'Expense',
    'tools': 'Expense', 'software': 'Expense', 'supplies': 'Expense',

    # === Income Categories ===
    'income': 'Income', 'revenue': 'Income', 'sales': 'Income', 'service_income': 'Income',
    'client_payment': 'Income', 'subscription': 'Income', 'refund': 'Income',
    'interest': 'Income', 'earning': 'Income', 'deposit_income': 'Income',

    # === Bank / Cash ===
    'bank': 'Bank', 'bank_name': 'Bank', 'account_no': 'Bank', 'account_number': 'Bank',
    'bank_deposit': 'Bank', 'bank_withdrawal': 'Bank', 'iban': 'Bank',
    'cash': 'Cash', 'cash_account': 'Cash', 'wallet': 'Cash',

    # === Assets / Liabilities / Equity ===
    'asset': 'Asset', 'fixed_asset': 'Asset', 'current_asset': 'Asset', 'inventory_asset': 'Asset',
    'liability': 'Liability', 'current_liability': 'Liability', 'long_term_liability': 'Liability',
    'equity': 'Equity', 'owner_equity': 'Equity', 'capital': 'Equity'
}

# === Normalize keys for detection ===
# Use this in your code before renaming columns:
NORMALIZED_COLUMN_MAP = {k.strip().lower(): v for k, v in COLUMN_MAP.items()}
