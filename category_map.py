CATEGORY_PATTERNS = [

    # -------------------
    # Payments Received / Revenue
    # -------------------
    (r'payment from .*', "Revenue", "Sales Income"),
    (r'received payment .*', "Revenue", "Sales Income"),
    (r'client payment .*', "Revenue", "Sales Income"),
    (r'customer payment .*', "Revenue", "Sales Income"),
    (r'invoice payment .*', "Revenue", "Sales Income"),
    (r'online order .*', "Revenue", "Ecommerce Sales"),
    (r'ecommerce sales .*', "Revenue", "Ecommerce Sales"),
    (r'amazon sales .*', "Revenue", "Ecommerce Sales"),
    (r'daraz sales .*', "Revenue", "Ecommerce Sales"),
    (r'shopify sales .*', "Revenue", "Ecommerce Sales"),
    (r'store sales .*', "Revenue", "Retail Sales"),
    (r'counter sales .*', "Revenue", "Retail Sales"),
    (r'bulk order .*', "Revenue", "Wholesale Sales"),
    (r'wholesale .*', "Revenue", "Wholesale Sales"),
    (r'room booking .*', "Revenue", "Room Revenue"),
    (r'hotel booking .*', "Revenue", "Room Revenue"),
    (r'event booking .*', "Revenue", "Event Income"),
    (r'consultation fee .*', "Revenue", "Medical Income"),
    (r'doctor fee .*', "Revenue", "Medical Income"),
    (r'lab test .*', "Revenue", "Laboratory Income"),
    (r'pharmacy sale .*', "Revenue", "Pharmacy Sales"),
    (r'medicine sale .*', "Revenue", "Pharmacy Sales"),
    (r'subscription income .*', "Revenue", "Subscription Income"),
    (r'membership fee .*', "Revenue", "Subscription Income"),
    (r'renewal fee .*', "Revenue", "Subscription Income"),
    (r'interest income .*', "Revenue", "Finance Income"),
    (r'bank interest .*', "Revenue", "Finance Income"),
    (r'rental income .*', "Revenue", "Rental Income"),
    (r'rent received .*', "Revenue", "Rental Income"),
    (r'other income .*', "Revenue", "Other Income"),
    (r'misc income .*', "Revenue", "Other Income"),
    (r'non-operating income .*', "Revenue", "Other Income"),
    (r'commission income .*', "Revenue", "Commission Income"),
    (r'agent commission .*', "Revenue", "Commission Income"),
    (r'brokerage .*', "Revenue", "Commission Income"),
    (r'maintenance income .*', "Revenue", "Maintenance Revenue"),
    (r'freelance income .*', "Revenue", "Service Income"),
    (r'consulting income .*', "Revenue", "Service Income"),
    (r'professional fee .*', "Revenue", "Service Income"),

    # -------------------
    # Payments Made / Expenses
    # -------------------
    (r'payment to .*', "Expense", "Vendor Payment"),
    (r'payed to .*', "Expense", "Vendor Payment"),
    (r'supplier payment .*', "Expense", "Vendor Payment"),
    (r'contractor payment .*', "Expense", "Contractor Expense"),
    (r'consulting fee paid .*', "Expense", "Contractor Expense"),
    (r'freelancer payment .*', "Expense", "Freelancer Expense"),
    (r'loan payment .*', "Expense", "Loan Payment"),
    (r'emi payment .*', "Expense", "Loan Payment"),
    (r'credit card payment .*', "Expense", "Credit Card"),
    (r'debit card fee .*', "Expense", "Bank Charge"),
    (r'bank service charge .*', "Expense", "Bank Charge"),
    (r'foreign transaction .*', "Expense", "Foreign Exchange"),
    (r'currency conversion .*', "Expense", "Foreign Exchange"),
    (r'remittance .*', "Expense", "Foreign Exchange"),
    (r'electricity bill .*', "Expense", "Utilities"),
    (r'water bill .*', "Expense", "Utilities"),
    (r'internet bill .*', "Expense", "Utilities"),
    (r'office rent .*', "Expense", "Rent"),
    (r'monthly rent .*', "Expense", "Rent"),
    (r'house rent .*', "Expense", "Rent"),
    (r'apartment rent .*', "Expense", "Rent"),
    (r'cleaning service .*', "Expense", "Cleaning"),
    (r'laundry service .*', "Expense", "Laundry"),
    (r'dry cleaning .*', "Expense", "Laundry"),
    (r'printer paper .*', "Expense", "Office Supplies"),
    (r'office supplies .*', "Expense", "Office Supplies"),
    (r'printer ink .*', "Expense", "Office Supplies"),
    (r'software subscription .*', "Expense", "Software"),
    (r'software license .*', "Expense", "Software License"),
    (r'plugin purchase .*', "Expense", "Software License"),
    (r'premium upgrade .*', "Expense", "Software License"),
    (r'cloud hosting .*', "Expense", "Hosting Expense"),
    (r'server maintenance .*', "Expense", "Hosting Expense"),
    (r'website hosting .*', "Expense", "Subscription"),
    (r'subscription .*', "Expense", "Subscription"),
    (r'gym subscription .*', "Expense", "Fitness"),
    (r'sports fee .*', "Expense", "Fitness"),
    (r'fitness club .*', "Expense", "Fitness"),
    (r'marketing campaign .*', "Expense", "Marketing"),
    (r'ad campaign .*', "Expense", "Marketing"),
    (r'google ads .*', "Expense", "Marketing"),
    (r'facebook ads .*', "Expense", "Marketing"),
    (r'seo services .*', "Expense", "Marketing"),
    (r'influencer .*', "Expense", "Marketing"),
    (r'banner printing .*', "Expense", "Marketing"),
    (r'flyer printing .*', "Expense", "Marketing"),
    (r'brand photoshoot .*', "Expense", "Branding"),
    (r'brand design .*', "Expense", "Branding"),
    (r'brand guidelines .*', "Expense", "Branding"),
    (r'driver salary .*', "Expense", "Driver Expense"),
    (r'driver overtime .*', "Expense", "Driver Expense"),
    (r'vehicle fuel .*', "Expense", "Fuel Expense"),
    (r'vehicle maintenance .*', "Expense", "Vehicle Expense"),
    (r'taxi .*', "Expense", "Travel"),
    (r'air ticket .*', "Expense", "Travel"),
    (r'hotel booking .*', "Expense", "Travel"),
    (r'restaurant bill .*', "Expense", "Food & Dining"),
    (r'food delivery .*', "Expense", "Food & Dining"),
    (r'food & beverage .*', "Expense", "Meals"),
    (r'entertainment .*', "Expense", "Entertainment"),
    (r'cinema .*', "Expense", "Entertainment"),
    (r'movie ticket .*', "Expense", "Entertainment"),
    (r'subscription service .*', "Expense", "Entertainment"),
    (r'music subscription .*', "Expense", "Entertainment"),
    (r'personal care .*', "Expense", "Personal Care"),
    (r'salon service .*', "Expense", "Personal Care"),
    (r'spa service .*', "Expense", "Personal Care"),
    (r'beauty products .*', "Expense", "Personal Care"),
    (r'grocery purchase .*', "Expense", "Grocery"),
    (r'supermarket purchase .*', "Expense", "Grocery"),
    (r'vegetable purchase .*', "Expense", "Grocery"),
    (r'fruit purchase .*', "Expense", "Grocery"),
    (r'insurance premium .*', "Expense", "Insurance Premium"),
    (r'policy renewal .*', "Expense", "Insurance Premium"),
    (r'claim payment .*', "Expense", "Insurance Claims"),
    (r'catering service .*', "Expense", "Catering"),
    (r'project expense .*', "Expense", "Project Expense"),
    (r'project material .*', "Expense", "Project Expense"),
    (r'production cost .*', "Expense", "Production Expense"),
    (r'manufacturing cost .*', "Expense", "Production Expense"),
    (r'factory overhead .*', "Expense", "Factory Expense"),
    (r'plant operation .*', "Expense", "Factory Expense"),
    (r'machine breakdown .*', "Expense", "Manufacturing Loss"),
    (r'production downtime .*', "Expense", "Manufacturing Loss"),
    (r'quality check .*', "Expense", "Quality Control"),
    (r'qc inspection .*', "Expense", "Quality Control"),
    (r'defect analysis .*', "Expense", "Quality Control"),
    (r'security guard salary .*', "Expense", "Security Expense"),
    (r'security service .*', "Expense", "Security Expense"),
    (r'cctv installation .*', "Expense", "Security Equipment"),
    (r'training material .*', "Expense", "Training Expense"),
    (r'skill development .*', "Expense", "Training Expense"),
    (r'online course .*', "Expense", "Training Expense"),
    (r'compliance fee .*', "Expense", "Compliance"),
    (r'regulatory filing .*', "Expense", "Compliance"),
    (r'statutory compliance .*', "Expense", "Compliance"),
    (r'government fee .*', "Expense", "Government Fee"),
    (r'license renewal .*', "Expense", "Government Fee"),
    (r'permit fee .*', "Expense", "Government Fee"),
    (r'zakat .*', "Expense", "Charity"),
    (r'sadqa .*', "Expense", "Charity"),
    (r'ngo donation .*', "Expense", "Charity"),

    # -------------------
    # Assets - Current
    # -------------------
    (r'cash .*', "Asset", "Cash & Cash Equivalents"),
    (r'petty cash .*', "Asset", "Cash & Cash Equivalents"),
    (r'bank .*', "Asset", "Bank Accounts"),
    (r'mobile wallet .*', "Asset", "Digital Wallets"),
    (r'easypaisa .*', "Asset", "Digital Wallets"),
    (r'jazzcash .*', "Asset", "Digital Wallets"),
    (r'paypal .*', "Asset", "Digital Wallets"),
    (r'stripe .*', "Asset", "Digital Wallets"),
    (r'accounts receivable .*', "Asset", "Accounts Receivable"),
    (r'customer receivable .*', "Asset", "Accounts Receivable"),
    (r'advance .*', "Asset", "Supplier Advances"),
    (r'inventory .*', "Asset", "Inventory"),
    (r'raw materials .*', "Asset", "Inventory"),
    (r'work in progress .*', "Asset", "Inventory"),
    (r'wip .*', "Asset", "Inventory"),
    (r'finished goods .*', "Asset", "Inventory"),
    (r'stock .*', "Asset", "Inventory"),
    (r'prepaid .*', "Asset", "Prepaid Expenses"),
    (r'short-term deposit .*', "Asset", "Short-term Investments"),

    # -------------------
    # Assets - Fixed
    # -------------------
    (r'land .*', "Asset", "Land"),
    (r'building .*', "Asset", "Buildings"),
    (r'factory building .*', "Asset", "Buildings"),
    (r'office building .*', "Asset", "Buildings"),
    (r'plant .*', "Asset", "Plant & Machinery"),
    (r'machinery .*', "Asset", "Plant & Machinery"),
    (r'machine .*', "Asset", "Plant & Machinery"),
    (r'equipment .*', "Asset", "Equipment"),
    (r'production equipment .*', "Asset", "Equipment"),
    (r'computer .*', "Asset", "IT Equipment"),
    (r'laptop .*', "Asset", "IT Equipment"),
    (r'server .*', "Asset", "IT Equipment"),
    (r'network device .*', "Asset", "IT Equipment"),
    (r'vehicle .*', "Asset", "Vehicles"),
    (r'car .*', "Asset", "Vehicles"),
    (r'truck .*', "Asset", "Vehicles"),
    (r'van .*', "Asset", "Vehicles"),
    (r'motorcycle .*', "Asset", "Vehicles"),
    (r'furniture .*', "Asset", "Furniture & Fixtures"),
    (r'fixtures .*', "Asset", "Furniture & Fixtures"),
    (r'office furniture .*', "Asset", "Furniture & Fixtures"),
    (r'software license .*', "Asset", "Intangible Assets"),
    (r'software .*', "Asset", "Intangible Assets"),
    (r'patent .*', "Asset", "Intangible Assets"),
    (r'trademark .*', "Asset", "Intangible Assets"),
    (r'copyright .*', "Asset", "Intangible Assets"),
    (r'goodwill .*', "Asset", "Goodwill"),

    # -------------------
    # Liabilities - Current
    # -------------------
    (r'accounts payable .*', "Liability", "Accounts Payable"),
    (r'supplier payable .*', "Liability", "Accounts Payable"),
    (r'vendor payable .*', "Liability", "Accounts Payable"),
    (r'short-term loan .*', "Liability", "Short-term Loans"),
    (r'working capital loan .*', "Liability", "Short-term Loans"),
    (r'overdraft .*', "Liability", "Bank Overdraft"),
    (r'tax payable .*', "Liability", "Taxes Payable"),
    (r'vat payable .*', "Liability", "Taxes Payable"),
    (r'gst payable .*', "Liability", "Taxes Payable"),
    (r'income tax payable .*', "Liability", "Taxes Payable"),
    (r'salary payable .*', "Liability", "Payroll Liabilities"),
    (r'wages payable .*', "Liability", "Payroll Liabilities"),
    (r'payroll payable .*', "Liability", "Payroll Liabilities"),
    (r'accrued expense .*', "Liability", "Accrued Liabilities"),
    (r'accrual .*', "Liability", "Accrued Liabilities"),
    (r'deferred revenue .*', "Liability", "Deferred Revenue"),
    (r'advance from customer .*', "Liability", "Deferred Revenue"),

    # -------------------
    # Liabilities - Long Term
    # -------------------
    (r'long-term loan .*', "Liability", "Long-term Loans"),
    (r'mortgage .*', "Liability", "Long-term Loans"),
    (r'bond payable .*', "Liability", "Bonds Payable"),
    (r'debenture .*', "Liability", "Bonds Payable"),
    (r'lease liability .*', "Liability", "Lease Obligations"),

    # -------------------
    # Equity
    # -------------------
    (r'owner capital .*', "Equity", "Owner's Equity"),
    (r'owner investment .*', "Equity", "Owner's Equity"),
    (r'partner capital .*', "Equity", "Partner Equity"),
    (r'share capital .*', "Equity", "Share Capital"),
    (r'paid-up capital .*', "Equity", "Share Capital"),
    (r'retained earnings .*', "Equity", "Retained Earnings"),
    (r'accumulated profit .*', "Equity", "Retained Earnings"),
    (r'dividend .*', "Equity", "Dividend Distribution"),
    (r'profit distribution .*', "Equity", "Dividend Distribution"),


    # -------------------
    # Shipping & Logistics
    # -------------------
    (r'shipping cost .*', "Expense", "Shipping & Logistics"),
    (r'courier charge .*', "Expense", "Shipping & Logistics"),
    (r'freight .*', "Expense", "Shipping & Logistics"),
    (r'parcel delivery .*', "Expense", "Shipping & Logistics"),
    (r'express delivery .*', "Expense", "Shipping & Logistics"),
    (r'logistics .*', "Expense", "Shipping & Logistics"),
    (r'tracking fee .*', "Expense", "Shipping & Logistics"),
    (r'import duty .*', "Expense", "Customs & Duties"),
    (r'customs charge .*', "Expense", "Customs & Duties"),
    (r'import tax .*', "Expense", "Customs & Duties"),
    (r'excise duty .*', "Expense", "Customs & Duties"),
    (r'export duty .*', "Expense", "Customs & Duties"),

    # -------------------
    # Maintenance & Repairs
    # -------------------
    (r'vehicle repair .*', "Expense", "Maintenance & Repairs"),
    (r'car service .*', "Expense", "Maintenance & Repairs"),
    (r'machine repair .*', "Expense", "Maintenance & Repairs"),
    (r'equipment repair .*', "Expense", "Maintenance & Repairs"),
    (r'plant maintenance .*', "Expense", "Maintenance & Repairs"),
    (r'facility maintenance .*', "Expense", "Maintenance & Repairs"),
    (r'office maintenance .*', "Expense", "Maintenance & Repairs"),
    (r'building repair .*', "Expense", "Maintenance & Repairs"),
    (r'generator repair .*', "Expense", "Maintenance & Repairs"),
    (r'air conditioner repair .*', "Expense", "Maintenance & Repairs"),
    (r'plumbing .*', "Expense", "Maintenance & Repairs"),
    (r'electrical repair .*', "Expense", "Maintenance & Repairs"),

    # -------------------
    # Legal & Professional
    # -------------------
    (r'lawyer fee .*', "Expense", "Legal & Professional"),
    (r'legal consultation .*', "Expense", "Legal & Professional"),
    (r'court fee .*', "Expense", "Legal & Professional"),
    (r'attorney fee .*', "Expense", "Legal & Professional"),
    (r'audit fee .*', "Expense", "Audit & Accounting"),
    (r'accounting fee .*', "Expense", "Audit & Accounting"),
    (r'financial advisor .*', "Expense", "Consulting Services"),
    (r'consultant fee .*', "Expense", "Consulting Services"),
    (r'hr consultant .*', "Expense", "Consulting Services"),
    (r'project consultant .*', "Expense", "Consulting Services"),

    # -------------------
    # HR & Employee Related
    # -------------------
    (r'employee bonus .*', "Expense", "Payroll & Benefits"),
    (r'employee incentive .*', "Expense", "Payroll & Benefits"),
    (r'staff welfare .*', "Expense", "Payroll & Benefits"),
    (r'medical insurance .*', "Expense", "Payroll & Benefits"),
    (r'life insurance .*', "Expense", "Payroll & Benefits"),
    (r'pension contribution .*', "Expense", "Payroll & Benefits"),
    (r'salary advance .*', "Asset", "Employee Advances"),
    (r'employee loan .*', "Asset", "Employee Advances"),

    # -------------------
    # Franchise & Licensing
    # -------------------
    (r'franchise fee .*', "Expense", "Franchise & Licensing"),
    (r'royalty payment .*', "Expense", "Franchise & Licensing"),
    (r'brand license .*', "Expense", "Franchise & Licensing"),
    (r'software royalty .*', "Expense", "Franchise & Licensing"),
    (r'license fee .*', "Expense", "Franchise & Licensing"),
    (r'patent fee .*', "Expense", "Franchise & Licensing"),

    # -------------------
    # Rebates & Discounts
    # -------------------
    (r'customer rebate .*', "Expense", "Rebates & Discounts"),
    (r'supplier discount .*', "Revenue", "Rebates & Discounts"),
    (r'cashback .*', "Revenue", "Rebates & Discounts"),
    (r'sales discount .*', "Expense", "Rebates & Discounts"),
    (r'promotional discount .*', "Expense", "Rebates & Discounts"),

    # -------------------
    # Miscellaneous Expenses
    # -------------------
    (r'bank penalty .*', "Expense", "Bank Charges & Penalties"),
    (r'late payment fee .*', "Expense", "Bank Charges & Penalties"),
    (r'fine .*', "Expense", "Bank Charges & Penalties"),
    (r'penalty .*', "Expense", "Bank Charges & Penalties"),
    (r'donation .*', "Expense", "Charity"),
    (r'charity .*', "Expense", "Charity"),
    (r'zakat .*', "Expense", "Charity"),
    (r'community support .*', "Expense", "Charity"),
    (r'sponsorship .*', "Expense", "Marketing"),
    (r'event sponsorship .*', "Expense", "Marketing"),
    (r'conference fee .*', "Expense", "Professional Events"),
    (r'exhibition fee .*', "Expense", "Professional Events"),
    (r'trade show .*', "Expense", "Professional Events"),
    (r'webinar fee .*', "Expense", "Professional Events"),
    (r'online event .*', "Expense", "Professional Events"),

    # -------------------
    # Miscellaneous Revenue
    # -------------------
    (r'refund received .*', "Revenue", "Refund & Recovery"),
    (r'claim received .*', "Revenue", "Refund & Recovery"),
    (r'insurance claim .*', "Revenue", "Refund & Recovery"),
    (r'grants received .*', "Revenue", "Grants & Donations"),
    (r'donation received .*', "Revenue", "Grants & Donations"),
    (r'government subsidy .*', "Revenue", "Government Subsidy"),
    (r'tax refund .*', "Revenue", "Tax Refund"),
    (r'rebate received .*', "Revenue", "Rebates & Discounts"),
    (r'cashback received .*', "Revenue", "Rebates & Discounts"),

    # -------------------
    # Utilities & Communication (Extended)
    # -------------------
    (r'mobile recharge .*', "Expense", "Utilities"),
    (r'telecom bill .*', "Expense", "Utilities"),
    (r'internet service .*', "Expense", "Utilities"),
    (r'cloud service .*', "Expense", "Utilities"),
    (r'data plan .*', "Expense", "Utilities"),
    (r'vpn service .*', "Expense", "Utilities"),

    # -------------------
    # Travel & Entertainment (Extended)
    # -------------------
    (r'business trip .*', "Expense", "Travel"),
    (r'flight booking .*', "Expense", "Travel"),
    (r'taxi fare .*', "Expense", "Travel"),
    (r'hotel stay .*', "Expense", "Travel"),
    (r'conference travel .*', "Expense", "Travel"),
    (r'dining with client .*', "Expense", "Food & Dining"),
    (r'client entertainment .*', "Expense", "Entertainment"),

    # -------------------
    # IT & Software (Extended)
    # -------------------
    (r'cloud storage .*', "Expense", "IT & Software"),
    (r'saas subscription .*', "Expense", "IT & Software"),
    (r'app subscription .*', "Expense", "IT & Software"),
    (r'website maintenance .*', "Expense", "IT & Software"),
    (r'domain renewal .*', "Expense", "IT & Software"),
    (r'data backup .*', "Expense", "IT & Software"),

    # -------------------
    # Manufacturing & Production (Extended)
    # -------------------
    (r'raw material purchase .*', "Expense", "Production Expense"),
    (r'packing material .*', "Expense", "Production Expense"),
    (r'production labor .*', "Expense", "Production Expense"),
    (r'factory overhead .*', "Expense", "Production Expense"),
    (r'maintenance of machinery .*', "Expense", "Production Expense"),
    (r'quality assurance .*', "Expense", "Production Expense"),

    # -------------------
    # Education & Training (Extended)
    # -------------------
    (r'course fee .*', "Expense", "Training & Education"),
    (r'training fee .*', "Expense", "Training & Education"),
    (r'online course .*', "Expense", "Training & Education"),
    (r'conference fee .*', "Expense", "Training & Education"),
    (r'workshop fee .*', "Expense", "Training & Education"),

    # -------------------
    # Investment & Finance (Extended)
    # -------------------
    (r'share purchase .*', "Asset", "Investments"),
    (r'bond purchase .*', "Asset", "Investments"),
    (r'mutual fund .*', "Asset", "Investments"),
    (r'stock dividend .*', "Revenue", "Investment Income"),
    (r'interest received .*', "Revenue", "Investment Income"),
    (r'capital gain .*', "Revenue", "Investment Income"),
    (r'loan interest .*', "Expense", "Finance Costs"),
    (r'bank fee .*', "Expense", "Finance Costs"),
     # -------------------
    (r'share purchase .*', "Asset", "Investments"),
    (r'bond purchase .*', "Asset", "Investments"),
    (r'mutual fund .*', "Asset", "Investments"),
    (r'stock dividend .*', "Revenue", "Investment Income"),
    (r'interest received .*', "Revenue", "Investment Income"),
    (r'capital gain .*', "Revenue", "Investment Income"),
    (r'loan interest .*', "Expense", "Finance Costs"),
    (r'bank fee .*', "Expense", "Finance Costs"),

    # -------------------
    # Travel & Meals
    # -------------------
   (r'travel expense .*', "Expense", "Travel"),
   (r'flight .*', "Expense", "Travel"),
   (r'hotel .*', "Expense", "Travel"),
   (r'taxi .*', "Expense", "Travel"),
   (r'lunch .*', "Expense", "Meals"),
   (r'dinner .*', "Expense", "Meals"),
   (r'coffee .*', "Expense", "Meals"),
   (r'lunch with client .*', "Expense", "Meals"),

   # -------------------
   # Office & Supplies
   # -------------------
  (r'printer ink .*', "Expense", "Office Supplies"),
  (r'printer paper .*', "Expense", "Office Supplies"),
  (r'stationery .*', "Expense", "Office Supplies"),
  (r'office supplies .*', "Expense", "Office Supplies"),
  (r'office furniture .*', "Expense", "Furniture"),
  (r'desk .*', "Expense", "Furniture"),
  (r'chair .*', "Expense", "Furniture"),

  # -------------------
  # Consulting & Fees
  # -------------------
  (r'consulting fee .*', "Expense", "Consulting/Fees"),
  (r'contractor payment .*', "Expense", "Consulting/Fees"),

  # -------------------
  # Website & Hosting
  # -------------------
  (r'website hosting .*', "Expense", "Hosting"),
  (r'domain .*', "Expense", "Hosting"),
  (r'hosting fee .*', "Expense", "Hosting"),


   

    # -------------------
    # Payments Received / Revenue
    # -------------------
    (r'payment from .*', "Revenue", "Sales Income"),
    (r'received payment .*', "Revenue", "Sales Income"),
    (r'square payment .*', "Revenue", "Sales Income"),
    (r'paypal .*', "Revenue", "Online Payments"),
    (r'stripe .*', "Revenue", "Online Payments"),
    (r'bank transfer .*', "Revenue", "Bank Transfers"),
    (r'ach deposit .*', "Revenue", "Bank Transfers"),
    (r'check received .*', "Revenue", "Checks"),

    # -------------------
    # Expenses - Food & Beverage
    # -------------------
    (r'starbucks.*', "Expense", "Food & Beverage"),
    (r'mcdonald.*', "Expense", "Food & Beverage"),
    (r'burger king.*', "Expense", "Food & Beverage"),
    (r'coffee bean.*', "Expense", "Food & Beverage"),

    # -------------------
    # Expenses - Travel
    # -------------------
    (r'uber .*', "Expense", "Travel Expense"),
    (r'lyft .*', "Expense", "Travel Expense"),
    (r'airbnb .*', "Expense", "Travel Expense"),
    (r'expedia .*', "Expense", "Travel Expense"),
    (r'hilton .*', "Expense", "Travel Expense"),
    (r'marriott .*', "Expense", "Travel Expense"),
    (r'flight .*', "Expense", "Travel Expense"),

    # -------------------
    # Expenses - Office Supplies / Equipment
    # -------------------
    (r'amazon marketplace .*', "Expense", "Office Supplies"),
    (r'staples .*', "Expense", "Office Supplies"),
    (r'office depot .*', "Expense", "Office Supplies"),
    (r'best buy .*', "Expense", "Office Equipment"),
    (r'apple .*', "Expense", "Office Equipment"),
    (r'dell .*', "Expense", "Office Equipment"),

    # -------------------
    # Expenses - Marketing / Advertising
    # -------------------
    (r'google ads.*', "Expense", "Marketing / Advertising"),
    (r'facebook ads.*', "Expense", "Marketing / Advertising"),
    (r'instagram ads.*', "Expense", "Marketing / Advertising"),
    (r'linkedin ads.*', "Expense", "Marketing / Advertising"),
    (r'bing ads.*', "Expense", "Marketing / Advertising"),

    # -------------------
    # Expenses - Utilities / Internet / Phone
    # -------------------
    (r'verizon .*', "Expense", "Utilities / Telecom"),
    (r'at&t .*', "Expense", "Utilities / Telecom"),
    (r'comcast .*', "Expense", "Utilities / Internet"),
    (r'spectrum .*', "Expense", "Utilities / Internet"),

    # -------------------
    # Expenses - Travel / Vehicle
    # -------------------
    (r'fuel .*', "Expense", "Vehicle / Fuel"),
    (r'gas station .*', "Expense", "Vehicle / Fuel"),
    (r'toll .*', "Expense", "Vehicle / Tolls"),
    (r'parking .*', "Expense", "Vehicle / Parking"),

    # -------------------
    # Payroll / Contractors
    # -------------------
    (r'payroll .*', "Expense", "Payroll"),
    (r'contractor payment .*', "Expense", "Freelance / Contractors"),
    (r'freelancer .*', "Expense", "Freelance / Contractors"),
    (r'upwork .*', "Expense", "Freelance / Contractors"),
    (r'fiverr .*', "Expense", "Freelance / Contractors"),

    # -------------------
    # Bank / Finance / Fees
    # -------------------
    (r'bank fee .*', "Expense", "Finance Costs"),
    (r'wire transfer .*', "Expense", "Finance Costs"),
    (r'interest received .*', "Revenue", "Investment Income"),
    (r'loan payment .*', "Expense", "Finance Costs"),
    (r'credit card .*', "Expense", "Finance Costs"),

    # -------------------
    # Insurance
    # -------------------
    (r'insurance .*', "Expense", "Insurance"),
    (r'health insurance .*', "Expense", "Insurance"),
    (r'vehicle insurance .*', "Expense", "Insurance"),

    # -------------------
    # Education / Subscriptions
    # -------------------
    (r'udemy .*', "Expense", "Education / Training"),
    (r'coursera .*', "Expense", "Education / Training"),
    (r'subscription .*', "Expense", "Miscellaneous"),

    # -------------------
    # Miscellaneous
    # -------------------
    (r'miscellaneous .*', "Expense", "Miscellaneous"),
    (r'other .*', "Expense", "Miscellaneous"),

    # -------------------
    # Investment / Asset / Capital
    # -------------------
    (r'share purchase .*', "Asset", "Investments"),
    (r'bond purchase .*', "Asset", "Investments"),
    (r'mutual fund .*', "Asset", "Investments"),
    (r'stock dividend .*', "Revenue", "Investment Income"),
    (r'capital gain .*', "Revenue", "Investment Income"),
]

CATEGORY_MAP = {

    # ===========================
    #          ASSETS
    # ===========================

    # --- Current Assets ---
    "cash": ("Asset", "Cash & Cash Equivalents"),
    "petty cash": ("Asset", "Cash & Cash Equivalents"),
    "bank balance": ("Asset", "Bank Accounts"),
    "bank cash": ("Asset", "Bank Accounts"),
    "savings account": ("Asset", "Bank Accounts"),
    "checking account": ("Asset", "Bank Accounts"),
    "mobile wallet": ("Asset", "Digital Wallets"),
    "easypaisa": ("Asset", "Digital Wallets"),
    "jazzcash": ("Asset", "Digital Wallets"),
    "paypal": ("Asset", "Digital Wallets"),
    "stripe": ("Asset", "Digital Wallets"),
    "accounts receivable": ("Asset", "Accounts Receivable"),
    "customer receivable": ("Asset", "Accounts Receivable"),
    "trade receivable": ("Asset", "Accounts Receivable"),
    "advance to supplier": ("Asset", "Supplier Advances"),
    "advance payment": ("Asset", "Supplier Advances"),
    "supplier advance": ("Asset", "Supplier Advances"),
    "inventory": ("Asset", "Inventory"),
    "raw materials": ("Asset", "Inventory"),
    "wip": ("Asset", "Inventory"),
    "work in progress": ("Asset", "Inventory"),
    "finished goods": ("Asset", "Inventory"),
    "stock": ("Asset", "Inventory"),
    "prepaid insurance": ("Asset", "Prepaid Expenses"),
    "prepaid rent": ("Asset", "Prepaid Expenses"),
    "prepaid expense": ("Asset", "Prepaid Expenses"),
    "short-term deposit": ("Asset", "Short-term Investments"),
    "short-term investment": ("Asset", "Short-term Investments"),
    "government grant receivable": ("Asset", "Grants & Subsidies"),
    "tax refund receivable": ("Asset", "Tax Refunds"),

    # --- Fixed Assets ---
    "land": ("Asset", "Land"),
    "building": ("Asset", "Buildings"),
    "factory building": ("Asset", "Buildings"),
    "office building": ("Asset", "Buildings"),
    "plant": ("Asset", "Plant & Machinery"),
    "machinery": ("Asset", "Plant & Machinery"),
    "machine": ("Asset", "Plant & Machinery"),
    "equipment": ("Asset", "Equipment"),
    "production equipment": ("Asset", "Equipment"),
    "computer": ("Asset", "IT Equipment"),
    "laptop": ("Asset", "IT Equipment"),
    "server": ("Asset", "IT Equipment"),
    "networking device": ("Asset", "IT Equipment"),
    "vehicle": ("Asset", "Vehicles"),
    "car": ("Asset", "Vehicles"),
    "truck": ("Asset", "Vehicles"),
    "van": ("Asset", "Vehicles"),
    "motorcycle": ("Asset", "Vehicles"),
    "furniture": ("Asset", "Furniture & Fixtures"),
    "fixtures": ("Asset", "Furniture & Fixtures"),
    "office furniture": ("Asset", "Furniture & Fixtures"),

    # --- Intangible Assets ---
    "software license": ("Asset", "Intangible Assets"),
    "software": ("Asset", "Intangible Assets"),
    "patent": ("Asset", "Intangible Assets"),
    "trademark": ("Asset", "Intangible Assets"),
    "copyright": ("Asset", "Intangible Assets"),
    "goodwill": ("Asset", "Goodwill"),

    # --- Investments ---
    "share investment": ("Asset", "Investments"),
    "bond investment": ("Asset", "Investments"),
    "mutual fund": ("Asset", "Investments"),
    "capital investment": ("Asset", "Investments"),

    # ===========================
    #        LIABILITIES
    # ===========================

    # --- Current Liabilities ---
    "accounts payable": ("Liability", "Accounts Payable"),
    "supplier payable": ("Liability", "Accounts Payable"),
    "vendor payable": ("Liability", "Accounts Payable"),
    "short-term loan": ("Liability", "Short-term Loans"),
    "working capital loan": ("Liability", "Short-term Loans"),
    "overdraft": ("Liability", "Bank Overdraft"),
    "tax payable": ("Liability", "Taxes Payable"),
    "vat payable": ("Liability", "Taxes Payable"),
    "gst payable": ("Liability", "Taxes Payable"),
    "income tax payable": ("Liability", "Taxes Payable"),
    "salary payable": ("Liability", "Payroll Liabilities"),
    "wages payable": ("Liability", "Payroll Liabilities"),
    "payroll payable": ("Liability", "Payroll Liabilities"),
    "accrued expense": ("Liability", "Accrued Liabilities"),
    "accrual": ("Liability", "Accrued Liabilities"),
    "deferred revenue": ("Liability", "Deferred Revenue"),
    "advance from customer": ("Liability", "Deferred Revenue"),
    "loan payable": ("Liability", "Loans Payable"),
    "interest payable": ("Liability", "Finance Costs"),

    # --- Long-Term Liabilities ---
    "long-term loan": ("Liability", "Long-term Loans"),
    "mortgage": ("Liability", "Long-term Loans"),
    "bond payable": ("Liability", "Bonds Payable"),
    "debenture": ("Liability", "Bonds Payable"),
    "lease liability": ("Liability", "Lease Obligations"),

    # ===========================
    #            EQUITY
    # ===========================
    "owner capital": ("Equity", "Owner's Equity"),
    "owner investment": ("Equity", "Owner's Equity"),
    "partner capital": ("Equity", "Partner Equity"),
    "share capital": ("Equity", "Share Capital"),
    "paid-up capital": ("Equity", "Share Capital"),
    "retained earnings": ("Equity", "Retained Earnings"),
    "accumulated profit": ("Equity", "Retained Earnings"),
    "dividend": ("Equity", "Dividend Distribution"),
    "profit distribution": ("Equity", "Dividend Distribution"),

    # ===========================
    #           REVENUE
    # ===========================
    "sales": ("Revenue", "Product Sales"),
    "product sales": ("Revenue", "Product Sales"),
    "sale of goods": ("Revenue", "Product Sales"),
    "item sold": ("Revenue", "Product Sales"),
    "invoice payment": ("Revenue", "Product Sales"),
    "service income": ("Revenue", "Service Income"),
    "professional fee": ("Revenue", "Service Income"),
    "consulting fee": ("Revenue", "Service Income"),
    "consulting income": ("Revenue", "Service Income"),
    "freelance income": ("Revenue", "Service Income"),
    "maintenance income": ("Revenue", "Maintenance Revenue"),
    "commission income": ("Revenue", "Commission Income"),
    "agent commission": ("Revenue", "Commission Income"),
    "brokerage": ("Revenue", "Commission Income"),
    "subscription income": ("Revenue", "Subscription Income"),
    "membership fee": ("Revenue", "Subscription Income"),
    "renewal fee": ("Revenue", "Subscription Income"),
    "interest income": ("Revenue", "Finance Income"),
    "bank interest": ("Revenue", "Finance Income"),
    "rental income": ("Revenue", "Rental Income"),
    "rent received": ("Revenue", "Rental Income"),
    "other income": ("Revenue", "Other Income"),
    "misc income": ("Revenue", "Other Income"),
    "non-operating income": ("Revenue", "Other Income"),
    "refund received": ("Revenue", "Refund & Recovery"),
    "grant received": ("Revenue", "Grants & Donations"),
    "cashback received": ("Revenue", "Rebates & Discounts"),

    # ===========================
    #        EXPENSES
    # ===========================
    "electricity bill": ("Expense", "Utilities"),
    "water bill": ("Expense", "Utilities"),
    "internet bill": ("Expense", "Utilities"),
    "mobile recharge": ("Expense", "Utilities"),
    "telecom bill": ("Expense", "Utilities"),
    "office rent": ("Expense", "Rent"),
    "freelancer payment": ("Expense", "Consulting"),
    "printer ink purchase": ("Expense", "Office Supplies"),
    "office supplies purchase": ("Expense", "Office Supplies"),
    "snacks for office": ("Expense", "Office Supplies"),
    "marketing campaign": ("Expense", "Marketing"),
    "sponsorship": ("Expense", "Marketing"),
    "travel expense": ("Expense", "Travel"),
    "flight booking": ("Expense", "Travel"),
    "hotel stay": ("Expense", "Travel"),
    "taxes paid": ("Expense", "Taxes"),
    "salary": ("Expense", "Salaries & Wages"),
    "salary payment": ("Expense", "Salaries & Wages"),
    "employee benefits": ("Expense", "Salaries & Wages"),
    "training expense": ("Expense", "Training & Education"),
    "conference fee": ("Expense", "Training & Education"),
    "insurance premium": ("Expense", "Insurance"),
    "vehicle maintenance": ("Expense", "Vehicles"),
    "generator repair": ("Expense", "Maintenance & Repairs"),
    "plumbing repair": ("Expense", "Maintenance & Repairs"),
    "electrical repair": ("Expense", "Maintenance & Repairs"),
    "legal fees": ("Expense", "Legal & Professional"),
    "audit fee": ("Expense", "Audit & Accounting"),
    "consultant fee": ("Expense", "Consulting"),
    "software subscription": ("Expense", "IT & Software"),
    "cloud service": ("Expense", "IT & Software"),
    "depreciation": ("Expense", "Depreciation"),
    "bank charges": ("Expense", "Finance Charges"),
    "loan interest": ("Expense", "Finance Charges"),
    "late payment fee": ("Expense", "Finance Charges"),
    "donation": ("Expense", "Charity"),
    "charity": ("Expense", "Charity"),
    
    # ===========================
    #        MISC / OTHER
    # ===========================
    "loan repayment": ("Liability", "Long Term Liability"),
    "bank deposit": ("Asset", "Bank Accounts"),
    "client payment": ("Revenue", "Sales Income"),
    "subscription": ("Revenue", "Subscription Income"),
    "supplier invoice": ("Expense", "Supplies"),
    "advance from employee": ("Liability", "Employee Advances"),
    "advance to employee": ("Asset", "Employee Advances"),
    "refund to customer": ("Expense", "Refund & Recovery"),
    "rebate given": ("Expense", "Rebates & Discounts"),
    "rebate received": ("Revenue", "Rebates & Discounts"),
    "patent fee": ("Expense", "Franchise & Licensing"),
    "royalty payment": ("Expense", "Franchise & Licensing"),
    "franchise fee": ("Expense", "Franchise & Licensing"),
    "import duty": ("Expense", "Customs & Duties"),
    "customs charge": ("Expense", "Customs & Duties"),
}
