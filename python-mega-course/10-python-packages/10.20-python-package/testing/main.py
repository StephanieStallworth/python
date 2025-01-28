# from invoicing import invoice
#
# invoice.generate("invoices", "output", "pythonhow.png",
#                  "product_id", "product_name", "amount_purchased",
#                  "price_per_unit", "total_price")

# Add generate to __init__.py to make name available at the package level (even though it a function inside a module)
from invoicing import generate

generate("invoices", "output", "pythonhow.png",
                 "product_id", "product_name", "amount_purchased",
                 "price_per_unit", "total_price")
