Data dictionary: shop database

== Table customers ==
One row per registered buyer.
- customers.id: integer surrogate key, referenced by orders.customer_id
- customers.name: given name only, unique in this dataset
- customers.city: shipping city; one of Berlin, Osaka, Porto, Austin, Nairobi, Oslo
- customers.joined_at: ISO date YYYY-MM-DD of account creation, all in 2024

== Table products ==
Catalog of sellable items with a fixed list price.
- products.id: integer surrogate key, referenced by order_items.product_id
- products.name: lower-case product name, e.g. usb hub, keyboard
- products.price: unit list price in dollars, REAL; revenue = quantity * price

== Table orders ==
One row per checkout; line items live in order_items.
- orders.id: integer surrogate key
- orders.customer_id: foreign key to customers.id
- orders.ordered_at: ISO date YYYY-MM-DD, all in 2025; month = substr(ordered_at, 1, 7)

== Table order_items ==
One row per (order, product); the only place quantities live.
- order_items.order_id: foreign key to orders.id
- order_items.product_id: foreign key to products.id
- order_items.quantity: units of the product in that order, 1 to 3

== Value ranges (from the current data) ==
customers: 20 rows
- customers.id: 20 distinct, min 1, max 20
- customers.name: 20 distinct, min Ana, max Tom
- customers.city: 6 distinct, min Austin, max Porto
- customers.joined_at: 20 distinct, min 2024-01-01, max 2024-12-18
products: 8 rows
- products.id: 8 distinct, min 1, max 8
- products.name: 8 distinct, min desk mat, max webcam
- products.price: 8 distinct, min 19.0, max 199.0
orders: 160 rows
- orders.id: 160 distinct, min 1, max 160
- orders.customer_id: 20 distinct, min 1, max 20
- orders.ordered_at: 128 distinct, min 2025-01-03, max 2025-12-28
order_items: 410 rows
- order_items.order_id: 160 distinct, min 1, max 160
- order_items.product_id: 8 distinct, min 1, max 8
- order_items.quantity: 3 distinct, min 1, max 3

== Glossary (question phrase -> SQL idiom) ==
- order total: SUM(order_items.quantity * products.price) over the lines of one order
- revenue: the same sum over any set of orders; never use products.price alone
- average order value: SUM of order totals divided by COUNT(DISTINCT orders.id)
- bought together: two products sharing an order_id in order_items
- recent: ORDER BY orders.ordered_at DESC
- share of revenue: revenue of the subset divided by total revenue, times 100
- distinct products in an order: COUNT(DISTINCT product_id) GROUP BY order_id
- never ordered X: customers whose id is NOT IN the customer_ids of orders containing X

== Worked query patterns (X stands for the literal from the question) ==
- customers per city: SELECT c.city, COUNT(*) AS n FROM customers c GROUP BY c.city ORDER BY n DESC
- top spenders: SELECT c.name, SUM(oi.quantity * p.price) AS spent FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id GROUP BY c.id ORDER BY spent DESC LIMIT 3
- orders per month: SELECT substr(o.ordered_at, 1, 7) AS month, COUNT(*) AS n FROM orders o GROUP BY month ORDER BY month
- product revenue: SELECT p.name, SUM(oi.quantity * p.price) AS revenue FROM products p JOIN order_items oi ON oi.product_id = p.id GROUP BY p.id ORDER BY revenue DESC LIMIT 1
- orders with many distinct products: SELECT COUNT(*) FROM (SELECT order_id FROM order_items GROUP BY order_id HAVING COUNT(DISTINCT product_id) > 2)
- bought together with X: SELECT p2.name, COUNT(*) AS n FROM order_items a JOIN order_items b ON a.order_id = b.order_id AND a.product_id <> b.product_id JOIN products p1 ON p1.id = a.product_id JOIN products p2 ON p2.id = b.product_id WHERE p1.name = X GROUP BY p2.id ORDER BY n DESC LIMIT 1

== Conventions ==
- Dates are ISO text YYYY-MM-DD; compare and slice them as strings (substr, LIKE, BETWEEN).
- Surrogate keys are INTEGER PRIMARY KEY; every foreign key is declared with REFERENCES.
- Money is REAL in dollars with no currency column; round only in the final SELECT.
- Names carry no apostrophes; string literals may be single quoted safely.
- Prefer explicit JOIN ... ON over comma joins; alias every table.
- Aggregations must GROUP BY every non-aggregated selected column.
- Use LIMIT for top-N questions and ORDER BY the aggregate, not the key.
- NULL means unknown or still open; filter with IS NULL / IS NOT NULL, never = NULL.
- Percent answers are 100.0 * part / whole; cast one side to REAL to avoid integer division.
- Return one result set; no temporary tables, no PRAGMA, no multiple statements.
- SQLite has no RIGHT JOIN and no FULL JOIN; rewrite with LEFT JOIN from the other side.
- Boolean expressions evaluate to 0 or 1, so SUM(status = X) counts matching rows.
- String comparison is case sensitive; the stored values are all lower case except names.
- julianday(text) turns an ISO date into a day number; subtract two for a duration in days.
- There is no schema qualifier; refer to tables by their bare names.
- Ties in top-N questions are broken by the surrogate key ascending.