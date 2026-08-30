-- DataCoding SQL interview patterns
-- Dialect: ANSI-style window functions with PostgreSQL-compatible date examples.
-- Always state the expected dialect before using date arithmetic in an interview.

-- -----------------------------------------------------------------------------
-- 1. Latest row per entity (deterministic deduplication)
-- Output grain: one row per customer_id.
-- -----------------------------------------------------------------------------
WITH ranked_events AS (
    SELECT
        customer_id,
        event_id,
        event_time,
        status,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY event_time DESC, event_id DESC
        ) AS recency_number
    FROM customer_events
)
SELECT
    customer_id,
    event_id,
    event_time,
    status
FROM ranked_events
WHERE recency_number = 1;


-- -----------------------------------------------------------------------------
-- 2. Top two products by revenue within each category
-- DENSE_RANK keeps all products tied at a rank; ROW_NUMBER would force two rows.
-- -----------------------------------------------------------------------------
WITH product_revenue AS (
    SELECT
        category_id,
        product_id,
        SUM(quantity * unit_price) AS revenue
    FROM order_items
    GROUP BY category_id, product_id
),
ranked_products AS (
    SELECT
        category_id,
        product_id,
        revenue,
        DENSE_RANK() OVER (
            PARTITION BY category_id
            ORDER BY revenue DESC
        ) AS revenue_rank
    FROM product_revenue
)
SELECT
    category_id,
    product_id,
    revenue,
    revenue_rank
FROM ranked_products
WHERE revenue_rank <= 2
ORDER BY category_id, revenue_rank, product_id;


-- -----------------------------------------------------------------------------
-- 3. Running total and trailing three-row average
-- Explicit ROWS frames make duplicate dates behave by row position.
-- -----------------------------------------------------------------------------
SELECT
    account_id,
    transaction_id,
    transaction_time,
    amount,
    SUM(amount) OVER (
        PARTITION BY account_id
        ORDER BY transaction_time, transaction_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_amount,
    AVG(amount) OVER (
        PARTITION BY account_id
        ORDER BY transaction_time, transaction_id
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS trailing_three_row_average
FROM transactions;


-- -----------------------------------------------------------------------------
-- 4. Time since the previous event
-- PostgreSQL returns an interval for timestamp subtraction.
-- -----------------------------------------------------------------------------
WITH event_gaps AS (
    SELECT
        user_id,
        event_id,
        event_time,
        LAG(event_time) OVER (
            PARTITION BY user_id
            ORDER BY event_time, event_id
        ) AS previous_event_time
    FROM user_events
)
SELECT
    user_id,
    event_id,
    event_time,
    previous_event_time,
    event_time - previous_event_time AS gap_from_previous_event
FROM event_gaps;


-- -----------------------------------------------------------------------------
-- 5. Month-over-month change
-- Aggregate first, then apply LAG to the monthly grain.
-- -----------------------------------------------------------------------------
WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', order_time) AS revenue_month,
        SUM(order_total) AS revenue
    FROM orders
    GROUP BY DATE_TRUNC('month', order_time)
),
with_previous AS (
    SELECT
        revenue_month,
        revenue,
        LAG(revenue) OVER (ORDER BY revenue_month) AS previous_revenue
    FROM monthly_revenue
)
SELECT
    revenue_month,
    revenue,
    previous_revenue,
    revenue - previous_revenue AS absolute_change,
    (revenue - previous_revenue) / NULLIF(previous_revenue, 0.0) AS growth_rate
FROM with_previous
ORDER BY revenue_month;


-- -----------------------------------------------------------------------------
-- 6. Conditional aggregation for a one-row-per-user funnel
-- -----------------------------------------------------------------------------
SELECT
    user_id,
    MAX(CASE WHEN event_name = 'viewed' THEN 1 ELSE 0 END) AS viewed_flag,
    MAX(CASE WHEN event_name = 'added_to_cart' THEN 1 ELSE 0 END) AS cart_flag,
    MAX(CASE WHEN event_name = 'purchased' THEN 1 ELSE 0 END) AS purchased_flag,
    MIN(CASE WHEN event_name = 'purchased' THEN event_time END) AS first_purchase_time
FROM user_events
GROUP BY user_id;


-- -----------------------------------------------------------------------------
-- 7. Pre-aggregate before a one-to-many join to protect revenue grain
-- -----------------------------------------------------------------------------
WITH order_totals AS (
    SELECT
        order_id,
        SUM(quantity * unit_price) AS order_revenue
    FROM order_items
    GROUP BY order_id
)
SELECT
    c.customer_id,
    COUNT(o.order_id) AS order_count,
    COALESCE(SUM(ot.order_revenue), 0) AS customer_revenue
FROM customers AS c
LEFT JOIN orders AS o
    ON o.customer_id = c.customer_id
LEFT JOIN order_totals AS ot
    ON ot.order_id = o.order_id
GROUP BY c.customer_id;


-- -----------------------------------------------------------------------------
-- 8. Gap-and-island grouping for consecutive active dates (PostgreSQL)
-- -----------------------------------------------------------------------------
WITH distinct_days AS (
    SELECT DISTINCT
        user_id,
        CAST(event_time AS DATE) AS active_date
    FROM user_events
),
numbered_days AS (
    SELECT
        user_id,
        active_date,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY active_date
        ) AS day_number
    FROM distinct_days
),
islands AS (
    SELECT
        user_id,
        active_date,
        active_date - (day_number * INTERVAL '1 day') AS island_key
    FROM numbered_days
)
SELECT
    user_id,
    MIN(active_date) AS streak_start,
    MAX(active_date) AS streak_end,
    COUNT(*) AS streak_days
FROM islands
GROUP BY user_id, island_key
ORDER BY user_id, streak_start;
