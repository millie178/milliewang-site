---
title: "Databricks ETL Pipeline (Medallion + Delta Lake)"
date: 2025-09-21
draft: false
tags: ["Databricks", "Delta Lake", "ETL", "Spark", "Data Engineering"]
summary: "Built an end-to-end ETL pipeline in Databricks using Bronze/Silver/Gold Delta tables, scheduled Jobs, and KPI dashboarding."
---

## Overview

Built an end-to-end ETL pipeline in **Databricks** using the **Medallion architecture (Bronze/Silver/Gold)** and **Delta Lake** tables. Ingested raw e-commerce data, cleaned and modeled it into analytics-ready tables, and produced KPI metrics for reporting.

## What I built

- **Bronze**: Ingest raw files into Delta tables
- **Silver**: Clean + deduplicate + enforce types
- **Gold**: Analytics layer with KPI tables (revenue, user geography, etc.)
- **Automation**: Configured a **Databricks Job** to run the notebook/pipeline on a schedule
- **Dashboarding**: Built KPI visualizations (total customers, total products, total revenue, top categories etc.)

## Tech stack

- Databricks, Apache Spark (PySpark)
- Delta Lake (ACID tables)
- SQL
- Python

## Key metrics produced

- Total Revenue
  ![Total Revenue](/databricks_etl/total_revenue.png)
- Total Customers
  ![Unique Customers](/databricks_etl/unique_customers.png)
- Top 10 Product Categories by Revenue
  ![Top 10 Product Categories by Revenue](/databricks_etl/top_10_categories_by_revenue.png)
- Average Daily Events by Days of Week\
  Events: like, view, purchase
  ![Average Daily Events by Days of Week](/databricks_etl/average_daily_events_by_day_of_week.png)
- Buyer VS Non-buyer
  ![Buyer VS Non-buyer](/databricks_etl/buyer_vs_non-buyer.png)
- User Distribution by US States
  ![User Distribution by US States](/databricks_etl/user_distribution_by_us_state.png)

## Challenges & solutions

- **Issue**: Mixed event types & null categories\  
  **Fix**: Enforced data types + filtered invalid values
- **Issue**: Duplicated labels in product categories\
  **Fix**: Split strings and standardized category labels

## Links

- GitHub: [databricks_etl_project](https://github.com/millie178/databricks-etl-ecommerce)
