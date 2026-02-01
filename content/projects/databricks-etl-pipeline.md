---
title: "Databricks ETL Pipeline (Medallion + Delta Lake)"
date: 2026-09-21
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
- Total Customers
- Purchase Over Time
- Revenus Over Time
- Top 10 Categories by Revenue
- Average Daily Events by Days of Week
- Buyer VS Non-buyer
- User Distribution by US States

## Challenges & solutions

- **Issue**: Mixed event types & null categories  
  **Fix**: Enforced data types + filtered invalid values
- **Issue**: Duplicated labels in product categories
  **Fix**: Split strings and standardized category labels

## Links

- GitHub: _(add link)_
- Demo / Dashboard: _(add link or screenshot)_
