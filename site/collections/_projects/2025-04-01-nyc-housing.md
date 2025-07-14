---
date: 2025-04-01 05:20:35 +0300
title: 'Sleeper Neighborhoods in NYC: Data Workflow & Anomaly Detection'
subtitle: Python + Docker + Airflow + Postgres + Pydeck
image: '/images/housing.jpg'
---
Rent in NYC is expensive. I'm building a system to find neighborhoods with unexpectedly low rent for their respective noise levels, crime rates, and historical rent prices. I built a pipeline that aggregated complaint data from NYPD, construction permit data from NYC, historic rent data from Zillow & StreetEasy, and geospatial transportation noise data from USDOT into each neighborhood. Then, I built an interactive visualization tool that highlights the location of each sleeper neighborhood with its rent, safety, etc.
