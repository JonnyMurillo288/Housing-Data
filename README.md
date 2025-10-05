# Housing & Rent Affordability Project (WIP)

This project is a **work in progress** exploring long-term housing affordability trends at the **U.S. county level**.  
Aiming to build a consistent panel from the **1950s to the present**, using **Decennial Census** (historical values) and the **American Community Survey (ACS)** for modern data.  

![County Housing Affordability Map](https://github.com/JonnyMurillo288/Housing-Data/blob/main/HAI_2000_2023_publish_v2.jpg)
---

## Core Metrics

### Housing Affordability Index (HAI)  

\[
\text{HAI} = \frac{\text{Median Home Value}}{\text{Median Household Income}}
\]

### Rent Affordability Index (RAI)  
\[
\text{RAI} = \frac{\text{Median\ Household\ Income}{Median\ Gross\ Rent}}
\]

- Uses **median gross rent** (rent + utilities) as reported by Census/ACS.  
- Captures how much local incomes can cover typical rental costs.  

---

## Data Sources

- **Income:** Median household income from Decennial Census, SAIPE, and ACS (county level).  
- **Home Values:** Median value of owner-occupied units from Decennial Census and ACS.  
- **Rents:** Median gross rent from Decennial Census and ACS.  

---

## Features

Users will be able to view:  

### Single-Year Snapshots
- Median household income  
- Median home value  
- Median gross rent  
- HAI (absolute or indexed)  
- RAI  

### Year-to-Year Comparisons
- % change in income, home prices, or rent  
- Change in affordability indices (HAI, RAI) between two user-selected years  
- Long-term divergence between income growth and housing costs  

---

## Status

- Scripts implemented for:  
  - Fetching Census/ACS data (income, home values, rent).  
  - Incorporating offline decennial tables (1980, 1990).  
  - Generating panel datasets (CSV & Parquet).
  - Mapping & GeoJSON integration
  - Dashboard visualization  
  - Data validation & cleaning
  - Adding data beyond year 2000 (Manual HPMS Downloads)
- **Next steps:**  
  - Adding data beyond 1980
  - Adjusting pipeline to allow for custom geographies
      - State, Block Group, MSA, CBSA, etc.
  

---
