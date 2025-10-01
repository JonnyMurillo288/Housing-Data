# Housing & Rent Affordability Project (WIP)

This project is a **work in progress** exploring long-term housing affordability trends at the **U.S. county level**.  
We aim to build a consistent panel from the **1950s to the present**, using **Decennial Census** (historical values) and the **American Community Survey (ACS)** for modern data.  

---

## Core Metrics

### Housing Affordability Index (HAI)  
\[
HAI = \frac{Median\ Home\ Value}{Median\ Household\ Income}
\]

### Rent Affordability Index (RAI)  
\[
RAI = \frac{Median\ Household\ Income}{Median\ Gross\ Rent}
\]

- Uses **median gross rent** (rent + utilities) as reported by Census/ACS.  
- Captures how much local incomes can cover typical rental costs.  

---

## Data Sources

- **Income:** Median household income from Decennial Census, SAIPE, and ACS (county level).  
- **Home Values:** Median value of owner-occupied units from Decennial Census and ACS.  
- **Rents:** Median gross rent from Decennial Census and ACS.  
- **Indices:** FHFA House Price Index (HPI) for relative affordability comparisons.  

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
- **Next steps:**  
  - Mapping & GeoJSON integration  
  - Dashboard visualization  
  - Data validation & cleaning
  - Adding data beyond year 2000 (Manual HPMS Downloads)

---
