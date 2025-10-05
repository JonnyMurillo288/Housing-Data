# Housing & Rent Affordability Project (WIP)

🔗 [Live Dashboard](https://us-housing-affordability.streamlit.app/)

This project is a **work in progress** exploring long-term housing affordability trends at the **U.S. county level**.  
It aims to build a consistent panel from the **1950s to the present**, combining **Decennial Census** (historical data) and **American Community Survey (ACS)** data for modern years.  

![County Housing Affordability Map](https://github.com/JonnyMurillo288/Housing-Data/blob/main/HAI_2000_2023_publish_v2.jpg)

---

## Core Metrics

### Housing Affordability Index (HAI)
**Formula:**  
`HAI = Median Home Value / Median Household Income`

### Rent Affordability Index (RAI)
**Formula:**  
`RAI = Median Household Income / Median Gross Rent`

- Uses **median gross rent** (rent + utilities) as reported by Census/ACS.  
- Captures how much local incomes can cover typical rental costs.  

---

## Data Sources

- **Income:** Median household income from Decennial Census, SAIPE, and ACS (county level).  
- **Home Values:** Median value of owner-occupied units from Decennial Census and ACS.  
- **Rents:** Median gross rent from Decennial Census and ACS.  

---

## Features

Users can explore:  

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

**Implemented so far:**
- ✅ Fetching Census/ACS data (income, home values, rent)  
- ✅ Incorporating offline decennial tables (1980, 1990)  
- ✅ Generating panel datasets (CSV & Parquet)  
- ✅ Mapping & GeoJSON integration  
- ✅ Dashboard visualization  
- ✅ Data validation & cleaning  
- ✅ Adding data beyond year 2000 (manual HPMS downloads)  

**Next Steps:**
- ⏳ Add data beyond 1980  
- ⏳ Adjust pipeline for custom geographies (State, Block Group, MSA, CBSA, etc.)  

---

