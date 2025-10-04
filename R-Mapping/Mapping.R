# Install/load packages
# install.packages(c("tidyverse", "sf", "tmap", "tigris"))
library(dplyr)
library(tidyr)
library(sf)
library(tmap)
library(tigris)
library(stringr)
library(devtools)
devtools::install_github(UrbanInstitute/urbnmapr)


#--------------------------
# 1. Load Your Data
#--------------------------
# Replace with your file path
data <- read.csv("county_income_hpi_data.csv")

# Peek at columns
glimpse(data)

# Example: assume your dataset has "GEOID" or "county_fips"
# If your key column is named differently, update below:
data <- data %>%
  mutate(GEOID = str_pad(as.character(county_fips_full), 5, pad = "0"))

#--------------------------
# 2. Get County Boundaries
#--------------------------
options(tigris_use_cache = TRUE)

counties <- counties(cb = TRUE, year = 2020, class = "sf") %>%
  st_transform(2163)  # Projected for CONUS + AK/HI repositioning

#--------------------------
# 3. Merge Your Data to Geometry
#--------------------------
map_data <- counties %>%
  left_join(data, by = c("GEOID" = "GEOID"))

#--------------------------
# 4. Make a Choropleth
#--------------------------
# Example: Map median household income
# Replace `median_household_income` with your actual column
tm_shape(map_data) +
  tm_polygons("median_household_income",
              style = "quantile",
              n = 7,
              palette = "viridis",
              title = "Median Household Income") +
  tm_borders(lwd = 0.1, col = "white") +
  tm_layout(
    frame = FALSE,
    legend.outside = TRUE,
    title = "US Counties - Median Household Income",
    title.size = 1.2
  )

map_data %>%
  ggplot() + 
  geom_polygon(data = counties, mapping = aes(long,lat)) +
  coord_map(projection = "albers", lat0 = 39, lat1 = 45) +
  theme(legend.title = element_text(),
        legend.key.width = unit(.5, "in")) +
  labs(fill = "Homeownership rate") +
  theme_urban_map()


map_data %>%
  ggplot(aes(long, lat, group = group, fill = horate)) +
  geom_polygon(color = NA) +
  scale_fill_gradientn(labels = scales::percent,
                       guide = guide_colorbar(title.position = "top")) +
  geom_polygon(data = states, mapping = aes(long, lat, group = group),
               fill = NA, color = "#ffffff") +
  coord_map(projection = "albers", lat0 = 39, lat1 = 45) +
  theme(legend.title = element_text(),
        legend.key.width = unit(.5, "in")) +
  labs(fill = "Homeownership rate") +
  theme_urban_map()

#--------------------------
# 5. Export to PNG/PDF
#--------------------------
tmap_save(
  tm = last_map(),
  filename = "county_map_income.png",
  dpi = 300,
  width = 10,
  height = 7,
  units = "in"
)
