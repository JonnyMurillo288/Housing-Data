# Install/load packages
# install.packages(c("tidyverse", "sf", "tmap", "tigris"))
library(dplyr)
library(tidyr)
library(sf)
library(tmap)
library(tigris)
library(stringr)
remotes::install_github("UrbanInstitute/urbnmapr",force=T)
library(urbnmapr)

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
  st_transform(5070)  # Projected for CONUS + AK/HI repositioning

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
# Example: zoom to continental US
map_conus <- map_data[!map_data$STATE_NAME %in% c("Alaska", "Hawaii", 
                                                  "Puerto Rico", 
                                                  "Guam", 
                                                  "American Samoa", 
                                                  "Northern Mariana Islands", "United States Virgin Islands",
                                                  "Virgin Islands","Commonwealth of the Northern Mariana Islands"), ]
map_conus <- map_conus %>%
  distinct(geometry, .keep_all = TRUE)

unique(map_data$STATE_NAME)
qts = quantile(map_conus$HAI_change,probs = c(0.01,.10,.20,.30,.40,.50,.60,.70,.80,.90,1),na.rm=T)

top10 <- map_conus %>%
  st_drop_geometry() %>%      # drop geometry for sorting
  arrange(desc(HAI_change)) %>%
  slice(1:10)

tm_shape(map_conus) +
  tm_polygons("HAI_change",
              style = "fixed",
              breaks = qts,
              palette = "BuRd",
              midpoint = 34,
              title = "HAI Change (%)") +
  tm_borders(lwd = 0.1, col = "white") +
  
  # highlight top 10
  tm_shape(map_conus %>% filter(county_name %in% top10$county_name)) +
  tm_symbols(size = 0.3, col = "black") +
  tm_text(text = "county_name", size = 0.6, ymod = -0.5) +
  
  tm_layout(frame = FALSE,
            legend.outside = TRUE,
            title = "US Counties - HAI Change %",
            title.size = 1.2)

# =============================================
# Plot with the labels for the top 10
# =============================================
library(sf)
library(dplyr)

top10 <- map_conus %>%
  st_drop_geometry() %>%
  arrange(desc(HAI_change)) %>%
  slice(1:10)

# 2. Get centroids of those counties
top10_sf <- map_conus %>%
  filter(county_name %in% top10$county_name) %>%
  slice(match(top10$county_name, county_name)) %>%
  st_centroid()


# 3. Create offset label points (push east/north for readability)
coords <- st_coordinates(top10_sf)
bb <- st_bbox(map_conus)

# 2. Midpoints and offsets
xmid <- (bb$xmin + bb$xmax) / 2
ymid <- (bb$ymin + bb$ymax) / 2
xoff <- (bb$xmax - bb$xmin) * 0.2
yoff <- (bb$ymax - bb$ymin) * 0.2

# 3. Quadrant-based anchor positions
label_points <- data.frame(
  X_label = ifelse(coords[,1] < xmid, 
                   coords[,1] - xoff,  # left side
                   coords[,1] + xoff), # right side
  Y_label = ifelse(coords[,2] < ymid, 
                   coords[,2] - yoff,  # bottom
                   coords[,2] + yoff)  # top
)

# 4. Add jitter (2% of width/height)
label_points$X_label <- label_points$X_label +
  runif(nrow(label_points), -0.2, 0.2) * (bb$xmax - bb$xmin)
label_points$Y_label <- label_points$Y_label +
  runif(nrow(label_points), -0.2, 0.2) * (bb$ymax - bb$ymin)

# 5. Make sf labels
label_sf <- st_as_sf(label_points,
                     coords = c("X_label", "Y_label"),
                     crs = st_crs(map_conus))

label_sf$label <- paste0(top10$county_name,
                         "\nHAI 2000: ", top10$HAI_2000,
                         "\nHAI 2023: ", top10$HAI_2023)

# 6. Create leader lines
cent_coords <- st_coordinates(top10_sf)
# Label coordinates
lab_coords  <- st_coordinates(label_sf)

# Build lines
lines_list <- lapply(1:nrow(cent_coords), function(i) {
  st_linestring(rbind(cent_coords[i, ], lab_coords[i, ]))
})

lines_sf <- st_sf(geometry = st_sfc(lines_list, crs = st_crs(map_conus)))

# 6. Plot with tmap
map <- tm_shape(map_conus, bbox = st_bbox(map_conus)) +
  tm_polygons("HAI_change", style = "fixed", breaks = qts,
              palette = "BuRd", midpoint = 34, title = "HAI Change (%)") +
  tm_borders(lwd = 0.1, col = "white") +
  tm_shape(lines_sf, bbox = st_bbox(map_conus)) + 
  tm_lines(col = "black", lwd = 0.7) +
  tm_shape(label_sf, bbox = st_bbox(map_conus)) + 
  tm_text("label", size = 0.6, just = "left", remove.overlap = FALSE) +
  tm_layout(
    frame = TRUE,
    legend.outside = TRUE,
    asp = 0,   # 🔑 don't force aspect ratio shrink
    outer.margins = c(0, 0, 0, 0),   # 🔑 let map fill full width
    inner.margins = c(0.02, 0.25, 0.02, 0.02), # space for labels if needed
    title = "US Counties - HAI Change %",
    title.size = 1.2
  )

map

tmap_save(map, "map.jpeg", width = 3000, height = 1800, units = "px", dpi = 300)


# =============================================
# Plot a Separate Bar Chart for these Top 10
# =============================================

# Reshape data for plotting
top10_long <- top10 %>%
  select(county_name, 
         median_household_income_change, 
         median_home_value_change) %>%
  pivot_longer(cols = c(median_household_income_change, median_home_value_change),
               names_to = "variable",
               values_to = "value")

# Plot a Bar Chart for these Top 10
ggplot(top10_long, aes(x = reorder(county_name, -value), 
                       y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  labs(x = "County",
       y = "Change (%)",
       fill = "Variable",
       title = "Top 10 Counties by HAI Change") +
  scale_fill_manual(values = c("Median_houshold_income_change" = "steelblue",
                               "median_home_value_change" = "orange")) +
  theme_minimal()




library(ggplot2)
map_data %>%
  ggplot() + 
  geom_polygon(data = counties, mapping = aes(long,lat)) +
  coord_map(projection = "albers", lat0 = 39, lat1 = 45) +
  theme(legend.title = element_text(),
        legend.key.width = unit(.5, "in")) +
  labs(fill = "Homeownership rate") 

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
