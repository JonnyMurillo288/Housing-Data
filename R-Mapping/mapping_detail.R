# =============================================
# Top 5 HAI_change + Selected Metro Counties
# with non-overlapping labels
# =============================================
library(dplyr)
library(sf)
library(tmap)

set.seed(123)

# 1) Get top 5 by HAI_change
top5 <- map_conus %>%
  st_drop_geometry() %>%
  arrange(desc(HAI_change)) %>%
  slice(1:5)

# 2) Explicit metro counties
metro_counties <- c(
  "Miami-Dade County, Florida",
  "District of Columbia",
  "Mecklenburg County, North Carolina",
  "Cook County, Illinois",
  "Clark County, Nevada",
  "Los Angeles County, California",
  "Orange County, California",
  "Davidson County, Tennessee"
)

# 3) Combine both sets (unique)
target_names <- unique(c(top5$county_name, metro_counties))

target_df <- map_conus %>%
  st_drop_geometry() %>%
  filter(county_name %in% target_names) %>%
  distinct(county_name, .keep_all = TRUE) %>%
  mutate(id = row_number())   # stable id for join

# 4) Get centroids for selected counties
target_sf <- map_conus %>%
  inner_join(target_df %>% select(county_name, id), by = "county_name") %>%
  distinct(county_name, .keep_all = TRUE) %>%   # drop duplicates
  arrange(id) %>%
  st_centroid()

# 5) Build label anchor points
coords <- st_coordinates(target_sf)
bb <- st_bbox(map_conus)
xmid <- (bb$xmin + bb$xmax)/2
ymid <- (bb$ymin + bb$ymax)/2
xoff <- (bb$xmax - bb$xmin)*0.2
yoff <- (bb$ymax - bb$ymin)*0.2

label_points <- data.frame(
  id = target_sf$id,
  X_label = ifelse(coords[,1] < xmid, coords[,1] - xoff, coords[,1] + xoff),
  Y_label = ifelse(coords[,2] < ymid, coords[,2] - yoff, coords[,2] + yoff)
)

# jitter
label_points$X_label <- label_points$X_label +
  runif(nrow(label_points), -0.02, 0.02)*(bb$xmax - bb$xmin)
label_points$Y_label <- label_points$Y_label +
  runif(nrow(label_points), -0.02, 0.02)*(bb$ymax - bb$ymin)

# 6) Convert to sf labels & join attributes
label_sf <- st_as_sf(label_points, coords = c("X_label","Y_label"), crs = st_crs(map_conus)) %>%
  left_join(target_df %>% select(id, county_name, HAI_2000, HAI_2023), by = "id")

# 7) Overlap resolver
resolve_overlap <- function(label_sf, min_gap, xmid) {
  coords <- st_coordinates(label_sf)
  df <- label_sf %>%
    st_drop_geometry() %>%
    mutate(x = coords[,1], y = coords[,2]) %>%
    mutate(side = ifelse(x < xmid, "left", "right"))
  
  adjust_group <- function(subdf) {
    subdf <- subdf[order(-subdf$y), ]  # top → bottom
    for (i in 2:nrow(subdf)) {
      if ((subdf$y[i-1] - subdf$y[i]) < min_gap) {
        subdf$y[i] <- subdf$y[i-1] - min_gap
      }
    }
    subdf
  }
  
  df_left  <- if (any(df$side == "left"))  adjust_group(df[df$side == "left", ])  else data.frame()
  df_right <- if (any(df$side == "right")) adjust_group(df[df$side == "right", ]) else data.frame()
  
  df_new <- dplyr::bind_rows(df_left, df_right) %>% arrange(id)
  
  # rebuild sf with updated coords
  out <- st_as_sf(df_new, coords = c("x","y"), crs = st_crs(label_sf))
  out
}

min_gap <- (bb$ymax - bb$ymin) * 0.15
label_sf <- resolve_overlap(label_sf, min_gap, xmid)
label_sf <- label_sf %>%
  mutate(label_txt = paste0(
    county_name, "\n",
    "HAI 2000: ", round(HAI_2000, 1), 
    ", HAI 2023: ", round(HAI_2023, 1)
  ))

# 8) Build lines by id
cent_coords <- st_coordinates(target_sf) %>%
  as.data.frame() %>% mutate(id = target_sf$id) %>% distinct(id, .keep_all = TRUE)

lab_coords <- st_coordinates(label_sf) %>%
  as.data.frame() %>% mutate(id = label_sf$id) %>% distinct(id, .keep_all = TRUE)

lines_df <- cent_coords %>%
  inner_join(lab_coords, by = "id", suffix = c("_cent","_lab"))

stopifnot(nrow(lines_df) == nrow(target_sf))

lines_list <- lapply(seq_len(nrow(lines_df)), function(i) {
  st_linestring(rbind(
    c(lines_df$X_cent[i], lines_df$Y_cent[i]),
    c(lines_df$X_lab[i], lines_df$Y_lab[i])
  ))
})
lines_sf <- st_sf(geometry = st_sfc(lines_list, crs = st_crs(map_conus)))

# expand bbox by 10% in each direction
expand_bbox <- function(bb, factor = 0.1) {
  dx <- (bb$xmax - bb$xmin) * factor
  dy <- (bb$ymax - bb$ymin) * factor
  bb_new <- bb
  bb_new$xmin <- bb$xmin - dx
  bb_new$xmax <- bb$xmax + dx
  bb_new$ymin <- bb$ymin - dy
  bb_new$ymax <- bb$ymax + dy
  bb_new
}

bb_zoom <- expand_bbox(st_bbox(map_conus), factor = 0.1)


# 9) Plot with tmap
map <- tm_shape(map_conus, bbox = st_bbox(map_conus)) +
  tm_polygons("HAI_change", style = "fixed", breaks = qts,
              palette = "BuRd", midpoint = 34, title = "HAI Change (%)") +
  tm_borders(lwd = 0.1, col = "grey") +
  #tm_shape(lines_sf) + tm_lines(col = "black", lwd = 0.7) +
  #tm_shape(label_sf %>% filter(side == "left")) +
  #tm_text("label_txt", size = 0.7, just = "right") +
  #tm_shape(label_sf %>% filter(side == "right")) +
  #tm_text("label_txt", size = 0.7, just = "left") +
  tm_layout(
    frame = TRUE,
    #legend.position = c("right","bottom"),
    legend.outside = TRUE,
    asp = 0,
    outer.margins = c(0,0,0,0),
    inner.margins = c(0.02, 0.25, 0.02, 0.02),
    title = "Top 5 + Major Metro Counties - HAI Change %",
    title.size = 1.2
  )

map
legend <- tm_shape(map_conus, bbox = st_bbox(map_conus)) +
  tm_polygons("HAI_change", style = "fixed", breaks = qts,
              palette = "RdBu", midpoint = 34, title = "HAI Change (%)") +
  tm_borders(lwd = 0.1, col = "grey") +
  #tm_shape(lines_sf) + tm_lines(col = "black", lwd = 0.7) +
  #tm_shape(label_sf %>% filter(side == "left")) +
  #tm_text("label_txt", size = 0.7, just = "right") +
  #tm_shape(label_sf %>% filter(side == "right")) +
  #tm_text("label_txt", size = 0.7, just = "left") +
  tm_layout(
    legend.only = TRUE,
    frame = TRUE,
    legend.position = c("right","bottom"),
    #legend.outside = TRUE,
    asp = 0,
    outer.margins = c(0,0,0,0),
    inner.margins = c(0.02, 0.25, 0.02, 0.02),
    title = "Top 5 + Major Metro Counties - HAI Change %",
    title.size = 1.2
  )

bb_super_zoom <- expand_bbox(st_bbox(map_conus), factor = 0.2)

tm_shape(label_sf, bbox = st_bbox(label_sf)) + 
  tm_text("label_txt", size = 0.7, just = "right") +
  tm_text("label_txt", size = 0.7, just = "left") 

label_layers <- list(
  tm_shape(label_sf, bbox = st_bbox(label_sf)) +
    tm_text("label_txt", size = 0.7, just = "right"))

tmap::tmap_arrange(label_layers)


map

legend

tmap_save(map, "map_top5_plus_metros_map_only_frame.jpeg", width = 3000, height = 1800, units = "px", dpi = 300)
tmap_save(legend, "legend.jpeg", width = 3000, height = 1800, units = "px", dpi = 300)

library(ggplot2)
labels = ggplot(label_sf %>% st_drop_geometry(),
       aes(x = 0, y = seq_along(label_txt), label = label_txt)) +
  geom_text(hjust = 0) +
  theme_void() +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank()) +
  coord_cartesian(clip = "off")

ggsave("labels.jpeg",plot=labels,width = 6, height = 4, units = "in", dpi = 300)

labels

legend_labels <- paste0(
  sprintf("%.1f", head(qts, -1)), " % – ",
  sprintf("%.1f", tail(qts, -1)), " %"
)

legend <- tm_shape(map_conus, bbox = st_bbox(map_conus)) +
  tm_polygons(
    "HAI_change",
    style = "fixed",
    breaks = qts,
    labels = legend_labels,                # <-- formatted legend labels
    palette = "BuRd",
    midpoint = 34,                   # center on zero change
    title = "HAI Change (%)"
  ) +
  tm_borders(lwd = 0.1, col = "grey") +
  tm_layout(
    legend.only = TRUE,
    legend.position = c("right", "bottom"),
    frame = TRUE,
    asp = 0,
    outer.margins = c(0, 0, 0, 0),
    inner.margins = c(0.02, 0.25, 0.02, 0.02),
    title = "HAI Change",
    title.size = 1.2,
    # subtitle via credits for clarity
    credits.text = "(HAI 2000 – HAI 2023) / HAI 2000",
    credits.size = 0.9,
    credits.position = c("RIGHT", "TOP")
  )
legend
