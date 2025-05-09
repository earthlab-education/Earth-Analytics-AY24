
# %% [markdown]
# # Habitat suitability under climate change
# 
# [Our changing climate is changing where key grassland species can live,
# and grassland management and restoration practices will need to take
# this into
# account.](https://www.frontiersin.org/articles/10.3389/fpls.2017.00730/full)
# 
# In this coding challenge, you will create a habitat suitability model
# for a species of your choice that lives in the continental United States
# (CONUS). We have this limitation because the downscaled climate data we
# suggest, the [MACAv2 dataset](https://www.climatologylab.org/maca.html),
# is only available in the CONUS – if you find other downscaled climate
# data at an appropriate resolution you are welcome to choose a different
# study area. If you don’t have anything in mind, you can take a look at
# Sorghastrum nutans, a grass native to North America. [In the past 50
# years, its range has moved
# northward](https://www.gbif.org/species/2704414).
# 
# Your suitability assessment will be based on combining multiple data
# layers related to soil, topography, and climate. You will also need to
# create a **modular, reproducible, workflow** using functions and loops.
# To do this effectively, we recommend planning your code out in advance
# using a technique such as pseudocode outline or a flow diagram. We
# recommend planning each of the blocks below out into multiple steps. It
# is unnecessary to write a step for every line of code unles you find
# that useful. As a rule of thumb, aim for steps that cover the major
# structures of your code in 2-5 line chunks.
# 
# ## STEP 1: STUDY OVERVIEW
# 
# Before you begin coding, you will need to design your study.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>What question do you hope to answer about potential future changes in
# habitat suitability?</p></div></div>

# %% [markdown]
# # **Study Question**
# 
# In the Comanche and Pawnee National Grasslands, what have been historical 
# suitable habitats compared to 100 years in the future (wosrt case emissions 
# scenario) - will the Rocky Mountain Juniper's suitable habitat move location 
# to find more suitable habitat in reponse to climate change? Does the species 
# find expansion in these suitable habitats, or contraction? 

# %% [markdown]
# ### Species
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Select the species you want to study, and research it’s habitat
# parameters in scientific studies or other reliable sources. You will
# want to look for reviews or overviews of the data, since an individual
# study may not have the breadth needed for this purpose. In the US, the
# National Resource Conservation Service can have helpful fact sheets
# about different species. University Extension programs are also good
# resources for summaries.</p>
# <p>Based on your research, select soil, topographic, and climate
# variables that you can use to determine if a particular location and
# time period is a suitable habitat for your species.</p></div></div>
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>Write a description of your species. What habitat is it found in?
# What is its geographic range? What, if any, are conservation threats to
# the species? What data will shed the most light on habitat suitability
# for this species?</p></div></div>

# %% [markdown]
# # **Plant Species Description**
# 
# <figure>
#     <img
#         src="https://plants.sc.egov.usda.gov/ImageLibrary/standard/jusc2_001_svp.jpg"
#         alt="Rocky Mountain Juniper, courtesy of EHerman, D.E., et al. Provided by 
#         ND State Soil Conservation Committee. United States, North Dakota)" 
#         height="400px"/>
#     <img
#         src="https://plants.sc.egov.usda.gov/ImageLibrary/standard/jusc2_002_shp.jpg"
#         alt="Rocky Mountain Juniper close up, courtesy of Herman, D.E., et al. 
#         Provided by ND State Soil Conservation Committee. United States, North Dakota)" 
#         height="400px"/>
#     <figcaption aria-hidden="true">
#         Rocky Mountain Juniper, courtesy of Herman, D.E., et al.
#         (<span 
#             class="citation">
#             Herman, D.E., et al. Provided by ND State Soil Conservation Committee. 
#             United States, North Dakota
#          </span>)
#     </figcaption>
# </figure>
# 
# *Juniperus scopulorum* Sargent, also known as Rocky Mountain Juniper 
# Tree, is a coniferous tree that slowly grows to about 20 feet at 20 years 
# but can grow up to 50 feet mature (U.S. Department of Agriculture - 
# Natural Resources Conservation Service 2025). It also has a long life 
# span with high drought tolerance which works to its favor in semi-airid 
# climates like that of the Eastern Colorado Plains (U.S. Department of 
# Agriculture -  Natural Resources Conservation Service 2025). It is native to 
# North America throughout the Rocky Mountain region between British Colombia 
# and Alberta, Canada all the way south through the continental U.S. to the 
# four corner states and east through some of the Great Plains states 
# (Stevens 2008). The Rocky Mountain juniper grows best below elevations 
# of 7,500 feet, but typically between 5,000 and 7,500 feet (U.S. Department 
# of Agriculture - Natural Resources Conservation Service 2002). While the 
# Rocky Mountain juniper is frequently used as an ornamental tree or shrub 
# tree in wildlife plantings and shelterbelts, it is also used by a range 
# of birds and mammals for its ground cover or nesting materials (Stevens 2002). 
# The Rocky Mountain junipers' berries also provide an important part of 
# both bird and mammal diets (Stevens 2002). Additionaly, First Peoples have 
# used this berries from this juniper for tea and other applications 
# for medicinal purposes (Stevens 2008). Some potential problems or concerns 
# about this tree/shrub tree are that it carries cedar-apple rust disease which 
# is not harmful to itself but is harmful to other tree species (U.S. Department of 
# Agriculture - Natural Resources Conservation Service 2002). Beyond diseases, 
# the Rocky Mountain juniper increases are "at the expense of other ecosystems, 
# such as shrublands, grasslands, riparian forests, and open pine and oak 
# forests, with consequent impacts on plant and wildlife species that flourish 
# in open or unique ecosystems" (Hanberry 2022).
# 
# This species was chosen because of its prevelance in both the Pawnee National 
# Grassland and the Comanche National Grassland. For the Pawnee National Grassland, 
# the 'General Technical Report' from the USDA, Forest Service on *Vascular Plant* 
# *Species of the Pawnee National Grassland* states that the *Juniperus scopulorum* 
# Sargent occurs in the cliffs and ravines of this National Grassland (Hazlett 
# 1998). Similarly in the Comanche National Grassland, this same type of technical 
# report states that *Juniperus scopulorum* Sargent occurs in all 3 counties 
# (Baca, Otero, and Las Animas Counties) that this grassland is in, which also 
# grow in "shaded rocky canyons and ravines" and "Rocky: exposed limestone/shale 
# barrens" (Hazlett 2004). Because this species is expanding habitats, I am curious 
# to see what the outcome of the habitat suitability model will be. 
# This project could be done with any plant species of choice and could 
# even be adapted to animal habitat suitability, but using other datasets 
# and variables that would pertain to that species or locations of study areas chosen. 
# 
# Considering its growth conditions, *Juniperus scopulorum* Sargent has optimal 
# pH values between 8.5 (max) and 5 (min), and annual precipitation between 26 
# inches and 9 inches (66cm - 23cm), and  (U.S. Department of Agriculture -  Natural 
# Resources Conservation Service 2025). Root depths at a minimum are 9 inches (23cm) 
# and the minimum temperature is -38 (°F) (U.S. Department of Agriculture -  Natural 
# Resources Conservation Service 2025). While there are other factors that could be 
# taken into consideration for growth, those are the two variables that will be 
# chosen for this project. To look at other growth condition factors as well as 
# additional information about *Juniperus scopulorum* Sargent, please visit 
# [Natural Resources Conservation Service - USDA, Plant Profile Characteristics for *Juniperus scopulorum* Sarg.](https://plants.usda.gov/plant-profile/JUSC2/characteristics).
# 
# 
# ### Citations
# 
# * Hanberry, Brice B. 2022. “Westward Expansion by Juniperus Virginiana 
# of the Eastern United States and Intersection with Western Juniperus 
# Species in a Novel Assemblage.” Forests 13 (1): 101. 
# https://doi.org/10.3390/f13010101.
# 
# * Hazlett, Donald. 1998. “Vascular Plant Species of the Pawnee National 
# Grassland.” U.S. Department of Agriculture, U.S. Forest Service. 
# https://www.fs.usda.gov/rm/pubs/rmrs_gtr017.pdf.
# 
# * Hazlett, Donald L. 2004. “Vascular Plant Species of the Comanche 
# National Grassland in Southeastern Colorado.” U.S. Department of 
# Agriculture, U.S. Forest Service. 
# https://www.fs.usda.gov/rm/pubs/rmrs_gtr130.pdf.
# 
# * Stevens, Michelle. 2002. “Plant Guide for Rocky Mountain Juniper 
# (Juniperus Scopulorum Sarg.).” U.S. Department of Agriculture - Natural 
# Resources Conservation Service. 
# https://plants.usda.gov/DocumentLibrary/plantguide/pdf/cs_jusc2.pdf.
# 
# * U.S. Department of Agriculture -  Natural Resources Conservation Service. 2025. 
# “Plant Profile Characteristics of Juniperus Scopulorum Sarg. (Rocky Mountain 
# Juniper).” Usda.gov. 2025. 
# https://plants.usda.gov/plant-profile/JUSC2/characteristics.
# 
# * U.S. Department of Agriculture - Natural Resources Conservation Service. 2002. 
# “Plant Fact Sheet: ROCKY MOUNTAIN JUNIPER (Juniperus Scopulorum Sarg.).” 
# U.S. Department of Agriculture, Natural Resources Conservation Service. 
# https://plants.usda.gov/DocumentLibrary/factsheet/pdf/fs_jusc2.pdf.
# 
# 
# 
# 
# 

# %% [markdown]
# ### Sites
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Select at least two site to study, such as two of the U.S. National
# Grasslands. You can download the <a
# href="https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.NationalGrassland.zip">USFS
# National Grassland Units</a> and select your study sites. Generate a
# site map for each location.</p>
# <p>When selecting your sites, you might want to look for places that are
# marginally habitable for this species, since those locations will be
# most likely to show changes due to climate.</p></div></div>
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>Write a site description for each of your sites, or for all of your
# sites as a group if you have chosen a large number of linked sites. What
# differences or trends do you expect to see among your sites?</p></div></div>

# %% [markdown]
# ## **Site Descriptions**
# 
# >**Land Acknowledgement**:
# First Peoples and Indigenous Peoples are the original stewards of the land taken before, 
# during, and beyond colonialism in the Americas. Much of recorded history has prioritized 
# Euro American perspectives and experiences that have misrepresented, excluded, and erased 
# Black, Indigenous, and People of Color perspectives and experiences (History Colorado, 
# Grounding Virtues). First People's and Indigenous People's stewardship and connection 
# to this land must be honored and respected.
# 
# *Please read the land acknowledgement above the map before continuing reading,* 
# *it is imperative to naming and confronting racialized systems that dominate history*
# *and our current era.*
# 
# Two study areas were chosen from the [U.S. National Grassland Units](https://data-usfs.hub.arcgis.com/datasets/usfs::national-grassland-units-feature-layer/explore?location=39.118879%2C-104.194688%2C7.05). 
# Both of these grasslands are in Colorado, one in nothern Colorado and 
# one in southern Colorado. I chose these two study areas specifically 
# because I wanted to see if there was a difference between the northern 
# grassland (Pawnee) and the southern grassland (Comanche) in terms of 
# habitat suitability. A possible disadvantage to choosing two grasslands 
# that are relatively close to one another is that the model outcomes showing 
# a more suitable habitat in a certain study area, may not be noticeable. 
# An advantage may be providing model outcomes within a singular unit of a state
# or Colorado Eastern Plains region, is providing insight for that area. 
# 
# The two grasslands are similar in that they are split geographically 
# in two units or areas, however the Comanche is about twice the size of the Pawnee 
# National Grassland.
# 
# The research sources will show that the Rocky Mountain Juniper can be found 
# in both National Grasslands chosen (Hazlett 1998 and 2004). So, I found it 
# unecessary to try to do something like plotting GBIF occurances of the species; 
# however, depening on the base of knowledge and, if you aren't sure if a certain 
# species occurs in an area, it would be a great idea to plot GBIF occurences and 
# see if they occur in the administrative boundaries chosen. 
# 
# * ### Comanche National Grassland
# ![Comanche National Grassland, Vogel Canyon](https://www.colorado.com/_next/image?url=https%3A%2F%2Fapi.colorado.com%2Fsites%2Fdefault%2Ffiles%2F2024-10%2Fw_Vogel_Canyon_Comanche_Natl_Park2.jpg&w=3840&q=75 "Vogel Canyon, Comanche National Grassland, courtesy of Colorado.com")
# 
# <embed>
#     <img
#         src="../img/map_of_comanche_national_grassland.png"
#         alt=" 'Map of the scattered 443,765 acres of Comanche National 
#         Grassland (green) in southeastern Colorado. State land is blue and 
#         Bureau of Land Management land is yellow. The northwestern Timpas 
#         Unit is in Otero and Las Animas counties.The southeastern Carrizo Unit 
#         is in Baca and Las Animas counties' courtesy of Donald L. Hazlett, 2004,
#         page 1)" 
#         height="700px"/>
#     <figcaption aria-hidden="true">
#         "Map of the scattered 443,765 acres of Comanche National 
#         Grassland (green) in southeastern Colorado. State land is blue and 
#         Bureau of Land Management land is yellow. The northwestern Timpas 
#         Unit is in Otero and Las Animas counties.The southeastern Carrizo Unit 
#         is in Baca and Las Animas counties"
#         (<span 
#             class="citation">
#              Hazlett, 2004, page 1
#          </span>)
#     </figcaption>
# </embed>
# 
# The Comanche National Grassland "is named in honor of the Comanche tribe. 
# The Comanche name is believed to be derived from Komontcia, a Ute word that
# means 'People Who Fight Us All the Time' (Pritzker 2000). This nickname, 
# assigned by the Utes, reflects the Comanche reputation among other tribes 
# and among pioneers as fierce fighters" (Hazlett 2004, page 3). Between the 
# 18th and 19th centuries there were many 'claims' to this land where the 
# Comanche National Grassland currently is situated by absentee ownership 
# of the U.S., France, and Mexico, but by 1853 the Comanche were 'asked' 
# (forced) to sign a 'treaty', which ceded any remaining lands that were 
# not previously ceded (taken)(Hazlett 2004, pages 1-4). Thus the 
# "remaining Comanche groups in southeastern Colorado had lost their homeland 
# and lifestyle" (Hazlett 2004, page 3).
# 
# After the Dust Bowl resulted in much abandoned farmland the "National
# Industrial Act and Emergency Relief Appropriations Act that passed 
# Congress in 1933 and 1935 gave the Federal government the authority 
# to purchase failed crop lands" (Worster 2004). It wasn't until 1954 
# that "the administration of these lands was transferred to the 
# USDA-Forest Service" (Hazlett 2004, page 4). In 1960 the Comanche
# National Grassland was created to be managed by the USDA-Forest Service 
# to conserve "the natural resources of grass, water and wildlife habitat(s)" 
# and protect prehistoric and historic, cultural and natural assets 
# (U.S. Department of Agriculture , U.S. Forest Service 2014). It consists 
# of around 440,000 acres of discontinuous land that is located in the 
# corner of southeastern Colorado (U.S. Department of Agriculture , U.S. 
# Forest Service 2014). This grassland is situated between three different 
# Colorado counties: Baca, Otero, and Las Animas and can be further identified 
# as two distinct units the *Carrizo Unit* south and west of Springfield 
# and the *Timpas Unit* south of La Junta" (Hazlett 2004, page 4).
# 
# The Comanche National Grassland is within what most plant geographers 
# classify as the 'North American Prairie Province' (Hazlett 2004, page 5). 
# Rockey Mountain junipers grow in 'rocky outcrops' which "are areas within 
# the open steppe, such as hilltops, where erosion has exposed a rocky surface 
# or barren (Hazlett 2004, page 8 and 17). A “barren” is defined here, in 
# a broad sense, as a sparsely vegetated exposed bedrock of shale, shale-derived 
# soils, chalk, or limestone soils with microorganisms in a calcite matrix 
# (Kelso et al. 2003). They also grow in the Comanche National Grassland in 
# 'shaded rock canyons and ravines' which are characterized as "the steep, rugged 
# relief areas that comprise the rocky cliffs, rock slicks, and shaded ledges 
# in the major canyons... Included here are hills with large boulders 
# and steep ridges... The greater water availability along cliff faces is 
# complemented by less evaporation due to greater amounts of shade. This is a 
# habitat of deep water percolation and occasional shade" (Hazlett 2004, page 
# 8 and 17). Due to the vast expanse of land the Comanche National Grassland 
# occupies (over 100 miles from one end to the other) the "Annual rainfall amounts 
# on the Comanche National Grassland have a high degree of spatial and temporal 
# variation." (Hazlett 2004, pages 4-5). The closest weather center, Western 
# Regional Climate Center (WRCC), reports a 55-year annual average as 11.5 inches 
# ; however, this climate center is 4 miles away from La Junta, which is north 
# of Vogel Canyon (see map above, La Junta would be off the map to the north of 
# Vogel Canyon) (Hazlett 2004, pages 4-5). So, that percipitation number would 
# not be accurate or reflective of all of this National Grassland due to its size.
# 
# The [*Vascular plant species of the Comanche National Grassland in southeastern Colorado*](https://www.fs.usda.gov/rm/pubs/rmrs_gtr130.pdf) 
# by Donald Hazlett, 2004, was very comprehensive not only of vegetation, but also history, 
# geology, and climate and should be read if interested in more in depth context 
# information than can be provided there.
# 
# * ### Pawnee National Grassland
# 
# <embed>
#     <img
#         src="../img/pawnee_national_grassland_img.png"
#         alt=" Image of Pawnee National Grassland showing Rocky Mountain
#         Juniper Trees courtesy of Donald L. Hazlett, 1998)" 
#         height="650px"/>
#             <figcaption aria-hidden="true">
#              Image of Pawnee National Grassland showing Rocky Mountain
#              Juniper Trees - courtesy of Hazlett, Donald L. (1998).
#                 (<span 
#                  class="citation">
#                  Hazlett 1998, page 'cover page'
#                 </span>)
#             </figcaption>
#     <img
#         src="../img/map_of_pawnee_national_grassland.png"
#         alt="Image of Pawnee National Grassland showing Rocky Mountain
#         Juniper Trees courtesy of Donald L. Hazlett, 1998)" 
#         height="500px"/>
#             <figcaption aria-hidden="true">
#              Map of Pawnee National Grassland.
#                 (<span 
#                  class="citation">
#                  Hazlett 1998, page 2
#                 </span>)
#             </figcaption>
# </embed>
# 
# There is a lack of information available on this particular grassland as it 
# relates to First Peoples, and most context information I am finding about the 
# history of this grassland starts with settlers or pioneers 'settling' the land 
# which feeds the Euro American perspective on history and excluding Indigenous 
# perspectives and experiences of this land. What little information I did find 
# was in someone's blog stating that "these lands were the home of the Arapaho 
# and Cheyenne, who were forcibly removed in the 1880s to allow white settlers 
# to establish homesteads and farm the land" (Diana 2021). However, there was 
# no source given as to where this blogger found that information (Diana 2021). 
# I acknowledge that much more research is needed for the historic and cultural 
# context that does not start with pioneers. 
# 
# That being said, settlers in the mid 19th century used this land for 
# cow grazing due to the aird nature of the land being difficult to grow 
# crops (Rhoads n.d.). Later in the late 19th century through the early 
# 20th century there were waves of newcomers and continued use of land for 
# grazing, but similar to the Comanche National Grassland the after effects 
# of the Dust Bowl lead to the U.S. Forest Service mangaing the area in 1954 
# then getting permanent control in 1960 (Rhoads n.d.). The Pawnee National 
# Grassland differs from the Comanche in that it has over 200 avtive oil and gas 
# leases on the grassland that are managed by the Bureau of Land Management, and 
# the U.S. Forest Service "specifies the revegetation procedures to be followed 
# by the private operators while conducting their exploration, drilling and 
# production activities" (Rhoads n.d.). This land is about 10% of the 
# Pawnee National Grassland and is labeled (CPER - Central Plains Experimental 
# Range) on the map above (Hazlett 1998, page 2). It should be noted that 
# the data used for the study area -
# [U.S. National Grassland Units](https://data-usfs.hub.arcgis.com/datasets/usfs::national-grassland-units-feature-layer/explore?location=39.118879%2C-104.194688%2C7.05), 
# include this private land and that should be kept in mind with any analyses.
# 
# The Pawnee National Grassland consists of around 193,000 acres of discontinuous 
# land (less than half the size of the Comanche National Grassland) that is 
# located in northeastern Colorado, Weld County, on the border of Wyoming (Hazlett 
# 1998, pages 1-2). This grassland is within the Central Shortgrass Prairie 
# Ecoregion and "geomorphic sections of the Great Plains province known 
# as the High Plains and the Colorado Piedmont (Trimble 1980)" (Hazlett 1998, 
# pages 3-4). Rocky Mountain junipers grow in similar areas in this grassland 
# as they do in the Comanche National Grassland - in cliffs and ravines (Hazlett 
# 1998, page 7). The climate here is affected by "continentality, air masses, 
# and mountain barrier" of the Rocky Mountains which block marine polar air masses 
# from the west contributing to the drier conditions on the Great Plains (Hazlett 
# 1998, page 1). The 21-year average annual rainfall, measured between 1969-1989, 
# was 12.6 inches which is similar to the annual average from the Comanche National 
# Grassland (11.5 inches) (Hazlett 1998, page 1). However, this data is out of 
# date considering the current increased impact of climate change.
# 
# 
# The [*Vascular plant species of the Pawnee National Grassland*](https://www.fs.usda.gov/rm/pubs/rmrs_gtr017.pdf) 
# by Donald Hazlett, 1998, was also comprehensive not only of vegetation, but also  
# geology, soils, and climate and should be read if interested in more in depth 
# context information than can be provided hhere. While this publication is missing 
# the historic context related to the site, unlike his later report on the Comanche 
# National Grassland, it is still very detailed and worth the read!
# 
# ### Citations
# 
# * Diana. 2021. “Colorado Destinations: Pawnee National Grassland.” 
# Handstands around the World. June 22, 2021. 
# https://handstandsaroundtheworld.blog/2021/06/21/colorado-destinations-pawnee-national-grassland/.
# 
# * Hazlett, Donald. 1998. “Vascular Plant Species of the Pawnee National 
# Grassland.” U.S. Department of Agriculture, U.S. Forest Service. 
# https://www.fs.usda.gov/rm/pubs/rmrs_gtr017.pdf.
# 
# * Hazlett, Donald L. 2004. “Vascular Plant Species of the Comanche 
# National Grassland in Southeastern Colorado.” U.S. Department of 
# Agriculture, U.S. Forest Service. 
# https://www.fs.usda.gov/rm/pubs/rmrs_gtr130.pdf.
# 
# * History Colorado. n.d. “History Colorado Anti-Racism Work: Grounding 
# Virtues | Goals | Accountability.” History Colorado. Accessed March 1, 2025. 
# https://www.historycolorado.org/sites/default/files/media/document/2020/Anti-Racism_Grounding_Virtues.pdf.
# 
# * Kelso, Sylvia, Nathan W Bower, Kirsten E Heckmann, Paul M Beardsley, 
# and Darren G Greve. 2003. “GEOBOTANY of the NIOBRARA CHALK BARRENS in 
# COLORADO: A STUDY of EDAPHIC ENDEMISM.” Western North American Naturalist 
# 63 (3): 299–313. https://doi.org/10.2307/41717298.
# 
# * Pritzker, Barry. 2000. “A Native American Encyclopedia : History, 
# Culture, and Peoples.” Internet Archive. Oxford ; New York : Oxford 
# University Press. 2000. 
# https://archive.org/details/nativeamericanen0000prit/page/n9/mode/2up.
# 
# * Rhoads, Dorothy, and Lee Rhoads. n.d. “Arapaho & Roosevelt National 
# Forests Pawnee National Grassland: Pawnee National Grassland History.” 
# Usda.gov. U.S. Department of Agriculture, U.S. Forest Service. Accessed 
# March 1, 2025. 
# https://www.fs.usda.gov/detail/arp/learning/history-culture/?cid=fsm91_058308.
# 
# * Trimble, Donald E. 1980. “GEOLOGICAL SURVEY BULLETIN 1493: The Geologic 
# History of the Great Plains.” Washington: U.S. Department of the Interior, 
# U.S. Geological Survey. https://pubs.usgs.gov/bul/1493/report.pdf.
# 
# * U.S. Department of Agriculture , U.S. Forest Service. 2014. “Pike-San 
# Isabel National Forests & Cimarron and Comanche National Grasslands - Comanche 
# National Grassland.” Fs.usda.gov. U.S. Department of Agriculture , U.S. 
# Forest Service. 2014. https://www.fs.usda.gov/recarea/psicc/recarea/?recid=12409.
# 
# * Worster, Donald. 2004. Dust Bowl : The Southern Plains in the 1930s. 
# 25th Anniversary Edition. New York: Oxford University Press.
# (https://global.oup.com/ushe/product/dust-bowl-9780195174885?cc=us&lang=en&)

# %% [markdown]
# ### Time periods
# 
# In general when studying climate, we are interested in **climate
# normals**, which are typically calculated from 30 years of data so that
# they reflect the climate as a whole and not a single year which may be
# anomalous. So if you are interested in the climate around 2050, download
# at least data from 2035-2065.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>Select at least two 30-year time periods to compare, such as
# historical and 30 years into the future. These time periods should help
# you to answer your scientific question.</p></div></div>

# %% [markdown]
# # **Time Periods Chosen**
# 
# Two different time periods are chosen *'historical' (1970-1999)* and 
# *late 21st century* (2071-2099). These specific years match up with the 
# individual file start and end years of the MACAv2 data. This route was chosen 
# because it will aid in the analysis of suitable habitats for the Rocky 
# Mountatin Juniper in both of the study areas chosen (Comanche and Pawnee 
# National Grasslands) that are roughly 100 years apart. Different time periods 
# provide insight into the past as well as possible future scenarios and the 
# two can be compared to look at the validity of the habitat suitability model 
# to be created for this project. Also, because the  detailed research sources 
# I found were from 1998 and 2004, those years fall within the 'historical' time 
# frame and the research can also be used to validate the habitat suitability 
# model regarding the 'historical' time period (Hazlett 1998 and 2004). The 
# MACA data can be downloaded as daily or monthly, and for the purposes of 
# this project, monthly will suffice, the amount of data the daily dataset 
# would have, is not necessary here. 
# 
# ### Citations:
# 
# * Hazlett, Donald. 1998. “Vascular Plant Species of the Pawnee National 
# Grassland.” U.S. Department of Agriculture, U.S. Forest Service. 
# https://www.fs.usda.gov/rm/pubs/rmrs_gtr017.pdf.
# 
# * Hazlett, Donald L. 2004. “Vascular Plant Species of the Comanche 
# National Grassland in Southeastern Colorado.” U.S. Department of 
# Agriculture, U.S. Forest Service. 
# https://www.fs.usda.gov/rm/pubs/rmrs_gtr130.pdf.
# 
# 

# %% [markdown]
# ### Climate models
# 
# There is a great deal of uncertainty among the many global climate
# models available. One way to work with the variety is by using an
# **ensemble** of models to try to capture that uncertainty. This also
# gives you an idea of the range of possible values you might expect! To
# be most efficient with your time and computing resources, you can use a
# subset of all the climate models available to you. However, for each
# scenario, you should attempt to include models that are:
# 
# -   Warm and wet
# -   Warm and dry
# -   Cold and wet
# -   Cold and dry
# 
# for each of your sites.
# 
# To figure out which climate models to use, you will need to access
# summary data near your sites for each of the climate models. You can do
# this using the [Climate Futures Toolbox Future Climate Scatter
# tool](https://climatetoolbox.org/tool/Future-Climate-Scatter). There is
# no need to write code to select your climate models, since this choice
# is something that requires your judgement and only needs to be done
# once.
# 
# If your question requires it, you can also choose to include multiple
# climate variables, such as temperature and precipitation, and/or
# multiple emissions scenarios, such as RCP4.5 and RCP8.5.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Choose at least 4 climate models that cover the range of possible
# future climate variability at your sites. How did you choose?</p></div></div>

# %% [markdown]
# # **Climate Models Chosen: Reasoning**
# Constant variables/scenarios of - RCP 8.5 as the Emission Scenario, 
# Climate Variables of precipitation and temperature minimum (°F). 
# The RCP8.5 Emission Scenario was chosen because that is the worst 
# case scenario for emissions and given that these two grasslands are 
# geographically close, I am interested in the most extreme emissions 
# scenario to see if that helps with the habitat model picking up on 
# differences that could be compared in the two grasslands. Those climate 
# varibales were chosen because that is information I could find about 
# the Rocky Mountain Juniper's growth conditions. In theory, if available, 
# I would be interested in the temperature max it can tolerate as well, 
# but that couldn't be found given the research sources I am using.
# 
# Using the Future Climate Scatter from the Climate Toolbox I chose 4
# climate models for each time period that was previously decided upon 
# (University of California MERCED n.d., "Climate Toolbox").
# Within each time period what I am trying to do is pick climate models 
# where the two warm scenarios are close in temperature, the two cold's 
# are close in temperature, the precipitation for dry is close, the 
# precipitation for wet is close. The goal of that is to have as little 
# variation within the time period, so more can hopefully be explained 
# by the climate model rather than the variation in the temp or precipitation.
# A full list of the specific climate models are available
# [here](https://climate.northwestknowledge.net/MACA/GCMs.php), and this 
# website also provides information on how the climate models are downscaled 
# for better resolution (University of California MERCED. n.d. 
# “CMIP5 GCMs and MACA Statistical Downscaling Method.)
# 
# ### Citations:
# 
# * University of California MERCED. n.d. “CMIP5 GCMs and MACA Statistical 
# Downscaling Method.” Northwestknowledge.net. University of California 
# MERCED. Accessed March 1, 2025. 
# https://climate.northwestknowledge.net/MACA/GCMs.php.
# 
# * University of California MERCED. n.d. “Climate Toolbox: Future Climate 
# Scatter.” Climatetoolbox.org. University of California MERCED. Accessed 
# March 1, 2025. https://climatetoolbox.org/tool/Future-Climate-Scatter.
# 

# %% [markdown]
# ## **Climate Models for the Comanche National Grassland**
# 
# ### **Time Period - 2071-2099**
# -   Warm and wet: CanESM-2 - 8.9in precipitation,  28.6(°F) temp min
# -   Warm and dry: IPSL-CM5A-MR - 3.3in precipitation, 29 (°F) temp min
# -   Cold and wet: MRI-CGCM3 - 8.4in precipitation,  23.5(°F) temp min
# -   Cold and dry: bcc-csm1-1-m - 4.8in precipitation,  25.4(°F) temp min
# 
# ### **Time Period - Historical 1970-1999**
# -   Warm and wet: CanESM2 - 7.9in precipitation, 19(°F) temp min
# -   Warm and dry: CCSM4 - 7.0in precipitation, 18.9(°F) temp min
# -   Cold and wet: HadGEM2-CC365 - 7.5in precipitation, 18.2(°F) temp min
# -   Cold and dry: inmcm4- 6.4in precipitation, 18.6(°F) temp min
# 

# %% [markdown]
# ## **Climate Models for the Pawnee National Grassland**
# 
# ### **Time Period - 2071-2099**
# -   Warm and wet: CanESM2 - 6.8in precipitation, 26.8(°F) temp min
# -   Warm and dry: HadGEM2-CC365 - 3.6in precipitation,  27.9(°F) temp min
# -   Cold and wet: GFDL-ESM2G - 7.2in precipitation, 21.9(°F) temp min
# -   Cold and dry: CSIRO-Mk3-6-0 - 4.2in precipitation, 22.5(°F) temp min
# 
# ### **Time Period - Historical 1970-1999** 
# -   Warm and wet: BNU-ESM - 6.7in precipitation, 15.5(°F) temp min
# -   Warm and dry: bcc-csm1-1-m - 6.1in precipitation, 15.4(°F) temp min
# -   Cold and wet: IPSL-CM5A-MR - 6.9in precipitation,  14.9(°F) temp min
# -   Cold and dry: bcc-csm1-1 - 6.1in precipitation,  14.8(°F) temp min

# %% [markdown]
# ## STEP 2: DATA ACCESS

# %% [markdown]
# ## Imports and Set Up

# %%
# Set Up Analysis Part 1 of 2

## Import packages that will help with...

# Reproducible file paths
import os # Reproducible file paths and basic formatting
from glob import glob  # returns list of paths
import pathlib # Find the home folder
import time # formatting time
import warnings # Filter warning messages
import zipfile # Work with zip files
import re # Compile regular expressions

# Work with tabular, vector, and raster data
import cartopy.crs as ccrs # CRSs (Coordinate Reference Systems)
import earthaccess # Access NASA data from the cloud
import geopandas as gpd # work with vector data
import geoviews as gv # holoviews extension for data visualization
import holoviews as hv # be able to save hvplots
import hvplot.pandas # Interactive tabular and vector data
import hvplot.xarray # Interactive raster
from math import floor, ceil # working with bounds, floor rounds down ciel rounds up
import matplotlib.pyplot as plt # Overlay pandas and xarry plots, Overlay raster and vector data
import numpy as np # numerical computing
import pandas as pd # Group and aggregate
import rioxarray as rxr # Work with geospatial raster data
from rioxarray.merge import merge_arrays # Merge rasters
import xarray as xr # Adjust images
import xrspatial # calculate slope

# import to visualize progress of iterative operations
from tqdm.notebook import tqdm 

# Suppress third party warnings - 'ignore'
warnings.simplefilter('ignore')

# %%
# Set Up Analysis Part 2 of 2

# Define and create the project data directory
part_2_hab_suit_data_dir = os.path.join(
    pathlib.Path.home(),
    'earth-analytics',
    'data',
    'part_2_habitat_suitability'
)
os.makedirs(part_2_hab_suit_data_dir, exist_ok=True)

# Call the data directory to check its location
part_2_hab_suit_data_dir

# %% [markdown]
# ## 1. Define Study Areas - USFS National Grassland Units
# ## (Comanche and Pawnee National Grasslands)

# %%
# Download USFS National Grasslands Units Data Part 1 of 1

# Define info for USFS National Forests download
usfs_grasslands_url = (
    "https://data.fs.usda.gov/geodata/edw/"
    "edw_resources/shp/S_USA.NationalGrassland.zip"
)
# Create directory and path for grassland data
usfs_grasslands_dir = os.path.join(
    part_2_hab_suit_data_dir, 'usfs_grasslands')
os.makedirs(usfs_grasslands_dir, exist_ok=True)
usfs_grasslands_path = os.path.join(
    usfs_grasslands_dir, 'usfs_grasslands.shp')

# Only download once - conditional
if not os.path.exists(usfs_grasslands_path):
    usfs_grasslands_gdf = gpd.read_file(usfs_grasslands_url)
    usfs_grasslands_gdf.to_file(usfs_grasslands_path)

# Load from file
usfs_grasslands_gdf = gpd.read_file(usfs_grasslands_path)

# %% [markdown]
# ### Plot Site Boundary Data

# %%
# Create plots of each study area, Part 1 of 2

# Create new variable for comanche gdf / select data from Comanche National Grassland
comanche_grassland_gdf = (
    # Pull comanche through the grassland name column
    usfs_grasslands_gdf[usfs_grasslands_gdf.GRASSLANDN=='Comanche National Grassland']
)
# Create new variable for plot in order to save it later
comanche_grassland_site_boundary = (
 # Create an interactive site map using hvplot 
 comanche_grassland_gdf.hvplot(
     geo=True,
     tiles='EsriImagery',
     title='Comanche National Grassland - Site Map',
     fill_color='lightblue', line_color='blue', line_width=2.5,
     width=600, height=400
 )
)

# Save the plot as html to be able to display online
hv.save(comanche_grassland_site_boundary, 'comanche_grassland_site_boundary.html') 
# Display the plot 
comanche_grassland_site_boundary

# %% [markdown]
# ## Comanche National Grassland has discontinuous boundaries - this reflects
# ## how the grasslands were purchased
# 
# The Comanche National Grassland has discontinuous boundaries. 
# This reflects how the land was purchased by the federal U.S. government 
# as parcels or groups of parcels (Worster 2004). A parcel is an area with 
# a distinct boundary and a unique identifier; they are drawn as polygons 
# and are often rectangles or squares unless defined by a natural boundary 
# like a river or lake. Parcels can be arbitrary or of different sizes, 
# ; parcels can be combined or divded into new parcels per local regulations. 
# Due to the variable nature of what can be done with the administrative 
# boundary of a parcel, is why there are so many separated parcels in the 
# grassland of different sizes and shapes. The USFS may not own the land 
# surrounding or in between these boundaries. The habitat suitability model 
# being built for this project may be applied on a very broad level to the 
# surrounding areas; however, the habitat suitability model is only pertaining 
# to land within these administrative boundaries. While the administrative 
# boundaries may seem somewhat arbitrary, the findings of the habitat suitability 
# model can help guide decisions made about the grassland itself, which the 
# USFS owns and makes decisons on (sometimes in conjunction with other agencies
# or entities).
# 
# The Comanche National Grassland is separated in two areas or units. One unit, 
# the Timpas Unit, is slightly north of the Carrizo Unit (Hazlett 2004). This 
# National Grassland has a range of longitude of -104.2 through 102.2. 
# 
# This information will lay the ground for further layers and the habitat 
# suitabaility model later.
# 
# ### Citations:
# 
# * Hazlett, Donald L. 2004. “Vascular Plant Species of the Comanche 
# National Grassland in Southeastern Colorado.” U.S. Department of 
# Agriculture, U.S. Forest Service. 
# https://www.fs.usda.gov/rm/pubs/rmrs_gtr130.pdf.
# 
# * Worster, Donald. 2004. Dust Bowl : The Southern Plains in the 1930s. 
# 25th Anniversary Edition. New York: Oxford University Press.
# (https://global.oup.com/ushe/product/dust-bowl-9780195174885?cc=us&lang=en&)

# %%
# Create plots of each study area, Part 2 of 2

# Create an interactive site map, select data from Pawnee National Grassland
pawnee_grassland_gdf = (
    # Pull pawnee through the grassland name column
    usfs_grasslands_gdf[usfs_grasslands_gdf.GRASSLANDN=='Pawnee National Grassland']
)
# Create new variable for plot in order to save it later
pawnee_grassland_site_boundary = (
    # Create an interactive site map using hvplot 
    pawnee_grassland_gdf.hvplot(
        geo=True, 
        tiles='EsriImagery',
        title='Pawnee National Grassland - Site Map',
        fill_color='lightblue', line_color='blue', line_width=2,
        width=700, height=400
    )
)
# Save the plot as html to be able to display online
hv.save(pawnee_grassland_site_boundary, 'pawnee_grassland_site_boundary.html') 

# Display the plot 
pawnee_grassland_site_boundary

# %% [markdown]
# ## Pawnee National Grassland has discontinuous boundaries - this reflects
# ## how the grasslands were purchased
# 
# Similar to the Comanche National Grassland, the Pawnee National Grassland
# also has discontinuous boundaries. This reflects how the land was purchased 
# by the federal U.S. government as parcels or groups of parcels (parcels 
# are described in the above description for the Comanche National Grassland) 
# (Worster 2004). The USFS may not own the land surrounding or in between 
# these boundaries. While the habitat suitability model being built for this 
# project may be applied on a very broad level to the surrounding areas, the 
# habitat suitability model is only pertaining to land within these administrative 
# boundaries.While the administrative boundaries may seem somewhat arbitrary, 
# the findings of the habitat suitability model can help guide decisions made 
# about the grassland itself, which the USFS owns and makes decisons on (sometimes 
# in conjunction with other agencies).
# 
# The Pawnee National Grassland is also separated in two areas. The difference 
# with these is that both areas are somewhat paralell to each other, with the 
# right one extending slightly more north. It has similar range of longitude 
# to the Comanche National Grassland, Pawnee's is -104.9 through -103.4 and 
# Comanche's is -104.2 through 102.2. However Pawnee National Grassland is 
# roughly 2 degrees latitude further north than Comanche. Both grasslands 
# chosen span roughly 1 degree in latitude.
# 
# This information will lay the ground for further layers and the habitat 
# suitabaility model later.
# 
# ### Citations:
# 
# * Worster, Donald. 2004. Dust Bowl : The Southern Plains in the 1930s. 
# 25th Anniversary Edition. New York: Oxford University Press.
# (https://global.oup.com/ushe/product/dust-bowl-9780195174885?cc=us&lang=en&)

# %% [markdown]
# # Soil data
# 
# The [POLARIS dataset](http://hydrology.cee.duke.edu/POLARIS/) is a
# convenient way to uniformly access a variety of soil parameters such as
# pH and percent clay in the US. It is available for a range of depths (in
# cm) and split into 1x1 degree tiles.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Write a <strong>function with a numpy-style docstring</strong> that
# will download POLARIS data for a particular location, soil parameter,
# and soil depth. Your function should account for the situation where
# your site boundary crosses over multiple tiles, and merge the necessary
# data together.</p>
# <p>Then, use loops to download and organize the rasters you will need to
# complete this section. Include soil parameters that will help you to
# answer your scientific question. We recommend using a soil depth that
# best corresponds with the rooting depth of your species.</p></div></div>

# %% [markdown]
# # 2. Wrangle the Raster Data (3 layers)
# # Part 1: POLARIS dataset - download 1 soil variable

# %%
# Process POLARIS Raster Image Part 1 of 2

# Create function with description to process raster images
def process_image(url, soil_prop, soil_stat, soil_depth, bounds_gdf):
    """
    Load, crop, and scale raster images for multiple sites.

    Parameters
    ----------
    url: str
      URL or path for raster files.
    soil_prop: str
      Soil property (e.g., "sand", "clay", etc.)
    soil_stat: str
      Soil statistic (e.g., "mean", "median", etc.)
    soil_depth: str
      Soil depth (e.g., "30-60cm", "60-100cm", etc.)
    bounds_gdf: gpd.GeoDataFrame
      Area of interest to crop to.
    site_names: list
      List of site names to be used as dictionary keys.
    Returns
    -------
   merged_da: rxr.DataArray
      Processed rasters 
    """

    # Iterate through the list of bounding GeoDataFrames (areas of interest)
    #for site_name, bounds_gdf in zip(site_names, bounds_gdfs):

    # Get the study bounds
    bounds_min_lon, bounds_min_lat, bounds_max_lon, bounds_max_lat = (
    bounds_gdf
    .to_crs(4326)
    .total_bounds 
    )

    # List to store cropped DataArrays for the current site
    da_list = []
    
    # Loop through bounding box coordinates
    for min_lon in range(floor(bounds_min_lon), ceil(bounds_max_lon)):
      for min_lat in range(floor(bounds_min_lat), ceil(bounds_max_lat)):

        # Format the URL with the current coordinates and other parameters
        formated_url = (
          url.format( 
              soil_prop = soil_prop, 
              soil_stat = soil_stat, 
              soil_depth = soil_depth,
              min_lat=min_lat , max_lat=min_lat+1,
              min_lon=min_lon, max_lon=min_lon+1 )
        )

        # Connect to the raster image
        da = rxr.open_rasterio(
        formated_url, 
        mask_and_scale=True
        ).squeeze()
        
        # Crop the raster image to the bounds of the study area
        cropped_da = (
        da.rio.clip_box(bounds_min_lon, bounds_min_lat, bounds_max_lon, bounds_max_lat)
        )

        # Append the cropped DataArray to the list
        da_list.append(cropped_da)   

    # Merge the cropped DataArrays for this site
    merged_da = merge_arrays(da_list)

    return merged_da

# %%
# Process POLARIS raster image part 2 of 2
# Test the function by defining variables to pass as arguments to the function

# Set the site parameters
# soil variables
soil_prop = 'ph'
soil_stat = 'mean'
soil_depth = '60_100'
# set up url template
soil_url_template = (
            "http://hydrology.cee.duke.edu"
            "/POLARIS/PROPERTIES/v1.0"
            "/{soil_prop}"
            "/{soil_stat}"
            "/{soil_depth}"
            "/lat{min_lat}{max_lat}_lon{min_lon}{max_lon}.tif"
            )

# output_directory - create data dir for polaris data 
polaris_dir= os.path.join(part_2_hab_suit_data_dir, 'polaris')
os.makedirs(polaris_dir, exist_ok=True)

# Call the directory to check the location
polaris_dir

# %%
# List of the study areas and corresponding GeoDataFrames to use in for loop
study_areas = (
    ("Comanche National Grassland", comanche_grassland_gdf),
    ("Pawnee National Grassland", pawnee_grassland_gdf)
)    

# Create a list to store the processed data for each study area
polaris_processed_da_list = []

# Loop through each study area and process the image
for area_name, area_gdf in study_areas:
    processed_data = process_image(
        soil_url_template,
        soil_prop, soil_stat, soil_depth,
        area_gdf
    )
    polaris_processed_da_list.append(processed_data)

# Call the list to make sure it worked/looks right
polaris_processed_da_list

# %%
# Plot Pawnee to make sure it works/ looks right

# Set the figure size (width, height in inches)
plt.figure(figsize=(8, 6))

polaris_processed_da_list[0].plot(
    cbar_kwargs={"label": "pH"},
    robust=True,
    )
comanche_grassland_gdf.to_crs(polaris_processed_da_list[0].rio.crs).boundary.plot(
    ax=plt.gca(),
    color='white').set(
        title='Comanche National Grassland - pH',
        xlabel='Longitude', 
        ylabel='Latitude',
    )

# Save the figure
plt.savefig("polaris_comanche_plot.png", dpi=300)

# Display the plot        
plt.show()

# %% [markdown]
# ## Comanche Grassland - pH - plotted correctly - the slightly acidic soil 
# ## areas appear to be outside the grassland visually. Full pH scale plotted 
# ## would work for the Rocky Mountain Juniper
# 
# The lower unit, Carrizo, is mostly in the upper range of the pH scale bar, 
# while the upper unit, Timpas, if in the middle range of the scale bar. This 
# range of pH plotted, is fully within the acceptable range for a Rocky Mountain 
# Juniper. There is a larger range of pH here than with Pawnee. Without adding 
# additional context or other data it is hard to draw further conclusions so 
# other raster sets will be downloaded next to eventually build a habitat 
# suitability model.

# %%
# Plot Pawnee to make sure it works/ looks right

# Set the figure size (width, height in inches)
plt.figure(figsize=(8, 5))

polaris_processed_da_list[1].plot(
    cbar_kwargs={"label": "pH"},
    robust=True,
    )

pawnee_grassland_gdf.to_crs(polaris_processed_da_list[1].rio.crs).boundary.plot(
    ax=plt.gca(),
    color='white').set(
        title='Pawnee National Grassland - pH',
        xlabel='Longitude', 
        ylabel='Latitude',
    )

# Save the figure
plt.savefig("polaris_pawnee_plot.png", dpi=300)

# Display the plot        
plt.show()

# %% [markdown]
# ## Pawnee Grassland - pH - plotted correctly - left area of grassland 
# ## appears to have slightly lower pH areas 
# 
# While the left area has slightly lower pH on the scale, pH of 7 and 8 is still 
# considered netural to alkaline soil overall and is within the range that would 
# be acceptable to a rocky mountain juniper. There is a full degree of longitude 
# difference between the left and right areas of this grassland, so it makes sense 
# that there is some variability in this range without adding additional context or 
# other data it is hard to draw further conclusions so other raster sets will be 
# downloaded next to eventually build a habitat suitability model.

# %% [markdown]
# # Topographic data
# 
# One way to access reliable elevation data is from the [SRTM
# dataset](https://www.earthdata.nasa.gov/data/instruments/srtm),
# available through the [earthaccess
# API](https://earthaccess.readthedocs.io/en/latest/quick-start/).
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Write a <strong>function with a numpy-style docstring</strong> that
# will download SRTM elevation data for a particular location and
# calculate any additional topographic variables you need such as slope or
# aspect.</p>
# <p>Then, use loops to download and organize the rasters you will need to
# complete this section. Include topographic parameters that will help you
# to answer your scientific question.</p></div></div>
# 
# > **Warning**
# >
# > Be careful when computing the slope from elevation that the units of
# > elevation match the projection units (e.g. meters and meters, not
# > meters and degrees). You will need to project the SRTM data to
# > complete this calculation correctly.

# %% [markdown]
# # 2. Wrangle the Raster Data (3 layers)
# # Part 2:  Elevation Data using SRTM

# %%
# Download Raster data through earthaccess Part 1 of 1
# Create function to login and search earthaccess, then download results

def download_earthaccess_data(bounds_gdfs, dir, short_name):
    """
    Download raster data from EarthAccess for the given areas of interest (bounding GeoDataFrames).
    
    Args:
    - chosen_grasslands_bounds_gdfs (list): List of GeoDataFrames defining the areas of interest.
    - elevation_dir (str): Directory where the downloaded files will be saved.
    - short_name (str): The short name of the dataset to search for (e.g., "SRTMGL1").

    Returns:
     - list: List of unique file paths that match the pattern for downloaded raster files.
    """
    # Login to earthaccess
    earthaccess.login(strategy="interactive", persist=True)

    # Initialize a list to store the downloaded files
    all_files = []

    # Iterate through the list of bounding GeoDataFrames (areas of interest)
    for bounds_gdf in bounds_gdfs:
        
        # Set bounds from the GeoDataFrame
        bounds = tuple(bounds_gdf.total_bounds)

        # Search EarthAccess for SRTM data within the bounds
        results = earthaccess.search_data(
            short_name=short_name,
            bounding_box=bounds
        )
        
        # Download the EarthAccess results
        files = earthaccess.download(results, dir)

        # Collect all downloaded files that match the pattern
        files = glob(os.path.join(dir, '*hgt.zip'))
        
        # Add the found files to the overall list
        all_files.extend(files)

    # Remove duplicates while preserving the order
    unique_raster_files = []
    seen = set()
    for file in all_files:
            if file not in seen:
                unique_raster_files.append(file)
                seen.add(file)

    return unique_raster_files


# %%
# Download Raster data through earthaccess Part 1 of 1

# Test the function above (download_earthaccess_data)
# First define variables to pass as arguments to the function

# bounds gdfs
chosen_grasslands_bounds_gdfs = [comanche_grassland_gdf, pawnee_grassland_gdf]

# Create data dir 
elevation_dir= os.path.join(part_2_hab_suit_data_dir, 'srtm')
os.makedirs(elevation_dir, exist_ok=True)
# call the variable to check location
elevation_dir

# Select and set the short_name of dataset wanted
short_name="SRTMGL1"

# Test the function
srtm_files = download_earthaccess_data(chosen_grasslands_bounds_gdfs, elevation_dir, short_name)

# Check the result
srtm_files

# %%
# Group SRTM files by coordinates Part 1 of 2

# Create function to group files by boundary 
"""The last function resulted in the files for SRTM being downloaded, 
but they need to be sorted based on the coordinates of the grasslands. 
I chose to do this as separate function to do things piece by piece. Being 
new to earth data science it's easier for me to understand these as separate 
ideas or groups of ideas, plus this lets me see that each step works on its 
own and I could use these functions separately on a different project or 
projects"""

def group_files_by_bounds(files, bounds):
    """
    Group the files by the bound areas based on the file coordinates.

    Args:
    - files (list): List of file paths for the downloaded raster files.
    - bounds (dict): Dictionary where the key is the bound name, 
      and the value is the tuple of (min_lat, max_lat, min_lon, max_lon).

    Returns:
    - dict: A dictionary with grassland names as keys and lists of files as values.
    """
    # Initialize the dictionary to store grouped files
    grouped_files = {bound: [] for bound in bounds}

    # Regex to extract coordinates from the filename (e.g., "N36W105" from "N36W105.SRTMGL1.hgt.zip")
    coord_pattern = re.compile(r'([NS])(\d{2})([EW])(\d{3})')

    # Iterate through each file and extract the coordinates
    for file_path in files:
        # Get the filename from the full path
        filename = os.path.basename(file_path)
        
        # Use regex to extract coordinates from the filename
        match = coord_pattern.search(filename)
        if match:
            lat_dir, lat_deg, lon_dir, lon_deg = match.groups()
            lat = int(lat_deg) * (-1 if lat_dir == 'S' else 1)  # Convert to signed integer for latitude
            lon = int(lon_deg) * (-1 if lon_dir == 'W' else 1)  # Convert to signed integer for longitude

            # Check which grassland the coordinates belong to
            for bound_name, bound_coords in bounds.items():
                min_lat, max_lat, min_lon, max_lon = bound_coords
                # Check if the coordinates are within the bounds
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    grouped_files[bound_name].append(file_path)

    return grouped_files


# %%
# Group SRTM files by coordinates Part 2 of 2
# Test the group_files_by_bounds function

# Use SRTM files defined in last cell

# Define the bounds for each grassland (min_lat, max_lat, min_lon, max_lon)
grasslands_bounds = {
    'Comanche National Grassland': (35, 37, -106, -103),
    'Pawnee National Grassland': (39, 41, -106, -104)
}

# Group the files by grassland based on the coordinates
grouped_srtm_files = group_files_by_bounds(srtm_files, grasslands_bounds)

# Check the grouped files
grouped_srtm_files

# %%
# Create function with description to process srtm raster images
# Part 1 of 2
def process_image_list(url_list, chosen_buffer, bounds_gdf):
    """
    Load, crop, and scale a raster image 

    Parameters
    ----------
    url: file-like or path-like
      File accessor downloaded or obtained 
    chosen_buffer: float number
      Amount of degrees to extend past the bounds of the bounds_gdf 
    bounds_gdf: gpd.GeoDataFrame
      Area of interest to crop to

    Returns
    -------
    merged_da: rxr.DataArray
      Processed raster
    """   
        
    # List to store cropped DataArrays for the current site 
    da_list= []
      
    buffer= chosen_buffer

    for url in url_list:

        # Connect to the raster image
        da = rxr.open_rasterio(
          url, 
          mask_and_scale=True
          ).squeeze()
        
          # Get the study bounds
        bounds_min_lon, bounds_min_lat, bounds_max_lon, bounds_max_lat = (
          bounds_gdf
          .to_crs(da.rio.crs)
          .total_bounds 
          )

        # Crop the raster image to the bounds of the study area
        cropped_da = (
          da.rio.clip_box(bounds_min_lon-buffer, bounds_min_lat-buffer, bounds_max_lon+buffer, bounds_max_lat+buffer)
          )
        
        # Append the cropped DataArray to the list
        da_list.append(cropped_da)

    # Merge the cropped DataArrays for this site
    merged_da = (
      merge_arrays(da_list)
      )
        
    return merged_da

# %%
# Create function with description to process srtm raster images
# Part 2 of 2

# Use process_image_list function in for loop on grouped_srtm_files

# List of the study areas and corresponding GeoDataFrames to use in 
# for loop use 'study_areas' defined earlier

# List to store the srtm results
srtm_da_results = []

# Loop over each grassland and process the corresponding files
for grassland_name, gdf in study_areas:
    # Process the SRTM files for the current grassland
    result_da = process_image_list(grouped_srtm_files[grassland_name], .025, gdf)
    
    # Append the result to the list
    srtm_da_results.append(result_da)

# Call srtm_da_results to check that it worked
srtm_da_results

# %%
# Plot the processed raster on Comanche National Grassland

# Set the figure size (width, height in inches)
plt.figure(figsize=(8, 5))

srtm_da_results[0].plot(
    cbar_kwargs={"label": "Elevation (meters)"},
    robust=True,
    cmap='terrain',
)
# Overlay the boundary of the same study area
comanche_grassland_gdf.boundary.plot(ax=plt.gca(),
    color='black').set(
        title='Comanche National Grassland - Elevation ',
        xlabel='Longitude', 
        ylabel='Latitude',
        xticks=[],
        yticks=[] 
    )

# Save the figure
plt.savefig("srtm_comanche_plot.png", dpi=300)
# Display the figure
plt.show()

# %% [markdown]
# ## Comanche Grassland Elevation - plotted correctly, wide range in 
# ## elevation is seen, the lower half of this range is not within the 
# ## perferred range for Rocky Mountain Juniper
# 
# The elevation scale (1200-1800 meters), is roughly 4000 - 5900 feet in 
# elevation. The rocky mountain juniper's perferred range is 5000 to 
# 7500 feet (1524 meters to 2285 meters), so only the upper half of the 
# elevation color bar scale would be applicapable to the species chosen. 
# The lower right unit, Carrizo, has most of the right half of it visually 
# in the elevations that would not be acceptable, and the upper left unit, 
# Timpas, is mostly in the lower half of the elevation scale. So, it will be 
# interesting in further analysis if this is a limiting factor for 
# habitat suitability.

# %%
# Plot the processed raster on Pawnee National Grassland

# Set the figure size (width, height in inches)
plt.figure(figsize=(8, 4))

srtm_da_results[1].plot(
    cbar_kwargs={"label": "Elevation (meters)"},
    robust=True,
    cmap='terrain',
)
# Overlay the boundary of the same study area
pawnee_grassland_gdf.boundary.plot(ax=plt.gca(),
    color='black').set(
        title='Pawnee National Grassland - Elevation ',
        xlabel='Longitude', 
        ylabel='Latitude',
        xticks=[],
        yticks=[] 
    )

# Save the figure
plt.savefig("srtm_pawnee_plot.png", dpi=300)
# Display the figure
plt.show()

# %% [markdown]
# ## Pawnee Grassland Elevation - plotted correctly, smaller range in 
# ## elevation is seen comapred to Comanche, most of this 
# ## range is within the perferred range for Rocky Mountain Juniper
# 
# Almost all of the left unity would be in areas that are within the 
# suitable range for the Rocky Mountain Juniper. Part of the right unit 
# (left half) would be in areas that are within the suitable range for 
# the Rocky Mountain Juniper. Overall, visually this grassland, compared 
# to the Comanche has more areas within the suitable range for the species 
# chosen, however based on the context, this grassland has less overall 
# acerage or span in longitude and latitude and is smaller.

# %% [markdown]
# # **Calculate Slope**

# %%
def calculate_slope(da_results, epsg_in, epsg_out):
    """
    Calculate the slope for each raster in the provided list of data arrays, 
    and reproject to the desired output CRS.

    Parameters:
    ----------
    da_results : list
        List of xarray DataArrays (like SRTM results).
    epsg_in : int
        The EPSG code of the input CRS (EPSG:32613).
    epsg_out : int
        The EPSG code for the desired output CRS (EPSG:4326).

    Returns:
    --------
    slope_da_list : list
        List of xarray DataArrays, each containing the calculated slope, reprojected to the desired CRS.
    """
    # List to store the slope results
    slope_da_list = []

    # Iterate through each SRTM data array and calculate slope
    for result in da_results:

        # Reproject to the input CRS (if it's not already)
        proj_da = result.rio.reproject(epsg_in)

        # Calculate slope (in UTM projection)
        slope_da = xrspatial.slope(proj_da)

        # Reproject slope to the desired output CRS (e.g., EPSG:4326)
        slope_da_reprojected = slope_da.rio.reproject(epsg_out)

        # Append to the result list
        slope_da_list.append(slope_da_reprojected)

    return slope_da_list

# Test function for slope, use 32613 as the input CRS and 4326 as the output CRS
slope_results = calculate_slope(srtm_da_results, 32613, 4326)

slope_results

# Now slope_results contains the calculated slope reprojected to EPSG:4326
# this matches the grassland boundaries and the polaris data (and eventually 
# the climate model rasters as well)

# %%
# Comanche Test to make sure the slope function worked 
# by plotting

# Set the figure size (width, height in inches)
plt.figure(figsize=(8, 5))

# Plot Comanche
slope_results[0].plot(
    cbar_kwargs={"label": "Slope (degrees)"},
    cmap='terrain',
)
# Overlay the boundary of the same study area
comanche_grassland_gdf.to_crs(4326).boundary.plot(
    ax=plt.gca(),
    color='white').set(
        title='Comanche National Grassland - Caluclated Slope ',
        xlabel='Longitude', 
        ylabel='Latitude',
        xticks=[],
        yticks=[] 
    )

# Save the figure
plt.savefig("slope_comanche_plot.png", dpi=300)
# Display the figure
plt.show()

# %% [markdown]
# ## Comanche Caluculated Slope - plotted correctly, visually 
# ## there are some areas of degress slope 10-30 which 
# ## would potentially be areas that the Rocky Mountain Juiper is 
# ## commonly found 
# 
# They Rocky Mountain Juniper is commonly found in rocky canyons and 
# ravines, there is an assumed slope, without finiding specifc degrees, 
# would be around 30. While there doesn't visually appear to be many 
# areas of slope at 30 degrees, there seems to be some areas between 
# maybe 10 and 30 degrees within the grassland boundaries but due to 
# the outline of  boundary, it's hard to see specifically where these 
# areas might be. The habitat suitability mdoel should help with further 
# conclusions.
# 

# %%
# Pawnee Test to make sure the slope function worked 
# by plotting

# Set the figure size (width, height in inches)
plt.figure(figsize=(8, 4))

# Plot Pawnee (reprojected slope)
slope_results[1].plot(
    cbar_kwargs={"label": "Slope (degrees)"},
    cmap='terrain',
)

# Overlay the boundary of the same study area
pawnee_grassland_gdf.to_crs(4326).boundary.plot(
    ax=plt.gca(),
    color='white').set(
        title='Pawnee National Grassland - Calculated Reprojected Slope',
        xlabel='Longitude', 
        ylabel='Latitude',
        xticks=[],
        yticks=[]
    )

# Save the figure
plt.savefig("slope_pawnee_reprojected_plot.png", dpi=300)
# Display the figure
plt.show()

# %% [markdown]
# ## Pawnee Caluculated Slope - plotted correctly, visually 
# ## there are few areas of degress slope 10-30 which 
# ## would potentially be areas that the Rocky Mountain Juiper is 
# ## commonly found 
# 
# Compared to the Comanche Calculated Slope plot, this one appears 
# to visually have fewer areas that have a slope between 10-30 degrees.
# Of the areas that have a slope greater than 0 it seems to be between 
# 0 and 15 degrees, however it is difficult to draw conclusions visually 
# when the resolution is so high. The habitat suitability model should 
# help with further conclusions.
# 

# %%
# I was having issues in the fuzzy logic model and 
# I think this raster is the problem I want to make 
# sure they are aligned 

# Check dimensions of the SRTM and slope results
for idx, result in enumerate(srtm_da_results):
    print(f"SRTM result {idx} shape: {result.shape}")
    
for idx, result in enumerate(slope_results):
    print(f"Slope result {idx} shape: {result.shape}")

# %%
# So, they are different and need to be aligned
# Align both srtm and slop da's to the first raster
aligned_srtm_da = []
aligned_slope_results = []

# Align the SRTM rasters
for srtm in srtm_da_results:
    aligned_srtm, _ = xr.align(srtm, srtm_da_results[0], 
        join='outer')  # Align with first element in the list
    aligned_srtm_da.append(aligned_srtm)

# Align the slope results
for slope in slope_results:
    aligned_slope, _ = xr.align(slope, slope_results[0], 
        join='outer')  # Align with first element in the list
    aligned_slope_results.append(aligned_slope)

# Call the aligned slope results to make sure I can use that later
aligned_slope_results

# %% [markdown]
# # Climate model data
# 
# You can use MACAv2 data for historical and future climate data. Be sure
# to compare at least two 30-year time periods (e.g. historical vs. 10
# years in the future) for at least four of the CMIP models. Overall, you
# should be downloading at least 8 climate rasters for each of your sites,
# for a total of 16. **You will *need* to use loops and/or functions to do
# this cleanly!**.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Write a <strong>function with a numpy-style docstring</strong> that
# will download MACAv2 data for a particular climate model, emissions
# scenario, spatial domain, and time frame. Then, use loops to download
# and organize the 16+ rasters you will need to complete this section. The
# <a
# href="http://thredds.northwestknowledge.net:8080/thredds/reacch_climate_CMIP5_macav2_catalog2.html">MACAv2
# dataset is accessible from their Thredds server</a>. Include an
# arrangement of sites, models, emissions scenarios, and time periods that
# will help you to answer your scientific question.</p></div></div>

# %% [markdown]
# # 2. Wrangle the Raster Data (3 layers)
# # Part 3: MACA v2 THREDDS - download climate models chosen with a function

# %%
# Create function  that converts longitude that is in the 0-360 
# range, to the -180 to 180 range 
def convert_longitude(longitude):
    """ Convert logitude range from  0-360 to -180-180
        Parameters:
        -----------
        longitude = int
            longitude in the 0-360 format

        Returns:
        --------
        longitude in the -180 to 180 range 
    """
    return (longitude - 360) if longitude > 180 else longitude

# Create function to download MACAv2 data
def download_maca_data(climate_model, ensemble, variable_short, var_long_name, 
    scenario, start_year, end_year, spatial_domain):
    """
        Download MACAv2 data from a particular climate model (including ensemble), 
        emissions scenario, variable(s), spatial domain, and time frame

        Parameters:
        -----------
        climate_model = str
            acronym for climate model from CMIP5 GCMs
        ensemble = str
            specific to the climate model (r1i1p1 for all models except CCSM4 (r6i1p1))
        variable_short = str
            short form of climate model variable ('pr', 'tasmin', 'tasmax', etc.)
        var_long_name = str
            long form of variable name ('precipitation', 'air_temperature')
        scenario = str 
            emissions scenario (historical, rcp4.5, rcp8.5, etc.)
        start_year = int
            ideally the start year of the climate model file
        end_year = int
            ideally the end year of the last year of the climate file wanted
        spatial_domain = gdf
            site boundary (national grassland gdf)

        Returns:
        ---------
        maca_da_list: da list
            downloaded climate model rasters
    """
  
    # Define template url for MACA v2 download
    maca_url = (
        'http://thredds.northwestknowledge.net:8080/thredds/dodsC/MACAV2'
        f'/{climate_model}/macav2metdata_{variable_short}_{climate_model}_{ensemble}_'
        f'{scenario}_{start_year}_{end_year}_CONUS_monthly.nc')
    # Connect to the raster image - with error handling
    try:
        # Open the dataset from the URL
        maca_da = xr.open_dataset(maca_url).squeeze()[var_long_name]
    except Exception as e:
        raise ValueError(f"Failed to open MACAv2 data: {e}")

    # Ensure spatial_domain is a GeoDataFrame
    if not isinstance(spatial_domain, gpd.GeoDataFrame):
        raise TypeError("Spatial domain must be a GeoDataFrame")
    
    # Get the study bounds
    bounds = spatial_domain.to_crs(maca_da.rio.crs).total_bounds
    # Apply function convert_longitude to convert longitude
    maca_da = maca_da.assign_coords(
        lon = ('lon', 
        [convert_longitude(l) for l in maca_da.lon.values]))
    # Set spatial dimensions - need lon = x-axis and lat = y-axis.
    maca_da = maca_da.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    # Crop the raster image to the bounds of the study area(s)
    maca_da = maca_da.rio.clip_box(*bounds)

    return maca_da

# %%
# Input a set of parameters to see if the download works
download_maca_data('CCSM4','r6i1p1', 'pr', 'precipitation', 'rcp85', 
    2071, 2075, comanche_grassland_gdf)

# %%
# Create a dictionary mapping short names to long names
variable_names_mapping = {
    "pr": "precipitation",
    "tasmin": "air_temperature"
}
# Create function to download climate data for various scenarios/variables/etc.
def download_for_scenarios(scenarios, common_params, spatial_domain):
    """
    Downloads climate data for different scenarios and variables and stores them in a dictionary.

    Parameters:
    -----------
    comanche_scenarios : list of dict
        A list of dictionaries where each dictionary represents a climate scenario
        with model, ensemble, description, start and end years, and scenario type.
    common_params : dict
        A dictionary of common parameters (e.g., variables) that should be used for 
        all scenarios.
    spatial_domain : GeoDataFrame
        The spatial domain (e.g., a GeoDataFrame representing the region of interest, 
        like a boundary).
    
    Returns:
    --------
    dict
        A dictionary where the keys are scenario descriptions and variables, 
        and the values are the downloaded data.
    """
    # To store the results for each scenario
    maca_data_dict = {}
    # Use tqdm to show progress for the scenarios
    for scenario in tqdm(scenarios, desc="Downloading Scenarios", unit="scenario"):
        # Merge common parameters with the scenario-specific ones
        scenario_params = {**common_params, **scenario}
        # Use tqdm for the variables to show progress for each variable within each scenario
        for variable_short in tqdm(scenario_params['variable_short'], 
            desc=f"Downloading Variables for {scenario['description']}", unit="variable", leave=False):
            
            # Retrieve the long name from the mapping dictionary using the short name
            var_long_name = variable_names_mapping.get(variable_short)
            
            if not var_long_name:
                print(f"Warning: No long name found for variable short name {variable_short}")
                continue
            
            # Print for tracking which variable is being downloaded
            print(f"Downloading {scenario_params['description']} data for variable {var_long_name}...")

            # Download for chunks of 30 years
            maca_data = []
            start_year = scenario_params['start_year']
            end_year = scenario_params['end_year']

            # Loop over time periods in 4-year increments
            for year_start in range(start_year, end_year + 1, 5):
                year_end = min(year_start + 5 - 1, end_year)
                print(f"Downloading data for years {year_start}-{year_end}...")
                
                # Download data for each 4-year chunk
                maca_chunk = download_maca_data(
                    climate_model=scenario_params['climate_model'],
                    ensemble=scenario_params['ensemble'],
                    variable_short=variable_short,
                    var_long_name=var_long_name,  # Use the long name here
                    scenario=scenario_params['scenario'],
                    start_year=year_start,
                    end_year=year_end,
                    spatial_domain=spatial_domain
                )
                maca_data.append(maca_chunk)

            # Combine all the downloaded chunks (this may need adjustments based on your data)
            combined_maca_data = xr.concat(maca_data, dim='time')

            # Store the downloaded data in the dictionary with a composite key (description + variable)
            key = f"{scenario_params['description']} - {var_long_name}"
            maca_data_dict[key] = combined_maca_data

    return maca_data_dict


# Define common parameters that apply to both spatial domains for historical, 
# then future (make a list of variables in dictionary format)
common_params_historical = {
    'variable_short': ["pr", "tasmin"],  # Precipitation (pr) and Minimum Temperature (tasmin)
    'var_long_name': ["precipitation", "air_temperature"], # Precipitation and Minimum Temperature
    'scenario': "historical",  # Default scenario, can be overridden
    'start_year': 1970,  # Default start year, can be overridden
    'end_year': 1999,  # Default end year, can be overridden
}
common_params_future = {
    'variable_short': ["pr", "tasmin"],  # Precipitation (pr) and Minimum Temperature (tasmin)
    'var_long_name': ["precipitation", "air_temperature"], # Precipitation and Minimum Temperature
    'scenario': "rcp85",  # Default scenario, can be overridden
    'start_year': 2071,  # Default start year, can be overridden
    'end_year': 2099,  # Default end year, can be overridden
}
# Define Comanche wet/dry, cold/warm scenarios
# for both historical (1971-2000) and the 2070-2099 time frames
comanche_scenarios_historical = [
    # Comanche Historical Warm and Wet
    {'climate_model':"CanESM2",'ensemble':"r1i1p1",'description':"C Historical Warm and Wet"},
    # Comanche Historical Warm and Dry
    {'climate_model':"CCSM4", 'ensemble':"r6i1p1",'description':"C Historical Warm and Dry"},
    # Comanche Historical Cold and Wet
    {'climate_model':"HadGEM2-CC365",'ensemble':"r1i1p1",'description':"C Historical Cold and Wet"},
    # Comanche Historical Cold and Dry
    {'climate_model':"inmcm4",'ensemble':"r1i1p1",'description':"C Historical Cold and Dry"}
]
comanche_scenarios_future = [
    # Comanche Future Warm and Wet
    {'climate_model':"CanESM2",'ensemble':"r1i1p1",'description':"C Future Warm and Wet"},
    # Comanche Future Warm and Dry
    {'climate_model':"IPSL-CM5A-MR",'ensemble':"r1i1p1",'description':"C Future Warm and Dry"},
    # Comanche Future Cold and Wet
    {'climate_model':"MRI-CGCM3",'ensemble':"r1i1p1",'description':"C Future Cold and Wet"},
    # Comanche Future Cold and Dry
    {'climate_model':"bcc-csm1-1-m",'ensemble': "r1i1p1",'description':"C Future Cold and Dry"}
    ]
pawnee_scenarios_historical = [
    # Pawnee Historical Warm and Wet
    {'climate_model':"BNU-ESM",'ensemble':"r1i1p1",'description':"P Historical Warm and Wet"},
    # Pawnee Historical Warm and Dry
    {'climate_model':"bcc-csm1-1-m",'ensemble':"r1i1p1",'description':"P Historical Warm and Dry"},
    # Pawnee Historical Cold and Wet
    {'climate_model':"IPSL-CM5A-MR",'ensemble':"r1i1p1",'description':"P Historical Cold and Wet"},
    # Pawnee Historical Cold and Dry
    {'climate_model':"bcc-csm1-1",'ensemble':"r1i1p1",'description':"P Historical Cold and Dry"}
]
pawnee_scenarios_future = [
    # Pawnee Future Warm and Wet
    {'climate_model':"CanESM2",'ensemble':"r1i1p1",'description':"P Future Warm and Wet"},
    # Pawnee Future Warm and Dry
    {'climate_model':"HadGEM2-CC365",'ensemble':"r1i1p1",'description':"P Future Warm and Dry"},
    # Pawnee Future Cold and Wet
    {'climate_model':"GFDL-ESM2G",'ensemble':"r1i1p1",'description':"P Future Cold and Wet"},
    # Pawnee Future Cold and Dry
    {'climate_model':"CSIRO-Mk3-6-0",'ensemble':"r1i1p1",'description':"P Future Cold and Dry"}
]

# Try function on one of these scenario dictionaries to see if it worked
# Comanche historical - Call the function to download data
comanche_historical_maca_data_dict = download_for_scenarios(
    comanche_scenarios_historical, common_params_historical, comanche_grassland_gdf)

# %%
# See what the key names are so that I can call them correctly
print(comanche_historical_maca_data_dict.keys())

# %%
# Select a scenario from the Comanche historical data dictionary
# (example: "C Historical Warm and Wet" scenario)
variable = 'pr'  # Precipitation variable (short name)

# Extract the data for the selected scenario 
# (e.g., "C Historical Warm and Wet") from the dict
c_ww_scenario_key = "C Historical Warm and Wet - precipitation"
c_ww_maca_da = comanche_historical_maca_data_dict[c_ww_scenario_key]

print(c_ww_maca_da['time'])

"""I was having issues in the next cell and wanted to see what type
of time or what format time was in, in order to aggregate over a 30 
year period. This tells me I need to change the time type or 
adjust so I can use the time slice correctly/ without a type error"""

# %%
# Try plotting "C Historical Warm and Wet" scenario), but 
# first create da to adjust to be able to plot aggregated 30 year

variable = 'pr'  # Precipitation variable (short name)
# Extract the data for the selected scenario 
# "C Historical Warm and Wet - precipitation" from the dict
c_ww_scenario_key = "C Historical Warm and Wet - precipitation"
c_ww_maca_da = comanche_historical_maca_data_dict[c_ww_scenario_key]
# Convert the 'time' dimension to pandas datetime format since this is cftime
c_ww_maca_da['time'] = c_ww_maca_da['time'].values.astype('datetime64[ns]')
# Slice the time range from 1970 to 1999
c_ww_maca_da = c_ww_maca_da.sel(time=slice('1970-01-01', '1999-12-31'))
# lat lon crs
c_ww_maca_da = c_ww_maca_da.rio.write_crs(4326)
# Set the spatial dimensions: lat and lon
c_ww_maca_da = c_ww_maca_da.rio.set_spatial_dims('lat', 'lon')
# Sum up the precipitation over the years, min
c_ww_maca_30_years_da = c_ww_maca_da.groupby('time.year').sum(dim='time').min('year')

# Plot the spatial map of precipitation for aggregated 30-year period
c_ww_maca_30_years_da.plot(
    cbar_kwargs={"label": "Precipitation (mm)"},
    robust=True
)
# Overlay the boundary of the Comanche Grassland study area
comanche_grassland_gdf.boundary.plot(ax=plt.gca(),
    color='black').set(
        title='Comanche Grassland - Wet and Warm - 1970-1999',
        xlabel='Longitude', 
        ylabel='Latitude',
        xticks=[],
        yticks=[] 
    )

plt.show()

# %% [markdown]
# # The download worked and plotted correctly, so I am 
# # moving forward with downloading the rest of the 
# # scenarios

# %%
# Comanche Future - Call the function to download data
comanche_future_maca_data_dict = download_for_scenarios(
    comanche_scenarios_future, common_params_future, comanche_grassland_gdf)

# %%
# Pawnee historical - Call the function to download data
pawnee_historical_maca_data_dict = download_for_scenarios(
    pawnee_scenarios_historical, common_params_historical, pawnee_grassland_gdf)

# %%
# Pawnee Future - Call the function to download data
pawnee_future_maca_data_dict = download_for_scenarios(
    pawnee_scenarios_future, common_params_future, pawnee_grassland_gdf)

# %%
# Compile all da's I put into separate dictionaries into 1 data dictionary?
combined_maca_data_dict = {
    **comanche_historical_maca_data_dict,
    **comanche_future_maca_data_dict,
    **pawnee_historical_maca_data_dict,
    **pawnee_future_maca_data_dict
}

# See what the key names are to make sure all of them are in there now
print(combined_maca_data_dict.keys())

# %%
# Make a function to process the climate model rasters
def process_scenario(scenario_key, start_date, end_date, data_dict):
    """
    Function to process a scenario by slicing the time range and 
    calculating the 30-year sum.

    Parameters:
    -----------
    scenario_key : str
        The key for the specific scenario in the data dictionary.
    start_date : str
        The start date for the time slice (e.g., '1970-01-01').
    end_date : str
        The end date for the time slice (e.g., '1999-12-31').
    data_dict : dict
        The data dictionary containing all the climate data.

    Returns:
    --------
    xarray.DataArray
        The aggregated 30-year data for the specified scenario.
    """
    # Extract the data for the selected scenario
    processed_maca_da = data_dict[scenario_key]
    # Convert the 'time' dimension to pandas datetime format since this is cftime
    processed_maca_da['time'] = processed_maca_da['time'].values.astype('datetime64[ns]')
    # Slice the time range
    processed_maca_da = processed_maca_da.sel(time=slice(start_date, end_date))
    # Set lat and lon CRS
    processed_maca_da = processed_maca_da.rio.write_crs(4326)
    # Set the spatial dimensions: lat and lon
    processed_maca_da = processed_maca_da.rio.set_spatial_dims('lat', 'lon')
    # Sum over the years and take the minimum
    maca_30_years_da = processed_maca_da.groupby('time.year').sum(dim='time').min('year')

    return maca_30_years_da

# Define the scenarios and their corresponding time slices
# I'm only going to process precipitation for sake of finishing 
# this assignment on time, otherwise I would add min temp
# I also think min temp would need slightly differnt processing
# That I don't have time to look into
processed_scenarios = [
    # Comanche Historical
    {"scenario_key": "C Historical Warm and Wet - precipitation", 
     "start_date": '1970-01-01', "end_date": '1999-12-31'},
    {"scenario_key": "C Historical Warm and Dry - precipitation", 
     "start_date": '1970-01-01', "end_date": '1999-12-31'},
    {"scenario_key": "C Historical Cold and Wet - precipitation", 
     "start_date": '1970-01-01', "end_date": '1999-12-31'},
    {"scenario_key": "C Historical Cold and Dry - precipitation", 
     "start_date": '1970-01-01', "end_date": '1999-12-31'},
    #{"scenario_key": "C Historical Warm and Wet - air_temperature", 
    # "start_date": '1970-01-01', "end_date": '1999-12-31'},
    #{"scenario_key": "C Historical Warm and Dry - air_temperature", 
    # "start_date": '1970-01-01', "end_date": '1999-12-31'},
    #{"scenario_key": "C Historical Cold and Wet - air_temperature", 
    # "start_date": '1970-01-01', "end_date": '1999-12-31'},
    #{"scenario_key": "C Historical Cold and Dry - air_temperature", 
    # "start_date": '1970-01-01', "end_date": '1999-12-31'},
    # Comanche Future
    {"scenario_key": "C Future Warm and Wet - precipitation", 
     "start_date": '2071-01-01', "end_date": '2099-12-31'},
    {"scenario_key": "C Future Warm and Dry - precipitation", 
     "start_date": '2071-01-01', "end_date": '2099-12-31'},
    {"scenario_key": "C Future Cold and Wet - precipitation", 
     "start_date": '2071-01-01', "end_date": '2099-12-31'},
    {"scenario_key": "C Future Cold and Dry - precipitation", 
     "start_date": '2071-01-01', "end_date": '2099-12-31'},
    #{"scenario_key": "C Future Warm and Wet - air_temperature", 
    # "start_date": '2071-01-01', "end_date": '2099-12-31'}, 
    #{"scenario_key": "C Future Warm and Dry - air_temperature", 
    # "start_date": '2071-01-01', "end_date": '2099-12-31'},
    #{"scenario_key": "C Future Cold and Wet - air_temperature", 
    # "start_date": '2071-01-01', "end_date": '2099-12-31'},
    #{"scenario_key": "C Future Cold and Dry - air_temperature", 
    # "start_date": '2071-01-01', "end_date": '2099-12-31'},
    # Pawnee Historical
    {"scenario_key": "P Historical Warm and Wet - precipitation", 
     "start_date": '1970-01-01', "end_date": '1999-12-31'},
    {"scenario_key": "P Historical Warm and Dry - precipitation", 
     "start_date": '1970-01-01', "end_date": '1999-12-31'},
    {"scenario_key": "P Historical Cold and Wet - precipitation", 
     "start_date": '1970-01-01', "end_date": '1999-12-31'},
    {"scenario_key": "P Historical Cold and Dry - precipitation", 
     "start_date": '1970-01-01', "end_date": '1999-12-31'},
    #{"scenario_key": "P Historical Warm and Wet - air_temperature", 
    # "start_date": '1970-01-01', "end_date": '1999-12-31'},
    #{"scenario_key": "P Historical Warm and Dry - air_temperature", 
    # "start_date": '1970-01-01', "end_date": '1999-12-31'},
    #{"scenario_key": "P Historical Cold and Wet - air_temperature", 
    # "start_date": '1970-01-01', "end_date": '1999-12-31'},
    #{"scenario_key": "P Historical Cold and Dry - air_temperature", 
    # "start_date": '1970-01-01', "end_date": '1999-12-31'},
    # Pawnee Future
    {"scenario_key": "P Future Warm and Wet - precipitation", 
     "start_date": '2071-01-01', "end_date": '2099-12-31'},
    {"scenario_key": "P Future Warm and Dry - precipitation", 
     "start_date": '2071-01-01', "end_date": '2099-12-31'},
    {"scenario_key": "P Future Cold and Wet - precipitation", 
     "start_date": '2071-01-01', "end_date": '2099-12-31'},
    {"scenario_key": "P Future Cold and Dry - precipitation", 
     "start_date": '2071-01-01', "end_date": '2099-12-31'},
    #{"scenario_key": "P Future Warm and Wet - air_temperature", 
    # "start_date": '2071-01-01', "end_date": '2099-12-31'}, 
    #{"scenario_key": "P Future Warm and Dry - air_temperature", 
    # "start_date": '2071-01-01', "end_date": '2099-12-31'},
    #{"scenario_key": "P Future Cold and Wet - air_temperature", 
    # "start_date": '2071-01-01', "end_date": '2099-12-31'},
    #{"scenario_key": "P Future Cold and Dry - air_temperature", 
    # "start_date": '2071-01-01', "end_date": '2099-12-31'},
]

# Create an empty dictionary to store the processed data
processed_climate_data = {}

# Loop through all the scenarios and process them
for processed_scenario in processed_scenarios:
    scenario_key = processed_scenario["scenario_key"]
    start_date = processed_scenario["start_date"]
    end_date = processed_scenario["end_date"]

    # Process each scenario and store the result in the dictionary
    processed_climate_data[scenario_key] = process_scenario(scenario_key, start_date, end_date, combined_maca_data_dict)

# %%
# See what the key names are to make sure all of them are in there now
print(processed_climate_data.keys())

# %%
# Example for plotting one of the processed scenarios:
scenario_key_to_plot = "P Historical Warm and Wet - precipitation" 

# Retrieve the processed data for the selected scenario
p_ww_h_maca_30_years_da = processed_climate_data[scenario_key_to_plot]

p_ww_h_maca_30_years_da.plot(
    cbar_kwargs={"label": "Precipitation (mm)"},
    robust=True
)
# Overlay the boundary of the Comanche Grassland study area
pawnee_grassland_gdf.boundary.plot(ax=plt.gca(),
    color='black').set(
        title='Pawnee Grassland - Wet and Warm - 1970-1999',
        xlabel='Longitude', 
        ylabel='Latitude',
        xticks=[],
        yticks=[] 
    )
plt.show()

# %% [markdown]
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>Make sure to include a description of the climate data and how you
# selected your models. Include a citation of the MACAv2 data</p></div></div>
# 
# 

# %% [markdown]
# * ### Climate Data: MACAv2 via THREDDS data server (climate scenarios)
# 
# I included how I selected the models in the section at the end of the 
# Study Overview section at the very beginning of this/before code starts.
# 
# Climate Projection Models are one way to guide restoration of many habitats 
# in face of climate change (Global Percipitation Measuremet Mission, NASA 
# n.d.). These projection models use outputs of Global Climate Models (GCM), 
# which simulate the global and regional scale climate processes that have data 
# collected from satellites, weather stations, oceanic buoys, and other methods 
# (NOAA 2019). As part of the Climate Model Intercomparison Project 
# phase 5 (CMIP5), data was drawn from over 40 GCM's from coutnries across the globe 
# to analyze and compare these many GCM's (Eyring et al. 2016). This comparison 
# allows for better understanding of climate change now/historically as well as 
# in the future (Global Percipitation Measuremet Mission, NASA n.d.). However, one 
# of the drawbacks of this being global data is that it has coarse resolution because 
# of the scale, so an image or map using the data would appear like a pixelated 
# photo of low resolution (University of California MERCED, "MACA Overview").
# 
# One way to try to combat this coarse resolution is to downscale this spatial data. 
# MACA V2 (Multivariate Adaptive Constructed Analogs, version 2) does just that 
# (Abatzoglou and Brown 2012). Using statistical operations 
# ([find more information here](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/joc.2312))
# , that result in a 4km resolution which produces a higher resolution image 
# (Abatzoglou and Brown 2012).It should be noted that MACA V2 is only for CONUS, 
# a similar type of dataset may heve been done in other regions globally but that 
# would need to be researched further if interested in a project location outside 
# CONUS. Each of the climate models avaialble here have slightly different 
# resolution, for the purposes of doing this project that shouldn't largely 
# affect the outcome (University of California MERCED, “MACA Statistical Downscaling 
# Method.”). While 4km and 30m are upon face value are vastly different numbers, 
# climate data represents broaderatmospheric conditions across a larger area, while 
# the soil data is at a 30m resolution due to the much higher variability in soil 
# properties and elevation across smaller distances, making finer detail necessary 
# to accurately capture local soil characteristics and elevation (Chaney and Duke 
# University n.d.). Essentially, climate is more uniform over larger areas compared 
# to soil and elevation which can change significantly within a short distance (Global 
# Percipitation Measuremet Mission, NASA n.d.). 
# 
# While there are benefits to downscaling, a disadvantage to it is that 
# due to the statistical methods used, introduces uncertainty (University of 
# California MERCED n.d., "GCM Selection"). This needs to be kept in mind, but 
# the MACA V2 dataset is widely accepted by many institutions and organizations 
# such as the U.S. Forest Service (Coulson and Joyce n.d.).  
# 
# The THREDDS (Thematic Real-time Environmental Distributed Data Services) data 
# server is a web server that can be used to access the MACA v2 dataset as well 
# as other scientific datasets (Unidata 2018). This catalog of datasets, accepts 
# a few different data formats including NetCDF, GRIB, and HDF (Unidata 2018). 
# Because the data in the MACA V2 dataset is available in NetCDF as its original format, 
# that will work for this project (Unidata 2018).
# 
# MACA V2 has 20 different climate models, each of these models can be found 
# [here](https://climate.northwestknowledge.net/MACA/GCMs.php)(University of 
# California MERCED, “MACA Statistical Downscaling Method.”). Each of these has 9 
# climate variables to choose from, the climate variable options can be found 
# [here](https://climate.northwestknowledge.net/MACA/MACAproducts.php)
# (University of California MERCED, “MACA Statistical Downscaling Method.”). 
# There are 3 climate scenarios available: actual/historic, intermediate, and worst 
# case climate scenarios (University of California MERCED, “MACA Statistical 
# Downscaling Method.”). Climate scenarios avilable here are either different time 
# periods or different emissions scenarios (University of California MERCED, “MACA 
# Statistical Downscaling Method.”). For this project, multiple climate models are 
# chosen as an ensemble, the climate variable chosen is percipitation and the two 
# temporal climate scenarios will be used. Two different time periods are chosen 
# (historic - 'historical , and worst case scenario - rcm85), and this route was 
# chosen because it will aid in the analysis of suitable habitats for the Rocky 
# Mountatin Juniper in both of the study areas chosen (Comanche and Pawnee National 
# Grasslands) (University of California MERCED n.d., "GCM Selection"). Different 
# time periods provide insight into the past as well as possible future scenarios 
# and the two can be compared to look at the validity of the habitat suitability 
# model to be created for this project (University of California MERCED n.d., "GCM 
# Selection"). 
# 
# 
# The MACA V2 data is available under a 
# [Creative Commons CC0 1.0 Universal dedication license](https://creativecommons.org/publicdomain/zero/1.0/) - 
# the dataset was "created with funding from the US government 
# and are in the public domain in the United States" (Creative Commons 2019).
# This data server can be accessed 
# [here](https://climate.northwestknowledge.net/NWTOOLBOX/mapping.php),
# but there are many ways the data can be downloaded, please see 
# [MACA website](https://climate.northwestknowledge.net/MACA/gallery_data.php) .
# 
# ## Citations:
# 
# * Abatzoglou, John T., and Timothy J. Brown. 2011. “A Comparison of Statistical 
# Downscaling Methods Suited for Wildfire Applications.” International Journal of 
# Climatology 32 (5). https://doi.org/10.1002/joc.2312.
# 
# * Chaney, Nathaniel, and Duke University. n.d. “POLARIS – a Probabilistic Soil 
# Classification and Property Database over the Contiguous United States at a 
# 30-Meter Spatial Resolution — Duke OTC.” Duke OTC. Accessed March 13, 2025. 
# https://otc.duke.edu/technologies/polaris-a-probabilistic-soil-classification-and-property-database-over-the-contiguous-united-states-at-a-30-meter-spatial-resolution/.
# 
# * Coulson, David P., and Linda A. Joyce. n.d. “RPA Historical Observational 
# Data (1979-2015) for the Conterminous United States at the 1/24 Degree Grid 
# Scale Based on MACA Training Data (METDATA).” Forest Service Research Data 
# Archive. Accessed March 13, 2025. https://doi.org/10.2737/rds-2017-0070-2.
# 
# * Creative Commons. 2019. “Creative Commons — CC0 1.0 Universal.” 
# Creativecommons.org. 2019. https://creativecommons.org/publicdomain/zero/1.0/.
# 
# * Eyring, Veronika, Sandrine Bony, Gerald A. Meehl, Catherine A. Senior, 
# Bjorn Stevens, Ronald J. Stouffer, and Karl E. Taylor. 2016. “Overview of 
# the Coupled Model Intercomparison Project Phase 6 (CMIP6)  Experimental 
# Design and Organization.” Geoscientific Model Development 9 (5): 1937–58. 
# https://doi.org/10.5194/gmd-9-1937-2016.
# 
# * Global Percipitation Measuremet Mission, NASA. n.d. “How Do We Predict Future 
# Climate?” Gpm.nasa.gov. Accessed March 12, 2025. 
# https://gpm.nasa.gov/education/sites/default/files/lesson_plan_files/climate-package/Climate%20Models.pdf.
# 
# * NOAA. 2019. “Climate Data Monitoring | National Oceanic and Atmospheric 
# Administration.” Www.noaa.gov. February 1, 2019. 
# https://www.noaa.gov/education/resource-collections/climate/climate-data-monitoring.
# 
# * unidata. 2018. “THREDDS Data Server Version 4.6 Documentation.” Ucar.edu. 2018. 
# https://docs.unidata.ucar.edu/tds/4.6/adminguide/.
# 
# * University of California MERCED. n.d. “GCM Selection.” Northwestknowledge.net. 
# Accessed March 13, 2025. https://climate.northwestknowledge.net/MACA/GCMselection.php.
# 
# * ———. n.d. “MACA Overview.” Northwestknowledge.net. University of California 
# MERCED. Accessed March 13, 2025. 
# https://climate.northwestknowledge.net/MACA/MACAmethod.php.
# 
# * ———. n.d. “MACA Statistical Downscaling Method.” Northwestknowledge.net. 
# Accessed March 12, 2025. https://climate.northwestknowledge.net/MACA/GCMs.php.
# 

# %% [markdown]
# 
# 
# ## STEP 3: HARMONIZE DATA
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Make sure that the grids for all your data match each other. Check
# out the <a
# href="https://corteva.github.io/rioxarray/stable/examples/reproject_match.html#Reproject-Match"><code>ds.rio.reproject_match()</code>
# method</a> from <code>rioxarray</code>. Make sure to use the data source
# that has the highest resolution as a template!</p></div></div>
# 
# > **Warning**
# >
# > If you are reprojecting data as you need to here, the order of
# > operations is important! Recall that reprojecting will typically tilt
# > your data, leaving narrow sections of the data at the edge blank.
# > However, to reproject efficiently it is best for the raster to be as
# > small as possible before performing the operation. We recommend the
# > following process:
# >
# >     1. Crop the data, leaving a buffer around the final boundary
# >     2. Reproject to match the template grid (this will also crop any leftovers off the image)

# %% [markdown]
# I've already clipped everything along the way, so I'm skipping #1

# %%
# Prep for harmonizing Part 1 of 1
# Create output directory for this harmonized raster
harmonized_raster_dir = os.path.join(part_2_hab_suit_data_dir, 'harmonized_rasters')
os.makedirs(harmonized_raster_dir, exist_ok=True) 

harmonized_raster_dir

# %%
# Harmonize raster layers by creating function
def harmonize_raster_layers(reference_raster, input_rasters, output_dir, scenario_keys):
    """
    Harmonize raster layers to ensure consistent spatial resolution and projection.

    Parameters:
    -----------
        reference_raster (str or DataArray): 
            Reference raster or a path to the reference raster.
        input_rasters (list): 
            List of input raster DataArrays or file paths to harmonize.
        output_dir (str): 
            Directory to save the harmonized raster files.
        scenario_keys (list): 
            List of scenario keys corresponding to each raster in input_rasters.

    Returns:
    --------
        list: List of file paths to the harmonized raster files.
    """
    
    # Load the reference raster (this can be a DataArray or a path)
    if isinstance(reference_raster, str):  # If it's a path
        ref_raster = rxr.open_rasterio(reference_raster, masked=True)
    else:  # If it's a DataArray
        ref_raster = reference_raster

    # Create list to save the harmonized data arrays back to
    harmonized_rasters = []

    # Create for loop to iterate through multiple rasters and corresponding scenario keys
    for i, raster in enumerate(input_rasters):
        scenario_key = scenario_keys[i]  # Get scenario key corresponding to this raster

        # If raster is a file path (string)
        if isinstance(raster, str):
            input_raster = rxr.open_rasterio(raster, masked=True)
        else:
            input_raster = raster  # It is already a DataArray

        # Reproject and align the input raster to match the reference raster
        harmonized_raster = input_raster.rio.reproject_match(ref_raster)

        # Create a file name based on the scenario key
        harmonized_output_file = os.path.join(output_dir, 
            f"{scenario_key}_harmonized.tif")

        # Save the harmonized raster to the output directory
        harmonized_raster.rio.to_raster(harmonized_output_file)

        # Print and append the harmonized raster path
        print(f"Harmonized raster saved to: {harmonized_output_file}")
        harmonized_rasters.append(harmonized_output_file)

    # Return the list of harmonized raster paths
    return harmonized_rasters

# %%
# Comanche Harmonization of Rasters

# Comanche raster data arrays
comanche_reference_raster = slope_results[0] 
# Create a list of scenario keys for Comanche
comanche_scenario_keys = [
    "C Historical Warm and Wet - precipitation",
    "C Historical Warm and Dry - precipitation",
    "C Historical Cold and Wet - precipitation",
    "C Historical Cold and Dry - precipitation",
    "C Future Warm and Wet - precipitation",
    "C Future Warm and Dry - precipitation",
    "C Future Cold and Wet - precipitation",
    "C Future Cold and Dry - precipitation"
]
# Add a specific scenario key for the Polaris raster
comanche_scenario_keys.append("C Polaris Processed")
# Initialize the list of input rasters
comanche_input_rasters = []
# Iterate through the processed climate data and add the corresponding rasters
for scenario_key in comanche_scenario_keys[:-1]:  # Exclude "C Polaris Processed" key
    if scenario_key in processed_climate_data:
        comanche_input_rasters.append(processed_climate_data[scenario_key])
# Add the Polaris raster to the list (with the assigned scenario key)
comanche_input_rasters.append(polaris_processed_da_list[0])
# Ensure both lists are of the same length
assert len(comanche_input_rasters) == len(comanche_scenario_keys
    ),"Number of input rasters and scenario keys do not match!"
# Harmonize Comanche rasters
comanche_harmonized_rasters = harmonize_raster_layers(
    comanche_reference_raster,
    comanche_input_rasters,
    harmonized_raster_dir,
    comanche_scenario_keys
)

# Pawnee Harmonization of Rasters

# Pawnee raster data arrays
pawnee_reference_raster = slope_results[1]
# Create a list of scenario keys for Pawnee
pawnee_scenario_keys = [
    "P Historical Warm and Wet - precipitation",
    "P Historical Warm and Dry - precipitation",
    "P Historical Cold and Wet - precipitation",
    "P Historical Cold and Dry - precipitation",
    "P Future Warm and Wet - precipitation",
    "P Future Warm and Dry - precipitation",
    "P Future Cold and Wet - precipitation",
    "P Future Cold and Dry - precipitation"
]
# Add a specific scenario key for the Polaris raster
pawnee_scenario_keys.append("P Polaris Processed")
# Initialize the list of input rasters
pawnee_input_rasters = []
# Iterate through the processed climate data and add the corresponding rasters
for scenario_key in pawnee_scenario_keys[:-1]:  # Exclude "C Polaris Processed" key
    if scenario_key in processed_climate_data:
        pawnee_input_rasters.append(processed_climate_data[scenario_key])
# Add the Polaris raster to the list (with the assigned scenario key)
pawnee_input_rasters.append(polaris_processed_da_list[1])
# Ensure both lists are of the same length
assert len(pawnee_input_rasters) == len(pawnee_scenario_keys
    ),"Number of input rasters and scenario keys do not match!"

# Harmonize Pawnee rasters
pawnee_harmonized_rasters = harmonize_raster_layers(
    pawnee_reference_raster,
    pawnee_input_rasters,
    harmonized_raster_dir,
    pawnee_scenario_keys
)

# %%
# Choose one harmonized raster to plot as an example to 
# make sure the harmonization worked 
example_harmonized_raster = rxr.open_rasterio(comanche_harmonized_rasters[0], masked=True)

example_harmonized_raster.plot(
    cbar_kwargs={"label": "Precipitation (mm)"},
    robust=True
)
# Overlay the boundary of the Comanche Grassland study area
comanche_grassland_gdf.boundary.plot(ax=plt.gca(),
    color='black').set(
        title='Comanche - Wet + Warm -1970-1999 - harmonized',
        xlabel='Longitude', 
        ylabel='Latitude',
        xticks=[],
        yticks=[]
    )
plt.show()

# %% [markdown]
# I don't love how cropped it is, I would like to 
# investigate further, but for the sake of time I 
# need to keep moving. I think there is a way to have 
# rasters go slightly beyond the bounrds and I must not 
# have done that along the way, probably to the calculated 
# slope. However, the harmonization worked because the 
# climate model raster which is plotted above is at a 
# higher resolution that it was prior to the harmonization. 
# So, I think I am good to move on.

# %% [markdown]
# ## STEP 4: DEVELOP A FUZZY LOGIC MODEL
# 
# A fuzzy logic model is one that is built on expert knowledge rather than
# training data. You may wish to use the
# [`scikit-fuzzy`](https://pythonhosted.org/scikit-fuzzy/) library, which
# includes many utilities for building this sort of model. In particular,
# it contains a number of **membership functions** which can convert your
# data into values from 0 to 1 using information such as, for example, the
# maximum, minimum, and optimal values for soil pH.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>To train a fuzzy logic habitat suitability model:</p>
# <pre><code>1. Research S. nutans, and find out what optimal values are for each variable you are using (e.g. soil pH, slope, and current climatological annual precipitation). 
# 2. For each **digital number** in each raster, assign a **continuous** value from 0 to 1 for how close that grid square is to the optimum range (1=optimal, 0=incompatible). 
# 3. Combine your layers by multiplying them together. This will give you a single suitability number for each square.
# 4. Optionally, you may apply a suitability threshold to make the most suitable areas pop on your map.</code></pre></div></div>
# 
# > **Tip**
# >
# > If you use mathematical operators on a raster in Python, it will
# > automatically perform the operation for every number in the raster.
# > This type of operation is known as a **vectorized** function. **DO NOT
# > DO THIS WITH A LOOP!**. A vectorized function that operates on the
# > whole array at once will be much easier and faster.

# %%
# Fuzzy logic model
# Part 1 - create function for calculating suitability score 
# to use in the habitat suitability model function
"""    
   - The calculate_suitability_score function uses 
   a **fuzzy Gaussian function** to assign suitability 
   scores between 0 and 1 for each raster cell.
   - Cells with values closer to the `optimal_value` 
   receive scores closer to 1, while cells farther away receive 
   lower scores.
   - The tolerance_range determines how quickly the score drops 
   off as values deviate from the optimal value.
"""

def calculate_suitability_score(raster, optimal_value, tolerance_range):
    """
    Calculate a fuzzy suitability score (0–1) for each raster 
    cell based on proximity to the optimal value.

    Parameters:
    -----------
    raster: xarray.DataArray
        Input raster layer.
    optimal_value: float 
        The optimal value for the variable.
    tolerance_range: float 
        The range within which values are considered suitable.

    Returns:
    --------
        xarray.DataArray: A raster of suitability scores (0–1).
    """
    # Calculate suitability scores using a fuzzy Gaussian function
    suitability = np.exp(-((raster - optimal_value) ** 2) / (2 * tolerance_range ** 2))
    return suitability 


# %%
# Fuzzy logic model
# Part 2 - create function for habitat suitability model
def build_habitat_suitability_model_from_da(
        input_data_arrays, optimal_values, tolerance_ranges, output_dir, threshold=None):
    """
    Build a habitat suitability model by combining fuzzy 
    suitability scores for each variable using xarray DataArrays.

    Args:
        input_data_arrays: list of Data Arrays
            List of xarray DataArrays representing environmental variables.
        optimal_values: list 
            List of optimal values for each variable.
        tolerance_ranges: list 
            List of tolerance ranges for each variable.
        output_dir: str 
            Directory to save the combined suitability raster.
        threshold: float, optional 
            Threshold for highlighting highly suitable areas (default is None).

    Returns:
       combined_suitability: da
            The combined suitability DataArray.
    """
    os.makedirs(output_dir, exist_ok=True)

    # List to store suitability score layers
    suitability_layers = []

    # Calculate suitability for each input data array
    for data_array, optimal_value, tolerance_range in zip(
        input_data_arrays, optimal_values, tolerance_ranges):

        # Type check for each input data array
        if not isinstance(data_array, xr.DataArray):
            raise TypeError(f"Expected xarray.DataArray, but got {type(data_array)}")

        suitability_layer = calculate_suitability_score(
            data_array, optimal_value, tolerance_range)
        suitability_layers.append(suitability_layer)

    # Combine suitability scores by multiplying across all layers
    combined_suitability = suitability_layers[0]
    for layer in suitability_layers[1:]:
        combined_suitability * layer

    # Apply a threshold if provided
    if threshold is not None:
        combined_suitability = xr.where(combined_suitability >= threshold, 
        combined_suitability, 0)

    # Save the combined suitability raster (optional)
    output_file = os.path.join(output_dir, "combined_suitability.tif")
    combined_suitability.rio.to_raster(output_file)
    print(f"Combined suitability raster saved to: {output_file}")

    return combined_suitability

# %%
# Prep for habitat suitability model Part 1 of 3
# Create output directory for these harmonized rasters
hab_suit_model_dir = os.path.join(
    part_2_hab_suit_data_dir, 'habitat_suitability_model_rasters')
os.makedirs(hab_suit_model_dir, exist_ok=True) 

hab_suit_model_dir

# %%
# Prep for habitat suitability model Part 2 of 3 - Comanche Dict
# Create dictionary of harmonized data in order to call in the function
# hold the file name as key and da as value
comanche_harmonized_rasters_dict = {}
# Populate the dictionary
for file_path in comanche_harmonized_rasters:
    # Extract the file name without extension - .tif
    file_name = os.path.basename(file_path).replace('.tif', '')  # Removing .tif from the file name
    
    # Open the raster file as da
    data_array = rxr.open_rasterio(file_path, masked=True)
    
    # Add the da to the dict with the file name as the key
    comanche_harmonized_rasters_dict[file_name] = data_array

# Access each da by its file name
print(comanche_harmonized_rasters_dict)

# %%
# Prep for habitat suitability model Part 3 of 3 -  Pawnee Dict
# Create dictionary of harmonized data in order to call in the function
# hold the file name as key and da as value
pawnee_harmonized_rasters_dict = {}
# Populate the dictionary
for file_path in pawnee_harmonized_rasters:
    # Extract the file name without the .tif
    file_name = os.path.basename(file_path).replace('.tif', '')  # Removing .tif from the file name
    
    # Open the raster file as da
    data_array = rxr.open_rasterio(file_path, masked=True)
    
    # Add the da to the dict with the file name as the key
    pawnee_harmonized_rasters_dict[file_name] = data_array

# Acess each da by its file name
print(pawnee_harmonized_rasters_dict)

# %%
# Test out the function on one set of input da's
# Comanche Historical Wet - Warm Set Up 
c_h_ww_input_data_arrays = [
    # harmonized climate model warm and wet historical for comanche
    comanche_harmonized_rasters_dict["C Historical Warm and Wet - precipitation_harmonized"], 
    # harmonized polaris da for comanche
    comanche_harmonized_rasters_dict["C Polaris Processed_harmonized"], 
    # calaculated slope for comanche
    aligned_slope_results[0]
    ] 
# Optimal and tolerance in order - precipitation, soil pH, slope
 # precipitation in mm (228.6 - 660.4 mm), soil pH (5-8.5), slope in degrees (10-30)
c_h_ww_optimal_values = [440, 6.75, 20 ]  
c_h_ww_tolerance_ranges = [220, 1.75, 10] 
c_h_ww_threshold = 0.7  

# Call the function
c_h_ww_combined_suitability = build_habitat_suitability_model_from_da(
    c_h_ww_input_data_arrays, c_h_ww_optimal_values, 
    c_h_ww_tolerance_ranges, hab_suit_model_dir, c_h_ww_threshold
)

# If you want to inspect the resulting suitability DataArray
print(c_h_ww_combined_suitability)

# %%
# Try plotting the above example to make sure it worked
c_h_ww_combined_suitability.plot(
    cbar_kwargs={"label": "Suitability Score"},
    robust=True
)
# Overlay the boundary of the Comanche Grassland study area
comanche_grassland_gdf.boundary.plot(ax=plt.gca(),
    color='black').set(
        title='Comanche Grassland - Wet and Warm - 1970-1999',
        xlabel='Longitude', 
        ylabel='Latitude',
        xticks=[],
        yticks=[] 
    )
plt.show()

# %%
# Comanche combined suitability - historical
# List of keys related to precipitation in the dictionary
comanche_h_precipitation_keys = [
    "C Historical Warm and Wet - precipitation_harmonized",
    "C Historical Warm and Dry - precipitation_harmonized", 
    "C Historical Cold and Wet - precipitation_harmonized",
    "C Historical Cold and Dry - precipitation_harmonized", 
]

# Optimal and tolerance in order - precipitation, soil pH, slope
# Precipitation in mm (228.6 - 660.4 mm), soil pH (5-8.5), slope in degrees
optimal_values = [440, 6.75, 20]
tolerance_ranges = [220, 1.75, 10]
threshold = 0.7

# Create a list to store the combined suitability results
comanche_h_combined_suitability_results = []

# Loop through the precipitation-related rasters
for precipitation_key in comanche_h_precipitation_keys:
    # Prepare the input data arrays for the current precipitation raster
    comanche_h_input_data_arrays = [
        comanche_harmonized_rasters_dict[precipitation_key],  # Looping through each precipitation raster
        comanche_harmonized_rasters_dict["C Polaris Processed_harmonized"],  # Fixed polaris data
        aligned_slope_results[0]  # Fixed slope data (assuming there's only one slope result)
    ]
    
    # Call the habitat suitability model function for each iteration
    comanche_h_combined_suitability = build_habitat_suitability_model_from_da(
        comanche_h_input_data_arrays, optimal_values, 
        tolerance_ranges, hab_suit_model_dir, threshold
    )
    
    # Append the resulting suitability to the list
    comanche_h_combined_suitability_results.append(comanche_h_combined_suitability)
    
    # Print the resulting suitability DataArray for inspection
    print(f"Combined suitability for {precipitation_key}:")
    print(comanche_h_combined_suitability)

# %%
# Comanche combined suitability - future
# List of keys related to precipitation in the dictionary
comanche_f_precipitation_keys = [
    "C Future Warm and Wet - precipitation_harmonized",
    "C Future Warm and Dry - precipitation_harmonized", 
    "C Future Cold and Wet - precipitation_harmonized",
    "C Future Cold and Dry - precipitation_harmonized"   
]

# Optimal and tolerance in order - precipitation, soil pH, slope
# Precipitation in mm (228.6 - 660.4 mm), soil pH (5-8.5), slope in degrees
optimal_values = [440, 6.75, 20]
tolerance_ranges = [220, 1.75, 10]
threshold = 0.7

# Create a list to store the combined suitability results
comanche_f_combined_suitability_results = []

# Loop through the precipitation-related rasters
for precipitation_key in comanche_f_precipitation_keys:
    # Prepare the input data arrays for the current precipitation raster
    comanche_f_input_data_arrays = [
        comanche_harmonized_rasters_dict[precipitation_key],  # Looping through each precipitation raster
        comanche_harmonized_rasters_dict["C Polaris Processed_harmonized"],  # Fixed polaris data
        aligned_slope_results[0]  # Fixed slope data (assuming there's only one slope result)
    ]
    
    # Call the habitat suitability model function for each iteration
    comanche_f_combined_suitability = build_habitat_suitability_model_from_da(
        comanche_f_input_data_arrays, optimal_values, 
        tolerance_ranges, hab_suit_model_dir, threshold
    )
    
    # Append the resulting suitability to the list
    comanche_f_combined_suitability_results.append(comanche_f_combined_suitability)
    
    # Print the resulting suitability DataArray for inspection
    print(f"Combined suitability for {precipitation_key}:")
    print(comanche_f_combined_suitability)

# %%
# Pawnee combined suitability
# List of keys related to precipitation in the dictionary
pawnee_h_precipitation_keys = [
    "P Historical Warm and Wet - precipitation_harmonized",
    "P Historical Warm and Dry - precipitation_harmonized", 
    "P Historical Cold and Wet - precipitation_harmonized",
    "P Historical Cold and Dry - precipitation_harmonized", 
]

# Optimal and tolerance in order - precipitation, soil pH, slope
# Precipitation in mm (228.6 - 660.4 mm), soil pH (5-8.5), slope in degrees
optimal_values = [440, 6.75, 20]
tolerance_ranges = [220, 1.75, 10]
threshold = 0.7

# Create a list to store the combined suitability results
pawnee_h_combined_suitability_results = []

# Loop through the precipitation-related rasters
for precipitation_key in pawnee_h_precipitation_keys:
    # Prepare the input data arrays for the current precipitation raster
    pawnee_h_input_data_arrays = [
        pawnee_harmonized_rasters_dict[precipitation_key],  # Looping through each precipitation raster
        pawnee_harmonized_rasters_dict["P Polaris Processed_harmonized"],  # Fixed polaris data
        aligned_slope_results[1]  # Fixed slope data 
    ]
    
    # Call the habitat suitability model function for each iteration
    pawnee_h_combined_suitability = build_habitat_suitability_model_from_da(
        pawnee_h_input_data_arrays, optimal_values, 
        tolerance_ranges, hab_suit_model_dir, threshold
    )
    
    # Append the resulting suitability to the list
    pawnee_h_combined_suitability_results.append(pawnee_h_combined_suitability)
    
    # Print the resulting suitability DataArray for inspection
    print(f"Combined suitability for {precipitation_key}:")
    print(pawnee_h_combined_suitability)

# %%
# Pawnee combined suitability - Future
# List of keys related to precipitation in the dictionary
pawnee_f_precipitation_keys = [
    "P Future Warm and Wet - precipitation_harmonized",
    "P Future Warm and Dry - precipitation_harmonized", 
    "P Future Cold and Wet - precipitation_harmonized",
    "P Future Cold and Dry - precipitation_harmonized"   
]

# Optimal and tolerance in order - precipitation, soil pH, slope
# Precipitation in mm (228.6 - 660.4 mm), soil pH (5-8.5), slope in degrees
optimal_values = [440, 6.75, 20]
tolerance_ranges = [220, 1.75, 10]
threshold = 0.7

# Create a list to store the combined suitability results
pawnee_f_combined_suitability_results = []

# Loop through the precipitation-related rasters
for precipitation_key in pawnee_f_precipitation_keys:
    # Prepare the input data arrays for the current precipitation raster
    pawnee_f_input_data_arrays = [
        pawnee_harmonized_rasters_dict[precipitation_key],  # Looping through each precipitation raster
        pawnee_harmonized_rasters_dict["P Polaris Processed_harmonized"],  # Fixed polaris data
        aligned_slope_results[1]  # Fixed slope data (assuming there's only one slope result)
    ]
    
    # Call the habitat suitability model function for each iteration
    pawnee_f_combined_suitability = build_habitat_suitability_model_from_da(
        pawnee_f_input_data_arrays, optimal_values, 
        tolerance_ranges, hab_suit_model_dir, threshold
    )
    
    # Append the resulting suitability to the list
    pawnee_f_combined_suitability_results.append(pawnee_f_combined_suitability)
    
    # Print the resulting suitability DataArray for inspection
    print(f"Combined suitability for {precipitation_key}:")
    print(pawnee_f_combined_suitability)

# %% [markdown]
# ## STEP 5: PRESENT YOUR RESULTS
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Generate some plots that show your key findings. Don’t forget to
# interpret your plots!</p></div></div>

# %% [markdown]
# # As a warning most of these plots take a minute or so to plot

# %%
# Comanche Historical Plots

# Create the plot grid (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Titles corresponding to each subplot
titles = [
    "Comanche Historical Warm and Wet - Suitability", 
    "Comanche Historical Warm and Dry - Suitability", 
    "Comanche Historical Cold and Wet - Suitability", 
    "Comanche Historical Cold and Dry - Suitability"
]

# Plot each da in the respective subplot
for i, ax in enumerate(axes.flat):
    # Extract the da for the current scenario
    c_h_suitability = comanche_h_combined_suitability_results[i]
    # Plot the suitability da
    c_h_suitability_plot = c_h_suitability.plot(ax=ax, cmap='PRGn', add_colorbar=False) 
    # Set the title for each subplot
    ax.set_title(titles[i])
    # Add colorbar to each subplot
    fig.colorbar(c_h_suitability_plot, ax=ax, orientation='vertical')
    # Add x and y axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # Overlay the boundary of the Comanche Grassland study area
    comanche_grassland_gdf.boundary.plot(ax=ax, color='orange', linewidth=1.5)

# Adjust layout to prevent overlap
plt.tight_layout()
# Save the figure
plt.savefig("comanche_historical_suitability.png", dpi=300)
# Show the plot
plt.show()

# %% [markdown]
# # Comanche Historical Plots: The Habitat Suitability Model results show 
# # that the most suitable scenario is the warm and dry scenario. Overall 
# # there is more suitability in dry scenarios than there are in wet scenarios
# 
# Another intersting thing about the warm-dry scenario is that almost all of the 
# suitability is in the south unit, the Carrizo Unit. Something to keep in mind 
# thought since this is the historical scenario 1970-1999 is that 'warm' would 
# be warm for this timeframe, and this warm is not exactly comparable to the future 
# scenarios. 
# 
# The lower right plot for cold-dry scenario shows most of the suitability 
# surrounding the Carrizo Unit and only some parts of that unit being suitable 
# for the Rocky Mountain Juniper.
# 
# Both of the wet scenarios on the left show very little suitability, and what 
# little there does show within the bounds, is not in either part of this grassland.

# %%
# Comanche Future plots

# Create the plot grid (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Titles corresponding to each subplot
titles = [
    "Comanche Future Warm and Wet - Suitability", 
    "Comanche Future Warm and Dry - Suitability", 
    "Comanche Future Cold and Wet - Suitability", 
    "Comanche Future Cold and Dry - Suitability"
]

# Plot each da in the respective subplot
for i, ax in enumerate(axes.flat):
    # Extract the da for the current scenario
    c_f_suitability = comanche_f_combined_suitability_results[i]
    # Plot the suitability da
    c_f_suitability_plot = c_f_suitability.plot(ax=ax, cmap='PRGn', add_colorbar=False) 
    # Set the title for each subplot
    ax.set_title(titles[i])
    # Add colorbar to each subplot
    fig.colorbar(c_f_suitability_plot, ax=ax, orientation='vertical')
    # Add x and y axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # Overlay the boundary of the Comanche Grassland study area
    comanche_grassland_gdf.boundary.plot(ax=ax, color='orange', linewidth=1.5)

# Adjust layout to prevent overlap
plt.tight_layout()
# Save the figure
plt.savefig("comanche_future_suitability.png", dpi=300)
# Show the plot
plt.show()

# %% [markdown]
# # Comanche Future Plots: The Habitat Suitability Model results show 
# # that the most suitable scenario is the cold and wet scenario. However,
# # that is the only plot that worked.
# 
# This is disappointing, but only 1 of these plots worked. I am guessing 
# either something went wrong in the climate model download for the 
# particular models chosen, or something went wrong with the merge or harmonizing. 
# If I had to guess it would be the former. The reason why I think it's the 
# climate models as the issue is because I plotted both versions of the 
# soil data, both versions of elevation/slope without issue. I only tested 
# maybe two different scenarios to make sure the climate model downlaod worked.
# So, I likely need to try plotting every single climate model scenario that 
# didn't plot.
# 
# Back to looking at the only plot that worked, the cold and wet scenario, the 
# entire sout unit, the Carrizo unit, is in the suitable range. Again it's 
# important to keep in mind that this cold and wet range is not the exact same 
# as the historical plots. In comparing to the historical though, it is interesting 
# that in both, the Carrizo Unit is the only unit with any suitability.

# %%
# Pawnee Historical Plots

# Create the plot grid (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 7))

# Titles corresponding to each subplot
titles = [
    "Pawnee Historical Warm and Wet - Suitability", 
    "Pawnee Historical Warm and Dry - Suitability", 
    "Pawnee Historical Cold and Wet - Suitability", 
    "Pawnee Historical Cold and Dry - Suitability"
]

# Plot each da in the respective subplot
for i, ax in enumerate(axes.flat):
    # Extract the da for the current scenario
    p_h_suitability = pawnee_h_combined_suitability_results[i]
    # Plot the suitability da
    p_h_suitability_plot = p_h_suitability.plot(ax=ax, cmap='PRGn', add_colorbar=False) 
    # Set the title for each subplot
    ax.set_title(titles[i])
    # Add colorbar to each subplot
    fig.colorbar(p_h_suitability_plot, ax=ax, orientation='vertical')
    # Add x and y axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # Overlay the boundary of the Pawnee Grassland study area
    pawnee_grassland_gdf.boundary.plot(ax=ax, color='orange', linewidth=1.5)

# Adjust layout to prevent overlap
plt.tight_layout()
# Save the figure
plt.savefig("pawnee_historical_suitability.png", dpi=300)
# Show the plot
plt.show()

# %% [markdown]
# # Pawnee Historical Plots: The Habitat Suitability Model results show 
# # that both the warm scenarios, no matter wet or dry, have similar 
# # suitability. And the two cold plots did not plot.
# 
# It appears I am continuing to have issues with some of the plots 
# plotting correctly. Similar to the Comanche future plots above this, 
# I am guessing either something went wrong in the climate model download 
# for the particular models chosen, or something went wrong with the merge 
# or the harmonizing.
# 
# For the two scenarios that did plot, the warm scenarios, have very 
# similar areas that are suitable for the Rocky Mountain Juniper that 
# are in the northwest and then a hill shaped area in the northeast that 
# kind of surrounds the upper portion of the right unit. Of the two 
# scenarios though, warm and wet seems to have slightly more pixels that 
# are suitable in both areas of the Pawnee Grassland.

# %%
# Pawnee Future Plots

# Create the plot grid (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 7))

# Titles corresponding to each subplot
titles = [
    "Pawnee Future Warm and Wet - Suitability", 
    "Pawnee Future Warm and Dry - Suitability", 
    "Pawnee Future Cold and Wet - Suitability", 
    "Pawnee Future Cold and Dry - Suitability"
]

# Plot each da in the respective subplot
for i, ax in enumerate(axes.flat):
    # Extract the da for the current scenario
    p_f_suitability = pawnee_f_combined_suitability_results[i]
    # Plot the suitability da
    p_f_suitability_plot = p_f_suitability.plot(ax=ax, cmap='PRGn', add_colorbar=False) 
    # Set the title for each subplot
    ax.set_title(titles[i])
    # Add colorbar to each subplot
    fig.colorbar(p_f_suitability_plot, ax=ax, orientation='vertical')
    # Add x and y axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # Overlay the boundary of the Pawnee Grassland study area
    pawnee_grassland_gdf.boundary.plot(ax=ax, color='orange', linewidth=1.5)

# Adjust layout to prevent overlap
plt.tight_layout()
# Save the figure
plt.savefig("pawnee_future_suitability.png", dpi=300)
# Show the plot
plt.show()

# %% [markdown]
# # Pawnee Future Plots: The Habitat Suitability Model results show 
# # that there is very little area in the cold-wet scenario for 
# # suitability. And the other 3 plots did not plot.
# 
# Again, similar to the above Pawnee Historical and the Comanche 
# Future plots, these also have issues. Likely for similar reasons; 
# it is concerning that so many didn't plot, but also confusing that 
# some did. So I will need to look into that. 
# 
# For the plot that did populate for the cold-wet scenario, there is 
# very little area that is suitable. It is difficult to compare to the 
# Pawnee Historical Plots because the cold plots didn't populate, but 
# doing some comparison to the Comanche future plots, which also only 
# populated the cold-wet, there is significantly more suitability in the 
# Comanche cold-wet scenario. The Comanche National Grassland is a bit 
# further south of the Pawnee National Grassland which may play a role 
# in this difference.

# %% [markdown]
# # Overall: Disappointed in plots, hard to draw many conclusions
# # with lack of complete plots.
# 
# I wish I had more time to figure this out, but because I work full 
# time in person, I likely would need a few days if not a week or so 
# to sort out the reasons why only some plots didn't work.
# 
# The code worked or ran which on the one hand is good, but makes it 
# difficult to know where something went wrong because all of it ran.


