Surface
=======

Leaf area index :math:`LAI` (-):

.. math::
  LAI=\frac{S_{int-tot}}{0.2}

Fraction of ground cover :math:`f_{ground-cover}` (-):

.. math::
  f_{ground-cover} = 1 - 0.7^{LAI}

Properties of upper interception storage
----------------------------------------



Properties of lower interception storage
----------------------------------------

Storage capacitiy (i.e. available storage volume) of lower interception storage
:math:`S_{int-lower-tot}` (mm) is land use dependent and varies seasonally. For example,
parameters used for grassland (lu_id=8):

.. _tbl-grid:

+-----------------------+------+------+------+------+------+------+------+------+------+------+------+------+
|                       | Jan  | Feb  | Mar  | Apr  | May  | Jun  | Jul  | Aug  | Sep  | Oct  | Nov  | Dec  |
+=======================+======+======+======+======+======+======+======+======+======+======+======+======+
| ground_cover [-]      | 0.51 | 0.51 | 0.66 | 0.76 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 | 0.66 | 0.59 | 0.51 |
+-----------------------+------+------+------+------+------+------+------+------+------+------+------+------+
| z_root [mm]           | 400  | 400  | 400  | 400  | 400  | 400  | 400  | 400  | 400  | 400  | 400  | 400  |
+-----------------------+------+------+------+------+------+------+------+------+------+------+------+------+
| lai [-]               | 2    | 2    | 3    | 4    | 5    | 5    | 5    | 5    | 5    | 3    | 2.5  | 2    |
+-----------------------+------+------+------+------+------+------+------+------+------+------+------+------+
| S_int_lower_tot [mm]  | 0.4  | 0.4  | 0.6  | 0.8  | 1    | 1    | 1    | 1    | 1    | 0.6  | 0.5  | 0.4  |
+-----------------------+------+------+------+------+------+------+------+------+------+------+------+------+


Short description of land use (lu_id)
-------------------------------------
- `0`: sealed surface
- `5`: arable land
- `501`: bean
- `502`: amaranth
- `503`: other commercial crops
- `504`: artichoke
- `505`: berry
- `506`: ornamental plant
- `507`: nettle
- `508`: buckwheat
- `509`: pea
- `510`: strawberry
- `511`: esparcet
- `512`: sunflower
- `513`: vegetables
- `514`: flax
- `515`: early potatoes
- `516`: fodder root crops
- `517`: fodder legumes
- `518`: hemp
- `519`: home garden
- `520`: hop
- `521`: legumes
- `522`: intensive fruit-growing
- `523`: potato
- `524`: to be determined
- `525`: grain corn
- `526`: herbs
- `527`: false flax
- `528`: lentil
- `529`: lupine
- `530`: lucerne
- `531`: summer phacelia
- `532`: flat pea
- `533`: grape
- `534`: grape school
- `535`: rhubarb
- `536`: beetroot
- `537`: nuts
- `538`: summer mustard
- `539`: silage corn
- `540`: silphium
- `541`: soybean
- `542`: summer barley
- `543`: summer wheat
- `544`: summer oat
- `545`: summer rape
- `546`: summer triticale
- `547`: sunflower
- `548`: other fruit-growing
- `549`: sorghum
- `550`: asparagus (growing only)
- `551`: asparagus (continued winter)
- `552`: asparagus (continued summer)
- `553`: tobacco
- `554`: helianthus
- `555`: vetch
- `556`: winter barley
- `557`: winter wheat
- `558`: winter oat
- `559`: winter rape
- `560`: winter triticale
- `561`: chicory
- `562`: sweet corn
- `563`: sugar beet
- `564`: winter green manure (Oct)
- `565`: summer grass
- `566`: winter grass
- `567`: clover
- `568`: winter phacelia
- `569`: winter green manure (Aug)
- `570`: winter green manure (Sep)
- `571`: summer grass (growing only)
- `572`: winter grass (growing only)
- `573`: summer grass (continued)
- `574`: winter grass (continued)
- `575`: summer faba bean
- `576`: winter faba bean
- `577`: summer grain pea
- `578`: winter grain pea
- `579`: winter rye
- `580`: summer clover (growing only)
- `581`: summer clover (continued winter)
- `582`: summer clover (continued summer)
- `583`: winter clover (growing only)
- `584`: winter clover (continued summer)
- `585`: winter clover (continued winter)
- `586`: yellow mustard (after wheat)
- `587`: yellow mustard (after corn)
- `588`: yellow mustard (summer)
- `589`: miscanthus (growing only)
- `590`: miscanthus (continued winter)
- `591`: miscanthus (continued summer)
- `598`: no crop
- `599`: bare
- `6`: vineyard
- `7`: orchard
- `8`: grass
- `9`: complex plot
- `10`: deciduous forest
- `11`: mixed forest
- `12`: coniferous forest
- `13`: wetland
- `14`: lake
- `15`: forest (unknown tree species)
- `16`: urban tree
- `17`: custom land cover including trees
- `20`: river
- `31`: gravel rooftop
- `32`: grass rooftop extensive
- `33`: grass rooftop intensive
- `41`: gravel
- `50`: percolation plant
- `60`: custom land cover
- `98`: grass intensive
- `100`: urban
- `999`: no value
