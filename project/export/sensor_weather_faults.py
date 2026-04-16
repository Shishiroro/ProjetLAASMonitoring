"""
sensor_weather_faults.py — Effets meteo post-traitement sur capteur camera
==========================================================================
Simule les degradations visuelles causees par la meteo sur la lentille
de la camera embarquee (pluie sur lentille, condensation, givre, etc.).

Appliques en post-traitement (OpenCV) apres la capture X-Plane,
independamment des effets meteo injectes dans X-Plane (xplane_weather.py).

TODO: A implementer. Types prevus :
  - rain_on_lens    : gouttes de pluie sur la lentille
  - condensation    : buee sur la lentille (existe dans sensor_faults, a migrer)
  - frost_on_lens   : givre / glace sur la lentille
  - dirt_on_lens    : salissure (existe dans sensor_faults, a migrer)
  - fog_on_lens     : brouillard proche lentille (existe dans sensor_faults, a migrer)

Meme pattern que sensor_faults.py : severity + from_pct + to_pct.
"""
