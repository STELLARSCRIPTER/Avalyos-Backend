import json
d = json.load(open("app/data/flood_analysis_results_historical.json"))
pts = d["step6_country_topology"]["country_points"]
print("countries:", len(pts))
print()
print("Sample - first 3:")
for p in pts[:3]:
    print(f"  {p['country']:<30} dmg={p['mean_damage_m']:.2f} deaths={p['mean_deaths']:.2f} driver={p['dominant_driver']}")
print()
print("Meta:", d.get("meta"))
