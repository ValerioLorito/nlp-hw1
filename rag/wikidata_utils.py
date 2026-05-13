import requests

WIKIDATA_GOLD_PROPERTIES = {
    "P31": "instance of",
    "P248": "stated in",
    "P17": "country",
    "P585": "point in time",
    "P106": "occupation",
    "P21": "sex or gender",
    "P9259": "epoch",
    "P569": "date of birth",
    "P582": "end time",
    "P27": "country of citizenship",
    "P734": "family name",
    "P361": "part of",
    "P279": "subclass of",
    "P276": "location",
    "P1082": "population",
    "P19": "place of birth",
    "P570": "date of death",
    "P69": "educated at",
    "P155": "follows",
    "P156": "followed by",
    "P495": "country of origin",
    "P527": "has part(s)",
}

def get_wikidata_entity(wikidata_id, language="en"):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "format": "json",
        "languages": language,
        "props": "labels|descriptions|aliases|claims"
    }
    headers = {
        "User-Agent": "NLPHomework Bot (lorito.1885657@studenti.uniroma1.it)" 
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        return None
    
    entity = data.get("entities", {}).get(wikidata_id, {})
    label = entity.get("labels", {}).get(language, {}).get("value", "Unknown label")
    description = entity.get("descriptions", {}).get(language, {}).get("value", "Unknown description")
    aliases = entity.get("aliases", {}).get(language, [])
    aliases = [alias.get("value") for alias in aliases] if aliases else []
    properties = get_entity_properties(wikidata_id, entity)

    wikidata_entity_info = f"Wikidata Information: {label} - {description}; Aliases: {', '.join(aliases) if aliases else 'None'}"

    for prop in properties:
        if "val_label" in prop:
            wikidata_entity_info += f"; {prop['property']}: {prop['val_label']}"
        elif "value" in prop:
            wikidata_entity_info += f"; {prop['property']}: {prop['value']}"

    print(f"\n--- WIKIDATA ENTITY INFO RETRIEVED FOR {wikidata_id} ---")
    print(wikidata_entity_info)
    print("-------------------------------------------------------------------\n")

    if response.status_code == 200:
        return wikidata_entity_info
    else:
        return None

def get_entity_properties(wikidata_id, entity):
    claims = entity.get("claims", {})
    selected_properties= []
    ids_to_fetch = set()

    for prop_id, prop_value in WIKIDATA_GOLD_PROPERTIES.items():
        # Iterate over gold properties to find ones present in the entity's claims.
        if len(selected_properties) >= 10: # We limit to 10 properties to avoid overwhelming the model with information.
            break
        if prop_id in claims: # If a gold property is in the claims
            snak = claims[prop_id][0].get("mainsnak", {}) # We take the mainsnak (property id and values) 
            if snak.get("snaktype") == "value":
                val_type = snak.get("datavalue", {}).get("type")
                val_data = snak.get("datavalue", {}).get("value")
                if val_type == "wikibase-entityid": # We check if the value is another Wikidata entity
                    val_id = val_data.get("id")
                    selected_properties.append({"property": prop_value, "val_id": val_id})
                    ids_to_fetch.add(val_id) 
                else:
                    selected_properties.append({"property": prop_value, "value": val_data})
        
    ids_to_fetch = fetch_ids(ids_to_fetch) # We fetch the labels for all the linked entities in one go to minimize API calls and speed up the process.
    
    for prop in selected_properties: # We substitute the val_id with the fetched label
        if "val_id" in prop and prop["val_id"] in ids_to_fetch:
            prop["val_label"] = ids_to_fetch.get(prop["val_id"])

    return selected_properties
    
def fetch_ids(ids_to_fetch):
    connected_ids = {}
    if ids_to_fetch:
        params = {
            "action": "wbgetentities",
            "ids": "|".join(ids_to_fetch),
            "format": "json",
            "languages": "en",
            "props": "labels"
        }
        response = requests.get("https://www.wikidata.org/w/api.php", params=params, headers={"User-Agent": "NLPHomework Bot (lorito.1885657@studenti.uniroma1.it)"})
        
        for id, data in response.json().get("entities", {}).items():
            connected_ids[id] = data.get("labels", {}).get("en", {}).get("value", id)
            
    return connected_ids

