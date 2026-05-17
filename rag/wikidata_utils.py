import requests

# Gold properties: properties that are more likely to be meaningful in a QA task for a given entity (manually selected from top-100 used properties in Wikidata). 
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

# Fetch the Wikidata entity information for a given Wikidata ID, including label, description, aliases, and selected properties (if 'id_only=False').
def get_wikidata_entity(wikidata_ids, id_only=False):
    # Check if if the function receives a single Wikidata ID or more, and convert it to a list if necessary.
    if isinstance(wikidata_ids, str):
        wikidata_ids = [wikidata_ids]
    elif isinstance(wikidata_ids, set):
        wikidata_ids = list(wikidata_ids)
    
    # Call Wikidata API to fetch entity information for the given Wikidata IDs.
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(wikidata_ids),
        "format": "json",
        "languages": "en",
        "props": "labels|descriptions|aliases|claims" # Retrieve labels, descriptions, aliases, and claims (properties) for the entities.
    }
    headers = {
        "User-Agent": "NLPHomework Bot (lorito.1885657@studenti.uniroma1.it)" 
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        print(f"Error fetching Wikidata entity: {data['error']}")
        return None
    
    # If 'id_only' is True, return a dictionary mapping Wikidata IDs to their labels.
    if id_only:
        fetched_dict = {}
        for wikidata_id in wikidata_ids:
            entity = data.get("entities", {}).get(wikidata_id, {})
            label = entity.get("labels", {}).get("en", {}).get("value", wikidata_id)
            fetched_dict[wikidata_id] = label
        return fetched_dict

    # Otherwise, return a detailed string with the entity's information and selected properties.
    wikidata_entity_info = ""

    for wikidata_id in wikidata_ids:
        # Get the entity's label, description, and aliases.
        entity = data.get("entities", {}).get(wikidata_id, {})
        label = entity.get("labels", {}).get("en", {}).get("value", "Unknown label")
        description = entity.get("descriptions", {}).get("en", {}).get("value", "Unknown description")
        aliases = entity.get("aliases", {}).get("en", [])
        aliases = [alias.get("value") for alias in aliases] if aliases else []
        
        # Add the basic information about the entity to the result string.
        wikidata_entity_info += f"Wikidata Information: {label} - {description}; Aliases: {', '.join(aliases) if aliases else 'None'}"

        # If 'id_only' is False, also fetch and include selected properties of the entity in the result string.
        if not id_only:
            properties = get_entity_properties(wikidata_id, entity)
            for prop in properties:
                if "entity" in prop:
                    wikidata_entity_info += f"; {prop['property']}: {prop['entity']}"
                elif "value" in prop:
                    wikidata_entity_info += f"; {prop['property']}: {prop['value']}"

    if response.status_code == 200:
        return wikidata_entity_info
    else:
        print(f"Failed to fetch Wikidata entity. Status code: {response.status_code}")
        return None    

# Get selected properties of a Wikidata entity, prioritizing "gold" properties (limit of retrived properties for each entity is set to 10 for bounding the context enrichment in any case).
def get_entity_properties(wikidata_id, entity):
    claims = entity.get("claims", {})
    selected_properties= []
    ids_to_fetch = set() # Set of ids that needs to be fetched later to retrieve their textual information (e.g., linked entities, non-gold properties, etc.).

    # We fetch properties until getting at most 10 gold ones.
    selected_properties, ids_to_fetch = fetch_property(claims, selected_properties, ids_to_fetch, needs_gold=True) # We first look for the gold properties that are most likely to contain the answer.
        
    # If there are less than 10 selected gold properties, we still look for other properties.
    if len(selected_properties) < 10: 
        selected_properties, ids_to_fetch = fetch_property(claims, selected_properties, ids_to_fetch, needs_gold=False) # We look for additional properties beyond the gold ones to enrich the context.

    # If there are any IDs to fetch, we do so to retrieve their textual information.
    if ids_to_fetch:
        # We fetch the entities for all the linked entities ids to retrieve their elements.
        fetched_ids = get_wikidata_entity(list(ids_to_fetch), id_only=True)
        
        # We substitute the entity's or property's id with the fetched element.
        for prop in selected_properties: 
            if "val_id" in prop and prop["val_id"] in ids_to_fetch:
                prop["entity"] = fetched_ids.get(prop["val_id"])

            if prop["property"] in fetched_ids:
                prop["property"] = fetched_ids[prop["property"]]

    return selected_properties

# Fetch selected properties of a Wikidata entity (based on its claims) and return a list of dictionaries with the property name and its value (either an entity or a literal value).
def fetch_property(claims, selected_properties, ids_to_fetch, needs_gold=False):
    for prop_id, prop_value in claims.items():
        # We stop fetching properties if we have already selected 10 of them.
        if len(selected_properties) >= 10:
            break
        # Check if the property is a gold property (if 'needs_gold' is True) or a non-gold property (if 'needs_gold' is False) and fetch its value accordingly.
        if (needs_gold and prop_id in WIKIDATA_GOLD_PROPERTIES) or (not needs_gold and prop_id not in WIKIDATA_GOLD_PROPERTIES):
            snak = claims[prop_id][0].get("mainsnak", {}) # We take the first mainsnak of the property, as it usually contains the most relevant value for our task (mainsnak is the tuple (property_id, value) of a property).
            if snak.get("snaktype") == "value":
                datatype = snak.get("datatype")

                # Avoids fetching properties with datatypes that are not properly meaningful in textual context (e.g., media, URLs, geo-shapes, etc.).
                if datatype in ["commonsMedia", "external-id", "url", "math", "geo-shape", "tabular-data", "musical-notation"]:
                    continue

                prop_name = WIKIDATA_GOLD_PROPERTIES.get(prop_id, prop_id) # If it's a gold property, we use its name, otherwise we keep the id as name.
                if prop_id not in WIKIDATA_GOLD_PROPERTIES:
                    ids_to_fetch.add(prop_id)

                val_type = snak.get("datavalue", {}).get("type")
                val_data = snak.get("datavalue", {}).get("value")

                # Add to 'ids_to_fetch' the id of the value if it's an entity.
                if val_type == "wikibase-entityid":
                    val_id = val_data.get("id")
                    selected_properties.append({"property": prop_name, "val_id": val_id})
                    ids_to_fetch.add(val_id)
                else:
                    # If it's a literal value, we check if it's a dictionary with a specific key (e.g., "amount", "text", "time") and we extract the relevant information or we convert it to string.
                    if isinstance(val_data, dict):
                        if "amount" in val_data:
                            val_data = val_data["amount"].replace("+", "")
                        elif "text" in val_data:
                            val_data = val_data["text"]
                        elif "time" in val_data:
                            val_data = val_data["time"].split("T")[0].replace("+", "")
                        else:
                            val_data = str(val_data)
                    selected_properties.append({"property": prop_name, "value": val_data})

    return selected_properties, ids_to_fetch

def get_wikidata_ground_truth(wikidata_id, short_answer):
    answers = set()
    answers.add(short_answer.strip().lower())

    url = "https://www.wikidata.org/w/api.php"
    headers = {
        "User-Agent": "NLPHomework Bot (lorito.1885657@studenti.uniroma1.it)"
    }

    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "format": "json",
        "languages": "en",
        "props": "claims"
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    entity = data.get("entities", {}).get(wikidata_id, {})

    properties = get_entity_properties(wikidata_id, entity)

    match = None

    for prop in properties:
        if "entity" in prop:
            if  short_answer.strip().lower() in prop["entity"].strip().lower() or prop["entity"].strip().lower() in short_answer.strip().lower():
                match = prop.get("val_id")
                break

    if match:
        params = {
            "action": "wbgetentities",
            "ids": match,
            "format": "json",
            "languages": "en",
            "props": "labels | alises"
        }

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        entity = data.get("entities", {}).get(match, {})
        
        label = entity.get("labels", {}).get("en", {}).get("value", None)
        if label and label not in answers:
            answers.add(label.strip().lower())

        aliases = entity.get("aliases", {}).get("en", [])
        for alias in aliases:
            alias_value = alias.get("value", "").strip().lower()
            if alias_value and alias_value not in answers:
                answers.add(alias_value)

    return answers