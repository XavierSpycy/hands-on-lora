PREFIX = 'You are an expert in the naming entity recognition domain. You have to extract entities from a given sentence, assign the labels to the corresponding entities in json format. Entity labels contain:\n'
ENTITY_LABELS = ['B-NAME_STUDENT', 'I-NAME_STUDENT', 'O']
EXPLANATION = '\nToken labels are presented in BIO (Beginning, Inner, Outer) format. The PII type is prefixed with “B-” when it is the beginning of an entity. If the token is a continuation of an entity, it is prefixed with “I-”. Tokens that are not PII are labeled “O”.\n'
JSON_SCHEMA = '{"token": "Nathalie", "label": "B-NAME_STUDENT"}'
NOTE = 'NOTE: 1. Make sure each line of the outputs must be a correct json string, such as: {}. 2. If no entities are found, return "There are no entities."'

DEFAULT_SYSTEM_PROMPT = PREFIX + ", ".join(ENTITY_LABELS) + EXPLANATION + NOTE.format(JSON_SCHEMA)