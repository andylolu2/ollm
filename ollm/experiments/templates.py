from ollm.utils import load_template

_PROMPT_TEMPLATE = """Title: {{ title }}
{{ abstract }}"""
PROMPT_TEMPLATE = load_template(_PROMPT_TEMPLATE)

_PROMPT_TEMPLATE_FULL = """The following is an article's title and abstract. Your task is to assign this article to a suitable category hierarchy. \
A category is typically represented by a word or a short phrase, representing broader topics/concepts that the article is about. \
A category hierarchy is represented by a collection of paths from the generic root category "Main topic classifications" \
to a specific category suitable for the article. The topic titles should become more and more specific as you move from the root to the leaf. \

{% if examples|length > 0 %}
{% for example in examples %}
### EXAMPLE {{ loop.index }} ###
### ARTICLE ###
Title: {{ example['title'] }}
{{ example['abstract'] }}
### END ARTICLE ###
{% for path in example['paths'] %}
{{ path | join(" -> ") }}
{% endfor %}
### END EXAMPLE {{ loop.index }} ###
{% endfor %}
{% else %}
You must answer in the format of:
Main topic classifications -> Broad topic 1 -> Subtopic 1 -> ... -> Most specific topic 1
Main topic classifications -> Broad topic 2 -> Subtopic 2 -> ... -> Most specific topic 2
...
{% endif %}

### ARTICLE ###
Title: {{ title }}
{{ abstract }}
### END ARTICLE ###

Provide a category hierarchy for the above article. \
{% if examples|length > 0 %}
Use the same format as the examples above.
{% else %}
Use the format described above.
{% endif %}"""
PROMPT_TEMPLATE_FULL = load_template(_PROMPT_TEMPLATE_FULL)

_COT_ROUND_ONE_TEMPLATE = """The following is an article's title and abstract. \
Briefly break down the topics (both specific and general concepts) relevant to this article. Explain your reasoning step by step.

### ARTICLE ###
Title: {{ title }}
{{ abstract }}
### END ARTICLE ###
"""
COT_ROUND_ONE_TEMPLATE = load_template(_COT_ROUND_ONE_TEMPLATE)
_COT_ROUND_TWO_TEMPLATE = """Your task now is to assign this article to suitable category hierarchy. \
A category is typically represented by a word or a short phrase, representing broader topics/concepts that the article is about. \
A category hierarchy is represented by a collection of paths from the generic root category "Main topic classifications" \
to a specific category suitable for the article. The topics titles should become more and more specific as you move from the root to the leaf. \

You must answer in the format of:
Main topic classifications -> Broad topic 1 -> Subtopic 1 -> ... -> Most specific topic 1
Main topic classifications -> Broad topic 2 -> Subtopic 2 -> ... -> Most specific topic 2
...
"""
COT_ROUND_TWO_TEMPLATE = load_template(_COT_ROUND_TWO_TEMPLATE)

_PRED_CONCEPT_TEMPLATE = """The following is an article's title and abstract. \
Suggest the categories that this article belongs to. \
A category is typically represented by a word or a short phrase, representing broader topics/concepts that the article is about. \
You should include both general and specific categories.

### ARTICLE ###
Title: {{ title }}
{{ abstract }}
### END ARTICLE ###

Give your answer as a comma-separated string.
"""
PRED_CONCEPT_TEMPLATE = load_template(_PRED_CONCEPT_TEMPLATE)

_RESPONSE_TEMPLATE = """{% for path in paths %}
{{ path | join(" -> ") }}
{% endfor %}"""
RESPONSE_TEMPLATE = load_template(_RESPONSE_TEMPLATE)
RESPONSE_REGEX = r"(Main topic classifications( -> [\w()\-\–\—,.?!/\\&\"\'+=\[\]\{\} ]+)+\n)*(Main topic classifications( -> [\w()\-\–\—,.?!/\\&\"\'+=\[\]\{\} ]+)+)\n?"

_MISTRAL_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
MISTRAL_TEMPLATE = load_template(_MISTRAL_TEMPLATE)
