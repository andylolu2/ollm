import jinja2

env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def load_template(s: str):
    s = s.lstrip("\n")
    return env.from_string(s)
