import asyncio
import json
from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from signal import SIGTERM, signal
from textwrap import dedent
from typing import Any, Iterable, Mapping

import httpx
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rich import live, print
from rich.spinner import Spinner

from gdpr_stats.ai import ask_claude, parse_chat_output

from .errors import GdprStatsError


def sigterm_handler(_, __):
    """
    Handle SIGTERM signal to gracefully exit the program.

    This function exists to ensure proper cleanup and termination when the
    program receives a SIGTERM signal.
    """

    raise SystemExit(1)


def retry(tries: int = 3):
    """
    Decorator to retry a function execution in case of ValueError.

    This decorator exists to improve the robustness of functions that may fail
    due to LLMs approximating their outputs.

    Parameters
    ----------
    tries
        Number of retry attempts, by default 3
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(tries):  # noqa: RET503
                try:
                    return await func(*args, **kwargs)
                except ValueError:
                    if i == tries - 1:
                        raise
                except:
                    raise

        return wrapper

    return decorator


def deep_as_dict(obj: Any, cache: dict[int, Any] | None = None) -> dict:
    """
    Convert complex nested objects into a dictionary representation.

    This function exists to facilitate serialization of complex data
    structures, especially those containing dataclasses or custom objects with
    circular references.

    Parameters
    ----------
    obj
        The object to convert
    cache
        Cache to keep references to objects already seen (internal parameter)
    """

    if cache is None:
        cache = {}

    if id(obj) in cache:
        return cache[id(obj)]

    if is_dataclass(obj):
        out = {}
        cache[id(obj)] = out
        out.update(
            {f.name: deep_as_dict(getattr(obj, f.name), cache) for f in fields(obj)}
        )
    elif isinstance(obj, str):
        out = obj
    elif isinstance(obj, Mapping):
        out = {}
        cache[id(obj)] = out
        out.update({k: deep_as_dict(v, cache) for k, v in obj.items()})
    elif isinstance(obj, Iterable):
        out = [deep_as_dict(x, cache) for x in obj]
    else:
        out = obj

    cache[id(obj)] = out

    return out


@contextmanager
def action(title: str):
    """
    Context manager to display progress and status of an action.

    This function exists to provide visual feedback to the user about the
    progress of long-running operations.

    Parameters
    ----------
    title
        Description of the action being performed
    """

    try:
        with live.Live(Spinner("dots2", title), refresh_per_second=12, transient=True):
            yield

        print(f"✅ {title}")
    except Exception:
        print(f"❌ {title}")
        raise


@dataclass
class Args:
    """
    Dataclass to store command-line arguments.

    This class exists to provide a structured way to handle and pass around
    command-line arguments throughout the program.
    """

    source: str
    destination: Path


@dataclass
class PrimaryEntry:
    """
    Dataclass to store primary parsed data from the source.

    This class exists to represent the initial structured data extracted from
    the raw source, before further processing.
    """

    date: str
    org_type: str
    main_themes: list[str]
    decision: str


@dataclass
class ThemeCategory:
    """
    Dataclass to represent a theme category and its subcategory.

    This class exists to provide a structured representation of theme
    hierarchies.
    """

    category: str
    sub_category: str


def parse_args(argv: list[str] | None = None) -> Args:
    """
    Parse command-line arguments.

    This function exists to handle user input and configure the program's
    behavior based on command-line options.

    Parameters
    ----------
    argv
        List of command-line arguments, by default the standard ones
    """

    parser = ArgumentParser()

    parser.add_argument(
        "-s",
        "--source",
        help="Source page to parse",
        default="https://www.cnil.fr/fr/les-sanctions-prononcees-par-la-cnil",
    )

    parser.add_argument(
        "destination",
        help="Destination YAML file",
        type=Path,
    )

    return Args(**vars(parser.parse_args(argv)))


def primary_parse(content: str) -> list[PrimaryEntry]:
    """
    Perform initial parsing of the source content.

    This function exists to extract structured data from the raw HTML content
    of the source page, but that still needs to be refined and processed.

    Parameters
    ----------
    content : str
        Raw HTML content of the source page
    """

    soup = BeautifulSoup(content, "html.parser")
    out = []

    for table in soup.find_all("table"):
        if not table.select("caption h2") and not table.find_parent(
            lambda tag: "ctn-gen-ascenseur-texte" in tag.get("class", [])
        ):
            continue

        for row in table.select("tr"):
            cells = [*row.select("td")]

            if len(cells) != 4:
                continue

            el_date, el_org_type, el_main_themes, el_decision = cells

            date = el_date.text.strip()
            org_type = el_org_type.text.strip()
            main_themes = [
                x.strip() for x in el_main_themes.text.splitlines() if x.strip()
            ]
            decision = el_decision.text.strip()

            out.append(PrimaryEntry(date, org_type, main_themes, decision))

    return out


@dataclass
class SubCategory:
    """
    Dataclass to represent a subcategory within a theme category.

    This class exists to provide a structured representation of theme
    subcategories.
    """

    name: str
    slug: str
    category: "Category" = field(repr=False)


@dataclass
class Category:
    """
    Dataclass to represent a theme category.

    This class exists to provide a structured representation of theme
    categories, including their subcategories.
    """

    name: str
    slug: str
    sub_categories: dict[str, SubCategory]


@dataclass
class Themes:
    """
    Dataclass to represent the entire theme structure.

    This class exists to provide a comprehensive view of all themes,
    categories, and subcategories in the dataset.
    """

    categories: dict[str, Category]
    themes: dict[str, SubCategory]


def hydrate_themes(all_themes, parsed: Any) -> Themes:
    """
    Convert parsed theme data into a structured Themes object.

    This function exists to transform the raw parsed theme data coming from the
    LLM, validate the output's consistency and convert it into a structure that
    will be convenient to use later on.

    Parameters
    ----------
    all_themes
        Raw theme data
    parsed
        Parsed theme structure
    """

    sub = {}
    cat = {}
    t = {}

    themes_by_id = {t["id"]: t["text"] for t in all_themes}

    match parsed:
        case {"categories": categories, "themes": themes}:
            if not isinstance(categories, dict):
                msg = "categories should be a dict"
                raise ValueError(msg)

            if not isinstance(themes, list):
                msg = "themes should be a list"
                raise ValueError(msg)

            _hydrate_themes_cat(cat, categories, sub)
            _hydrate_themes_themes(sub, t, themes, themes_by_id)
        case _:
            msg = "Invalid themes"
            raise ValueError(msg)

    return Themes(cat, t)


def _hydrate_themes_themes(sub, t, themes, themes_by_id):
    for theme in themes:
        match theme:
            case {"initial_id": initial_id, "sub_category": sub_category}:
                if sub_category not in sub:
                    msg = f"Unknown sub_category: {sub_category}"
                    raise ValueError(msg)

                if initial_id not in themes_by_id:
                    msg = f"Unknown initial_id: {initial_id}"
                    raise ValueError(msg)

                t[themes_by_id[initial_id]] = sub[sub_category]
            case _:
                msg = "Invalid themes"
                raise ValueError(msg)


def _hydrate_themes_cat(cat, categories, sub):
    for cat_slug, cat_info in categories.items():
        local_sub = {}

        match cat_info:
            case {"name": name, "sub_categories": sub_categories}:
                if not isinstance(sub_categories, dict):
                    msg = "sub_categories should be a dict"
                    raise ValueError(msg)

                cat[cat_slug] = Category(name, cat_slug, local_sub)

                for sub_slug, sub_cat in sub_categories.items():
                    if not isinstance(sub_cat, str):
                        msg = "sub_cat should be a string"
                        raise ValueError(msg)

                    local_sub[sub_slug] = SubCategory(sub_cat, sub_slug, cat[cat_slug])

                    if sub_slug in sub:
                        msg = f"Duplicate sub-category: {sub_slug}"
                        raise ValueError(msg)

                    sub[sub_slug] = local_sub[sub_slug]
            case _:
                msg = "Invalid category"
                raise ValueError(msg)


@retry()
async def generate_themes(entries: list[PrimaryEntry]) -> Themes:
    """
    Generate a structured theme hierarchy from primary entries.

    This function exists to create a meaningful categorization of themes
    based on the raw data extracted from the source.

    Parameters
    ----------
    entries
        List of primary parsed entries
    """

    all_themes = [
        dict(id=i, text=t)
        for i, t in enumerate(sorted({t for e in entries for t in e.main_themes}))
    ]

    system = dedent(
        """
        You receive a JSON list of themes, that you need to group by
        category and sub-category. Sub-categories are close to the initial text
        that you receive, but you can chose to group together entries that
        are very similar. The category on the other hand should group broad
        concepts of sub-categories together. These are all related to GDPR, so
        use your general knowledge of GDPR to build decent categories.

        You must be exhaustive in your output, and I expect to see the YAML and
        only the YAML, no comments around it.

        All texts except the "initial_text" should be in English.

        The expected output is a YAML file with the following format:

        categories:
            cat-slug:
                name: Category name
                sub_categories:
                    sub-cat-slug: Sub-category name
                    other-sub-cat-slug: Other sub-category name
            other-cat-slug:
                name: Other category name
                sub_categories:
                    sub-cat-slug: Sub-category name
                    other-sub-cat-slug: Other sub-category name
        themes:
            - initial_id: 12
              sub_category: sub-cat-slug
            - initial_id: 8
              sub_category: other-sub-cat-slug
        """
    ).strip()

    chat = await ask_claude(system, json.dumps(list(all_themes)))

    return hydrate_themes(all_themes, parse_chat_output(chat))


@dataclass
class OrgInfo:
    """
    Dataclass to store information about an organization.

    This class exists to provide a structured representation of organization
    data.
    """

    name: str
    type: str
    simplified: bool


def hydrate_company_names(all_names, parsed: Any) -> dict[str, OrgInfo]:
    """
    Convert parsed company name data into a structured dictionary.

    This function exists to transform raw LLM output into a more usable
    structure, also taking care to validate its consistency.

    Parameters
    ----------
    all_names
        Raw company name data
    parsed
        Parsed company name structure
    """

    out = {}
    name_by_id = {t["id"]: t["text"] for t in all_names}

    match parsed:
        case [*items]:
            for item in items:
                match item:
                    case {
                        "initial_id": initial_id,
                        "type": org_type,
                        "simplified": simplified,
                    }:
                        if initial_id not in name_by_id:
                            msg = f"Unknown initial_id: {initial_id}"
                            raise ValueError(msg)

                        if not isinstance(simplified, bool):
                            msg = "simplified should be a boolean"
                            raise ValueError(msg)

                        name = name_by_id[initial_id]
                        out[name] = OrgInfo(name, org_type, simplified)
                    case _:
                        msg = "Invalid item"
                        raise ValueError(msg)
        case _:
            msg = "Invalid output"
            raise ValueError(msg)

    return out


@retry()
async def generate_company_names(entries: list[PrimaryEntry]) -> dict[str, OrgInfo]:
    """
    Generate structured company name data from primary entries.

    This function exists to create a comprehensive list of organizations
    mentioned in the dataset, including some metadata that is written in plain
    text in the source.

    Parameters
    ----------
    entries
        List of primary parsed entries
    """

    all_names = [
        dict(id=i, text=t) for i, t in enumerate(sorted({e.org_type for e in entries}))
    ]

    system = dedent(
        """
        You receive a JSON list of organization types, alongside with the
        information of whether they got through a simplified procedure or not.
        Your job is to transform this into a YAML format.

        You must be exhaustive in your output, and I expect to see the YAML and
        only the YAML, no comments around it.

        Please translate all organization types to English.

        The expected output is a YAML file with the following format:

        - initial_id: 12
          type: Organization type
          simplified: true
        - initial_id: 8
          type: Other organization type
          simplified: false
        """
    ).strip()

    chat = await ask_claude(system, json.dumps(list(all_names)))

    return hydrate_company_names(all_names, parse_chat_output(chat))


@dataclass
class SecondaryEntry:
    """
    Dataclass to store secondary parsed data.

    This class exists to represent the fully processed and structured data
    after all parsing and processing steps.
    """

    date: str
    org_type: str
    simplified: bool
    themes: list[SubCategory]
    amount: float


@retry()
async def parse_amounts(amounts: list[str]) -> list[float]:
    """
    Since the input hides amounts in plain text sentences, sometimes not even
    being written as numbers, we use a LLM to convert all those items into
    float values.

    Parameters
    ----------
    amounts
        List of strings potentially containing monetary amounts
    """

    system = dedent(
        """
        You will see a list of strings that might mention amounts of money. For
        each string:

        - If it mentions an amount of money, reply with the float
          representation of that amount.
        - If it doesn't mention an amount of money, reply with zero.

        Provide your answers as a JSON list of numbers, with no additional
        comments.

        For example: [100000.0, 0.0, 50000.0, 75000.0]
        """
    ).strip()

    parsed = await ask_claude(system, json.dumps(amounts))

    try:
        return json.loads(parsed.strip())
    except json.JSONDecodeError:
        msg = "Invalid JSON response from Claude"
        raise ValueError(msg) from None


async def secondary_parse(
    entries: list[PrimaryEntry],
    themes: Themes,
    company_names: dict[str, OrgInfo],
) -> list[SecondaryEntry]:
    """
    Perform secondary parsing to create fully structured entries.

    This function exists to combine all parsed and processed data into a final,
    comprehensive dataset.

    Parameters
    ----------
    entries
        List of primary parsed entries
    themes
        Structured theme hierarchy
    company_names
        Dictionary of company names to OrgInfo objects
    """

    out = []

    decisions = [e.decision for e in entries]
    parsed_amounts = await parse_amounts(decisions)
    amount_dict = dict(zip(decisions, parsed_amounts))

    for entry in entries:
        org_type = company_names[entry.org_type]
        _themes = []

        for t in entry.main_themes:
            if t not in themes.themes:
                msg = f"Unknown theme: {t}"
                raise ValueError(msg)

            _themes.append(themes.themes[t])

        out.append(
            SecondaryEntry(
                date=datetime.strptime(f"{entry.date}Z", "%d/%m/%Y%z")
                .date()
                .isoformat(),
                org_type=org_type.type,
                simplified=org_type.simplified,
                themes=_themes,
                amount=amount_dict[entry.decision],
            )
        )

    return out


async def main(argv: list[str] | None = None):
    """
    Main function to orchestrate the entire data processing pipeline.

    This function exists to coordinate all steps of data fetching, parsing,
    processing, and output generation.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments, by default the standard ones
    """

    args = parse_args(argv)
    http = httpx.AsyncClient()

    with action("Fetching source page"):
        resp = await http.get(args.source)
        resp.raise_for_status()

    with action("Primary parsing"):
        content = resp.text
        entries = primary_parse(content)

    with action("Generate company names"):
        company_names = await generate_company_names(entries)

    with action("Generate themes"):
        themes = await generate_themes(entries)

    with action("Secondary parse"):
        secondary_entries = await secondary_parse(entries, themes, company_names)

    with action("Writing output"):
        out = deep_as_dict(
            dict(
                themes=themes.categories,
                entries=secondary_entries,
            )
        )

        with args.destination.open("w") as f:
            yaml.safe_dump(out, f)


def __main__():
    """
    Entry point for the script when run as a module.

    This function exists to handle the script's execution, including
    environment setup and error handling.

    If you want to call this script from another Python module, you can call
    main() directly to avoid having to fight our signal handling and so forth.
    """

    signal(SIGTERM, sigterm_handler)
    load_dotenv()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("ok, bye")
        exit(1)
    except GdprStatsError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    __main__()
